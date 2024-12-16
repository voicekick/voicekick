use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use candle_core::{Device, IndexOp, Shape, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::{
    api::sync::{Api, ApiError},
    Repo, RepoType,
};
use inference_candle::{
    inference_device,
    proto::{self, DecodingResult, Segment},
    InferenceError, InferenceResult, SpeechRecognitionDecoder, SpeechRecognitionModel,
};
use tokenizers::Tokenizer;
use tracing::debug;

mod audio;
mod multilingual;

const ZERO: f32 = 0.0;

pub mod audio_params {
    // Sample rate
    pub const SAMPLE_RATE: usize = 16000;

    /// For each FFT operation:
    /// - Takes 400 samples window
    /// - Overlaps by (400-128) = 272 samples with previous window
    /// - Gives ~75% overlap between consecutive windows
    /// - Results in 16000/128 ≈ 125 frames per second
    /// Each window:
    /// |----400 samples----|
    ///       |----400 samples----|
    ///             |----400 samples----|
    /// Moving by 128 samples each time
    pub const N_FFT: usize = 400;

    /// The overlap (N_FFT - HOP_LENGTH) helps capture smooth transitions and
    /// transient features in the audio signal.
    ///
    /// Original: (400-128)/400 = 68% overlap
    /// - 200;  // 50% overlap
    /// - 300;  // 25% overlap
    ///
    /// Tradeoff: larger HOP_LENGTH = faster processing but may miss audio transitions
    /// Minimum recommended overlap is 25%
    pub const HOP_LENGTH: usize = 160;

    /// For Whisper, preferred max segment sizes in mel spectrogram frames:
    /// - Optimal: 1500 frames (~30 seconds of audio)
    /// - Maximum supported: 3000 frames (~60 seconds)
    /// - Real-time recommended: 800-1000 frames (~16-20 seconds)
    pub const N_FRAMES_MAX: usize = 3000;

    pub const NO_SPEECH_THRESHOLD: f64 = 0.7;
    pub const LOGPROB_THRESHOLD: f64 = -1.5;

    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0. This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
    // pub const TEMPERATURES: [f64; 7] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    pub const TEMPERATURES: [f64; 1] = [0.0];
    // pub const TEMPERATURES: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    TinyEn,
    BaseEn,
    SmallEn,
    MediumEn,

    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,

    DistilMediumEn,
    DistilLargeV2,
    DistilLargeV3,

    QuantizedTiny,
    QuantizedTinyEn,
}

impl Default for WhichModel {
    fn default() -> Self {
        Self::TinyEn
    }
}

impl WhichModel {
    fn is_quantized(&self) -> bool {
        match self {
            Self::QuantizedTiny | Self::QuantizedTinyEn => true,
            _ => false,
        }
    }

    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
            Self::QuantizedTiny => true,
            Self::QuantizedTinyEn => false,
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
            Self::QuantizedTiny | Self::QuantizedTinyEn => ("lmz/candle-whisper", "main"),
        }
    }
}

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Model {
    fn config(&self) -> &Config {
        match self {
            Model::Normal(model) => &model.config,
            Model::Quantized(model) => &model.config,
        }
    }
}

pub fn model_filenames(which_model: WhichModel) -> Result<(PathBuf, PathBuf, PathBuf), ApiError> {
    let (default_model, default_revision) = which_model.model_and_revision();

    // TODO: add support of env variables to set custom
    let model_id = default_model.to_string();
    // TODO: add support of env variables to set custom
    let revision = default_revision.to_string();

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let (config, tokenizer, weights) = match which_model {
        WhichModel::QuantizedTiny => (
            repo.get("config-tiny.json")?,
            repo.get("tokenizer-tiny.json")?,
            repo.get("model-tiny-q80.gguf")?,
        ),

        WhichModel::QuantizedTinyEn => (
            repo.get("config-tiny-en.json")?,
            repo.get("tokenizer-tiny-en.json")?,
            repo.get("model-tiny-en-q80.gguf")?,
        ),
        _ => (
            repo.get("config.json")?,
            repo.get("tokenizer.json")?,
            repo.get("model.safetensors")?,
        ),
    };

    Ok((config, tokenizer, weights))
}

pub fn vector_into_tokens(tokenizer: &Tokenizer, input: &[&str]) -> Vec<u32> {
    input
        .iter()
        .flat_map(|n| {
            tokenizer
                // .token_to_id(n)
                .encode(*n, false)
                .expect(&format!("no token-id for {n}"))
                .get_ids()
                .to_vec()
        })
        .collect()
}

pub fn new(which_model: WhichModel, language: Option<&str>) -> InferenceResult<Decoder> {
    let device = inference_device()?;

    let (config_filename, tokenizer_filename, weights_filename) = model_filenames(which_model)?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let model = if which_model.is_quantized() {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_filename,
            &device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
    } else {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };

    let language_token = if which_model.is_multilingual() {
        let lang = language.expect("language must be set for multilingual models");

        assert!(
            multilingual::SUPPORTED_LANGUAGES
                .iter()
                .any(|(t, _)| t == &lang),
            "provided language {lang} is not supported"
        );

        Some(token_id(&tokenizer, &format!("<|{lang}|>"))?)
    } else {
        None
    };

    Decoder::new(device, config, model, tokenizer, language_token)
}

impl SpeechRecognitionModel for Model {
    fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> InferenceResult<Tensor> {
        match self {
            Model::Normal(model) => model.encoder.forward(x, flush),
            Model::Quantized(model) => model.encoder.forward(x, flush),
        }
        .map_err(Into::into)
    }

    fn decoder_forward(&mut self, x: &Tensor, xa: &Tensor, flush: bool) -> InferenceResult<Tensor> {
        match self {
            Model::Normal(model) => model.decoder.forward(x, xa, flush),
            Model::Quantized(model) => model.decoder.forward(x, xa, flush),
        }
        .map_err(Into::into)
    }

    fn decoder_final_linear(&self, x: &Tensor) -> InferenceResult<Tensor> {
        match self {
            Model::Normal(model) => model.decoder.final_linear(x),
            Model::Quantized(model) => model.decoder.final_linear(x),
        }
        .map_err(Into::into)
    }
}

pub struct Decoder {
    device: Device,
    model: Model,
    config: Config,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,

    suppress_tokens: Tensor,

    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    space_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,

    boost_tokens: Vec<u32>,
    command_tokens: Vec<u32>,
    penalty_tokens: Vec<u32>,

    temperatures: Vec<f64>,
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> InferenceResult<u32> {
    match tokenizer.token_to_id(token) {
        // TODO: replace panic with bail! ?
        None => panic!("no token-id for {token}"),
        Some(id) => Ok::<u32, InferenceError>(id),
    }
    .map_err(Into::into)
}

impl Decoder {
    /// Create a new decoder.
    pub fn new(
        device: Device,
        config: Config,
        model: Model,
        tokenizer: Tokenizer,
        language_token: Option<u32>,
    ) -> InferenceResult<Self> {
        let seed = 299792458;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    ZERO
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            // TODO: replace panic with bail! ?
            None => panic!("unable to find any non-speech token"),
            Some(n) => n,
        };

        // let language_token = match (args.model.is_multilingual(), args.language) {
        //     (true, None) => Some(multilingual::detect_language(&mut model, &tokenizer, &mel)?),
        //     (false, None) => None,
        //     (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
        //         Ok(token_id) => Some(token_id),
        //         Err(_) => anyhow::bail!("language {language} is not supported"),
        //     },
        //     (false, Some(_)) => {
        //         anyhow::bail!("a language cannot be set for non-multilingual models")
        //     }
        // };

        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => panic!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![ZERO; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let space_token = vector_into_tokens(&tokenizer, &[" "])[0];

        let boost_tokens = vector_into_tokens(
            &tokenizer,
            &[
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten",
            ],
        );

        let command_tokens = vector_into_tokens(
            &tokenizer,
            &[
                // "four", "off", "forward", "five", "on", "yes", "six", "down", "house", "two",
                // "marvin", "visual", "up", "seven", "zero", "bird", "one", "sheila", "three",
                // "stop", "left", "nine", "follow", "wow", "no", "dog", "go", "happy", "bed", "tree",
                // "learn", "backward", "cat", "eight", "right",
            ],
        );

        #[rustfmt::skip]
        let penalty_tokens = vector_into_tokens(
            &tokenizer,
            &[
                " ",
                // Punctuation tokens
                "!", "'", "\\", "`", "(", ")", "*", ",", ".", "..", "...",
                ":", ";", "?", "[", "]", "_", "{", "|", "}", "~",

                // Punctuation tokens with spaces
                ". ", ", ", "! ", "? ", "... ", "; ", ": ",
                // Numbers
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                // Numbers with spaces
                " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
            ],
        );

        Ok(Self {
            device,
            model,
            config,
            mel_filters,
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            command_tokens,
            eot_token,
            space_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            boost_tokens,
            penalty_tokens,
            temperatures: audio_params::TEMPERATURES.to_vec(),
        })
    }

    /// Set temperatures for decoding.
    pub fn with_temperatures(&mut self, temperatures: Vec<f64>) {
        self.temperatures = temperatures;
    }

    /// Convert samples into mel spectrogram.
    pub fn pcm_to_mel(&self, pcm: &[f32]) -> InferenceResult<Tensor> {
        let pcm_len = pcm.len();
        let mel = audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let num_mel_bins = self.config.num_mel_bins;
        let shape_size = mel_len / num_mel_bins;
        let shape: Shape = (1, num_mel_bins, shape_size).into();

        debug!(
            "PCM length {} MEL length {} / {} = shape size {} // Shape {:?}",
            pcm_len, mel_len, num_mel_bins, shape_size, shape
        );

        Tensor::from_vec(mel, shape, &self.device).map_err(Into::into)
    }

    pub fn with_mel_segments(&mut self, pcm: &[f32]) -> InferenceResult<Vec<proto::Segment>> {
        let mel = self.pcm_to_mel(pcm)?;

        self.segments(&mel)
    }

    fn special_tokens(&self) -> [u32; 4] {
        [
            self.sot_token,
            self.transcribe_token,
            self.no_timestamps_token,
            self.space_token,
        ]
    }

    fn is_special_token(&self, token: u32) -> bool {
        self.special_tokens().contains(&token)
    }
}

impl SpeechRecognitionDecoder for Decoder {
    fn decode(&mut self, mel: &Tensor, temperature: f64) -> InferenceResult<proto::DecodingResult> {
        let special_tokens = self.special_tokens();

        // 1. Initial setup
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob: f64 = 0.0;
        let mut no_speech_prob = f64::NAN;

        // Track token frequencies for repetition penalty
        let mut token_frequencies: HashMap<u32, usize> = HashMap::new();
        let repetition_penalty: f32 = 4.2; // Adjust this value to control penalty strength
        let repetition_frequency = 3; // Adjust this value to control how many times a token must appear before penalty is applied

        let mut inc_freq = |token: u32| {
            *token_frequencies.entry(token).or_insert(0) += 1;
        };

        // Create boost mask
        let boost_value = 2.0f32; // Adjust this value to control how much we favor word numbers

        let command_boost_value = 3.0f32; // Adjust this value to control how much we favor commands

        // 2. Token tracking
        let mut seen_tokens: HashSet<u32> = HashSet::new();
        let mut word_tokens = 0;
        let max_word_tokens = 1;

        // 3. Create initial token sequence
        // The tokens must be in a specific sequence that matches how the model was trained:
        // - Start with SOT token
        // - Language token (if multilingual model)
        // - Task token (transcribe)
        // - Settings tokens (notimestamps)
        let mut tokens = vec![self.sot_token];
        inc_freq(self.sot_token);

        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
            inc_freq(language_token);
        }
        tokens.push(self.transcribe_token);
        inc_freq(self.transcribe_token);
        tokens.push(self.no_timestamps_token);
        inc_freq(self.no_timestamps_token);

        // 4. Create combined mask once (not repeatedly)
        let mask = {
            let mut m = self.suppress_tokens.clone();
            let dims1: u32 = m.dims1()? as u32;

            // Suppress space token always
            // TODO: questionable
            m = m.slice_assign(
                &[self.space_token as usize..=self.space_token as usize],
                &Tensor::new(&[-f32::INFINITY], mel.device())?,
            )?;

            // Penalties
            for &token in self.penalty_tokens.iter().filter(|t| *t < &dims1) {
                m = m.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[-f32::INFINITY], mel.device())?,
                )?;
            }

            // Apply boosts
            for &token in self.boost_tokens.iter().filter(|t| *t < &dims1) {
                m = m.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[boost_value], mel.device())?,
                )?;
            }

            // Commands boost
            for &token in self.command_tokens.iter().filter(|t| *t < &dims1) {
                m = m.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[command_boost_value], mel.device())?,
                )?;
            }

            m
        };

        println!("sample_len {}", sample_len);

        // 5. Main decoding loop
        for i in 0..sample_len {
            // Reduced for speech commands
            // Get next token logits
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            println!(
                "token_frequencies {:?} seen_tokens {:?} word_tokens {}",
                token_frequencies
                    .iter()
                    .map(|(k, v)| {
                        let kk = self
                            .tokenizer
                            .decode(&[*k], true)
                            .unwrap_or_else(|_| format!("{}", k));
                        format!("{}:{}", kk, v)
                    })
                    .collect::<Vec<String>>(),
                seen_tokens,
                word_tokens
            );

            // Handle first iteration special cases
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let mut logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            // Apply repetition penalty using token frequencies
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;

            // for (token, frequency) in &token_frequencies {
            //     if *frequency > repetition_frequency {
            //         // Instead of repetition_frequency
            //         let token_idx = *token as usize;

            //         if let Some(last_token) = tokens.last() {
            //             println!(
            //                 "Last token {} token {} frequency {}",
            //                 last_token, token, frequency
            //             );
            //         }

            //         // TODO: apply penalty only to the tokens repeating last token
            //         if token_idx < logits_vec.len() {
            //             let penalty = repetition_penalty.powi(*frequency as i32);
            //             logits_vec[token_idx] = if logits_vec[token_idx] < ZERO {
            //                 logits_vec[token_idx] * penalty as f32
            //             } else {
            //                 logits_vec[token_idx] / penalty as f32
            //             };
            //         }
            //     }
            // }
            // Apply repetition penalty only for tokens that repeat the last token
            let last_two_tokens = tokens
                .get(tokens.len().saturating_sub(2 + special_tokens.len())..)
                .unwrap_or(&[])
                .into_iter()
                .filter(|&t| !special_tokens.contains(t))
                .rev()
                .take(2)
                .collect::<Vec<_>>();

            for (token, frequency) in &token_frequencies {
                // Only penalize if this token matches the last token and has been used
                if last_two_tokens.contains(&token) || *frequency > repetition_frequency {
                    let token_idx = *token as usize;
                    if token_idx < logits_vec.len() {
                        // Strong penalty for immediate repetition
                        let penalty = repetition_penalty.powi(*frequency as i32);
                        logits_vec[token_idx] = if logits_vec[token_idx] < ZERO {
                            logits_vec[token_idx] * penalty as f32
                        } else {
                            logits_vec[token_idx] / penalty as f32
                        };
                    }
                }
            }

            // Convert back and apply mask
            logits = Tensor::new(logits_vec.as_slice(), mel.device())?;
            logits = logits.broadcast_add(&mask)?;

            // Get next token
            let next_token = if temperature > 0f64 {
                let prs = softmax(&(&logits / temperature)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            // Check stopping conditions first
            if next_token == self.eot_token
            // || word_tokens >= max_word_tokens
            // || seen_tokens.contains(&next_token)
            {
                break;
            }

            // Add token if valid
            if !self.penalty_tokens.contains(&next_token) {
                tokens.push(next_token);
                seen_tokens.insert(next_token);
                word_tokens += 1;
                *token_frequencies.entry(next_token).or_insert(0) += 1; // Add this line

                let prob = softmax(&logits, candle_core::D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()? as f64;
                sum_logprob += prob.ln();
            } else {
                println!("Skipping penalty token {next_token}");
            }
        }

        // 6. Final processing
        let text = self.tokenizer.decode(&tokens, true)?.trim().to_lowercase();
        println!("Token {:?} decoded to text: {}", tokens, text);
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio: None,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> InferenceResult<proto::DecodingResult> {
        for (i, &t) in self.temperatures.clone().iter().enumerate() {
            let dr: InferenceResult<DecodingResult> = self.decode(segment, t);

            debug!("Decoding with temperature {t}: {dr:?}");

            if i == self.temperatures.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr
                        .compression_ratio
                        .map(|cr| cr > audio_params::COMPRESSION_RATIO_THRESHOLD)
                        .unwrap_or(true)
                        || dr.avg_logprob < audio_params::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > audio_params::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn segments(&mut self, mel: &Tensor) -> InferenceResult<Vec<proto::Segment>> {
        let (_, _, content_frames) = mel.dims3()?;

        // For very short content, process as single chunk
        if content_frames <= audio_params::N_FRAMES_MAX {
            let dr = self.decode_with_fallback(mel)?;
            return Ok(vec![Segment {
                start: 0.0,
                duration: (content_frames * audio_params::HOP_LENGTH) as f64
                    / audio_params::SAMPLE_RATE as f64,
                dr,
            }]);
        }

        let mut seek = 0;
        let mut segments = vec![];

        // Max chunk size for Whisper is 3000 frames
        let max_chunk_size = audio_params::N_FRAMES_MAX;

        let max_real_time_chunk_size: usize = 8 * audio_params::HOP_LENGTH;
        let content_chunks = content_frames / audio_params::HOP_LENGTH;
        let chunk_size = std::cmp::min(
            content_chunks * audio_params::HOP_LENGTH,
            max_real_time_chunk_size,
        );

        let segment_size = usize::min(content_frames, max_chunk_size);

        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, segment_size);

            println!(
                "content_frames {} seek {} segment_size {}",
                content_frames, seek, segment_size
            );

            let mel_segment = mel.narrow(2, seek, segment_size)?;

            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;

            // no > 0.6 && avg < -1.0
            if dr.no_speech_prob > audio_params::NO_SPEECH_THRESHOLD
                && dr.avg_logprob < audio_params::LOGPROB_THRESHOLD
            {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: (seek * audio_params::HOP_LENGTH) as f64 / audio_params::SAMPLE_RATE as f64,
                duration: (segment_size * audio_params::HOP_LENGTH) as f64
                    / audio_params::SAMPLE_RATE as f64,
                dr,
            };
            // println!(
            //     "{:.1}s -- {:.1}s: {}",
            //     segment.start,
            //     segment.start + segment.duration,
            //     segment.dr.text,
            // );

            // println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            segments.push(segment)
        }
        Ok(segments)
    }
}
