use std::{collections::HashMap, path::PathBuf};

use audio_params::{HOP_LENGTH, SAMPLE_RATE};
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

mod audio;
mod multilingual;

pub mod audio_params {
    // Audio parameters.
    pub const SAMPLE_RATE: usize = 16000;
    pub const N_FFT: usize = 400;
    pub const HOP_LENGTH: usize = 128;
    // Responsiveness: Shorter chunks (e.g., 10-30 ms) provide faster response times, which is crucial for real-time applications.
    pub const CHUNK_LENGTH: usize = 20;
    pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 320000 samples in a 30-second chunk
    pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 2500 frames in a mel spectrogram input

    pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
    pub const LOGPROB_THRESHOLD: f64 = -1.0;
    // pub const LOGPROB_THRESHOLD: f64 = -0.5; TODO:

    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0. This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
    pub const TEMPERATURES: [f64; 7] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    // pub const TEMPERATURES: [f64; 1] = [0.5];
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
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,

    boost_tokens: Vec<u32>,
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
                    0f32
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
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let boost_tokens = vector_into_tokens(
            &tokenizer,
            &[
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten",
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
            eot_token,
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

    pub fn pcm_to_mel(&self, pcm: &[f32]) -> InferenceResult<Tensor> {
        let pcm_len = pcm.len();
        let mel = audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let num_mel_bins = self.config.num_mel_bins;
        let shape_size = mel_len / num_mel_bins;
        let shape: Shape = (1, num_mel_bins, shape_size).into();

        // println!(
        //     "PCM length {} MEL length {} / {} = shape size {} // Shape {:?}",
        //     pcm_len, mel_len, num_mel_bins, shape_size, shape
        // );

        Tensor::from_vec(mel, shape, &self.device).map_err(Into::into)
    }

    pub fn real_time(&mut self, pcm: &[f32]) -> InferenceResult<Vec<proto::Segment>> {
        let mel = self.pcm_to_mel(pcm)?;

        self.segments2(&mel)
    }

    pub fn bulk_time(&mut self, pcm: &[f32]) -> InferenceResult<Vec<proto::Segment>> {
        let mel = self.pcm_to_mel(pcm)?;

        self.segments2(&mel)
    }

    pub fn real_time_segments(&mut self, mel: &Tensor) -> InferenceResult<Vec<proto::Segment>> {
        let (_, _, content_frames) = mel.dims3()?;

        let mut seek = 0;
        let mut segments = vec![];

        let min_chunk_size = 512;
        let max_chunk_size = 2048;
        let segment_size = usize::min(content_frames, max_chunk_size);

        while seek < content_frames {
            let chunk_size = usize::min(segment_size, content_frames - seek);

            // println!(
            //     "content_frames {} segment_size {} seek {} chunk_size {}",
            //     content_frames, segment_size, seek, chunk_size
            // );

            if chunk_size < min_chunk_size {
                println!("BREAK");
                break; // Exit loop if the remaining chunk is too small
            }

            let mel_segment = mel.narrow(2, seek, chunk_size)?;

            let dr = self.decode_with_fallback(&mel_segment)?;

            seek += chunk_size;

            if dr.no_speech_prob > audio_params::NO_SPEECH_THRESHOLD
                && dr.avg_logprob < audio_params::LOGPROB_THRESHOLD
            {
                println!("No speech detected, skipping {seek} {dr:?}");
                continue;
            }

            let segment = Segment {
                start: (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64,
                duration: (chunk_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64,
                dr,
            };

            // println!(
            //     "{:.1}s -- {:.1}s: {} no_speech_prob {:.2} avg_logprob {:.2}",
            //     segment.start,
            //     segment.start + segment.duration,
            //     segment.dr.text,
            //     segment.dr.no_speech_prob,
            //     segment.dr.avg_logprob,
            // );

            segments.push(segment);
        }

        Ok(segments)
    }

    pub fn segments2(&mut self, mel: &Tensor) -> InferenceResult<Vec<proto::Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];

        // Parameters for segment size control
        let min_chunk_size = 512;
        let max_chunk_size = 2048;
        let preferred_size = usize::min(content_frames, max_chunk_size);

        while seek < content_frames {
            // Determine optimal chunk size based on remaining content
            let remaining_frames = content_frames - seek;
            let chunk_size = if remaining_frames < min_chunk_size {
                // If remaining frames are too small, process everything at once
                remaining_frames
            } else {
                // Otherwise use the optimal size or remaining frames, whichever is smaller
                usize::min(preferred_size, remaining_frames)
            };

            // Calculate timing information
            let time_offset =
                (seek * audio_params::HOP_LENGTH) as f64 / audio_params::SAMPLE_RATE as f64;
            let segment_duration =
                (chunk_size * audio_params::HOP_LENGTH) as f64 / audio_params::SAMPLE_RATE as f64;

            // Process the audio segment
            let mel_segment = mel.narrow(2, seek, chunk_size)?;
            let dr = self.decode_with_fallback(&mel_segment)?;

            // Skip segments with no meaningful speech content
            if dr.no_speech_prob > audio_params::NO_SPEECH_THRESHOLD
                && dr.avg_logprob < audio_params::LOGPROB_THRESHOLD
            {
                seek += chunk_size;
                continue;
            }

            // Create and store the segment
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };

            segments.push(segment);
            seek += chunk_size;

            // Optional: Break if remaining chunk is too small for processing
            if remaining_frames - chunk_size < min_chunk_size {
                break;
            }
        }

        Ok(segments)
    }

    fn bulk_segments(&mut self, mel: &Tensor) -> InferenceResult<Vec<proto::Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];

        while seek < content_frames {
            let time_offset =
                (seek * audio_params::HOP_LENGTH) as f64 / audio_params::SAMPLE_RATE as f64;

            let segment_size = usize::min(content_frames - seek, audio_params::N_FRAMES);

            let segment_duration =
                (segment_size * audio_params::HOP_LENGTH) as f64 / audio_params::SAMPLE_RATE as f64;

            // println!(
            //     "content_frames {} seek {} time_offset {} segment_size {} segment_duration {}",
            //     content_frames, seek, time_offset, segment_size, segment_duration
            // );

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
                start: time_offset,
                duration: segment_duration,
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

impl SpeechRecognitionDecoder for Decoder {
    fn decode(&mut self, mel: &Tensor, t: f64) -> InferenceResult<proto::DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];

        // Track token frequencies for repetition penalty
        let mut token_frequencies: HashMap<u32, usize> = HashMap::new();
        let repetition_penalty = 1.2; // Adjust this value to control penalty strength

        // Create suppress mask for punctuation
        let mut suppress_mask = self.suppress_tokens.clone();
        for &token in &self.penalty_tokens {
            if token < suppress_mask.dims1()? as u32 {
                suppress_mask = suppress_mask.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[-f32::INFINITY], mel.device())?,
                )?;
            }
        }

        // Create boost mask
        let boost_value = 5.0f32; // Adjust this value to control how much we favor word numbers
        for &token in self.boost_tokens.iter() {
            if token < suppress_mask.dims1()? as u32 {
                suppress_mask = suppress_mask.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[boost_value], mel.device())?,
                )?;
            }
        }

        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
            *token_frequencies.entry(language_token).or_insert(0) += 1;
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);

        // Initialize frequencies for initial tokens
        *token_frequencies.entry(self.transcribe_token).or_insert(0) += 1;
        *token_frequencies
            .entry(self.no_timestamps_token)
            .or_insert(0) += 1;

        for i in 0..sample_len {
            let space_token = 220;
            if i == 0 {
                // Only penalize space at the beginning
                suppress_mask = suppress_mask.slice_assign(
                    &[space_token as usize..=space_token as usize],
                    &Tensor::new(&[-f32::INFINITY], mel.device())?,
                )?;
            }

            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

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

            // Apply repetition penalty
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;
            for (token, frequency) in &token_frequencies {
                if *frequency > 0 {
                    let token_idx = *token as usize;
                    if token_idx < logits_vec.len() {
                        // Penalize tokens that have appeared before
                        logits_vec[token_idx] = if logits_vec[token_idx] < 0.0 {
                            logits_vec[token_idx] * repetition_penalty as f32
                        } else {
                            logits_vec[token_idx] / repetition_penalty as f32
                        };
                    }
                }
            }

            // Convert back to tensor and apply punctuation suppression
            logits = Tensor::new(logits_vec.as_slice(), mel.device())?;
            logits = logits.broadcast_add(&suppress_mask)?;

            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
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

            // Only add token if it's not a penalty token
            if !self.penalty_tokens.contains(&next_token) {
                // Update token frequency before adding
                *token_frequencies.entry(next_token).or_insert(0) += 1;

                tokens.push(next_token);
                let prob = softmax(&logits, candle_core::D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()? as f64;

                if next_token == self.eot_token
                    || tokens.len() > model.config().max_target_positions
                {
                    break;
                }
                sum_logprob += prob.ln();
            }
        }

        let text = self.tokenizer.decode(&tokens, true)?.trim().to_lowercase();
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: None,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> InferenceResult<proto::DecodingResult> {
        for (i, &t) in self.temperatures.clone().iter().enumerate() {
            let dr: InferenceResult<DecodingResult> = self.decode(segment, t);

            // println!("Decoding with temperature {t}: {dr:?}");

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
        self.bulk_segments(mel)
    }
}
