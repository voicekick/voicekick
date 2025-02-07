use core::fmt;
use std::{collections::HashMap, ops::Deref, path::PathBuf};

use candle_core::{Device, IndexOp, Shape, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m, Config};

pub use hf_hub::api::Progress;
use hf_hub::{
    api::sync::{ApiBuilder, ApiError, ApiRepo},
    Cache, CacheRepo, Repo, RepoType,
};
use inference_candle::{
    proto::{self, DecodingResult, Segment},
    InferenceResult, SpeechRecognitionDecoder, SpeechRecognitionModel,
};
use tokenizers::Tokenizer;
use tracing::{debug, error, warn};

mod audio;
mod builder;
mod multilingual;
pub use builder::WhisperBuilder;
pub use multilingual::SUPPORTED_LANGUAGES;

pub mod audio_params {
    // Sample rate
    pub const SAMPLE_RATE: usize = 16000;

    /// For Whisper, preferred max segment sizes in mel spectrogram frames:
    /// - Optimal: 1500 frames (~30 seconds of audio)
    /// - Maximum supported: 3000 frames (~60 seconds)
    /// - Real-time recommended: 800-1000 frames (~16-20 seconds)
    pub const N_FRAMES_MAX: usize = 3000;
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

// Create to Iter for WhichModel
impl WhichModel {
    pub fn iter() -> impl Iterator<Item = WhichModel> {
        [
            WhichModel::TinyEn,
            WhichModel::BaseEn,
            WhichModel::SmallEn,
            WhichModel::MediumEn,
            WhichModel::Tiny,
            WhichModel::Base,
            WhichModel::Small,
            WhichModel::Medium,
            WhichModel::Large,
            WhichModel::LargeV2,
            WhichModel::LargeV3,
            WhichModel::LargeV3Turbo,
            WhichModel::DistilMediumEn,
            WhichModel::DistilLargeV2,
            WhichModel::DistilLargeV3,
            WhichModel::QuantizedTiny,
            WhichModel::QuantizedTinyEn,
        ]
        .iter()
        .copied()
    }
}

impl fmt::Display for WhichModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.deref())
    }
}

impl From<&str> for WhichModel {
    fn from(s: &str) -> Self {
        match s {
            "TinyEn" => WhichModel::TinyEn,
            "BaseEn" => WhichModel::BaseEn,
            "SmallEn" => WhichModel::SmallEn,
            "MediumEn" => WhichModel::MediumEn,
            "Tiny" => WhichModel::Tiny,
            "Base" => WhichModel::Base,
            "Small" => WhichModel::Small,
            "Medium" => WhichModel::Medium,
            "Large" => WhichModel::Large,
            "LargeV2" => WhichModel::LargeV2,
            "LargeV3" => WhichModel::LargeV3,
            "LargeV3Turbo" => WhichModel::LargeV3Turbo,
            "DistilMediumEn" => WhichModel::DistilMediumEn,
            "DistilLargeV2" => WhichModel::DistilLargeV2,
            "DistilLargeV3" => WhichModel::DistilLargeV3,
            "QuantizedTiny" => WhichModel::QuantizedTiny,
            "QuantizedTinyEn" => WhichModel::QuantizedTinyEn,
            _ => panic!("Unknown model: {s}"),
        }
    }
}

impl Deref for WhichModel {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            WhichModel::TinyEn => "TinyEn",
            WhichModel::BaseEn => "BaseEn",
            WhichModel::SmallEn => "SmallEn",
            WhichModel::MediumEn => "MediumEn",
            WhichModel::Tiny => "Tiny",
            WhichModel::Base => "Base",
            WhichModel::Small => "Small",
            WhichModel::Medium => "Medium",
            WhichModel::Large => "Large",
            WhichModel::LargeV2 => "LargeV2",
            WhichModel::LargeV3 => "LargeV3",
            WhichModel::LargeV3Turbo => "LargeV3Turbo",
            WhichModel::DistilMediumEn => "DistilMediumEn",
            WhichModel::DistilLargeV2 => "DistilLargeV2",
            WhichModel::DistilLargeV3 => "DistilLargeV3",
            WhichModel::QuantizedTiny => "QuantizedTiny",
            WhichModel::QuantizedTinyEn => "QuantizedTinyEn",
        }
    }
}

impl Default for WhichModel {
    fn default() -> Self {
        Self::TinyEn
    }
}

impl WhichModel {
    fn is_quantized(&self) -> bool {
        matches!(self, Self::QuantizedTiny | Self::QuantizedTinyEn)
    }

    /// Check if the model is multilingual.
    pub fn is_multilingual(&self) -> bool {
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

/// Whisper model.
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

fn _model_filenames(
    which_model: WhichModel,
) -> Result<(&'static str, &'static str, &'static str, ApiRepo, CacheRepo), ApiError> {
    let (default_model, default_revision) = which_model.model_and_revision();

    let model_id = default_model.to_string();
    let revision = default_revision.to_string();

    let cache = Cache::default();
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let cache_repo = cache.repo(repo.clone());

    let api = ApiBuilder::from_cache(cache.clone())
        .with_retries(2)
        .build()?;
    let api_repo = api.repo(repo);

    let (config_file, tokenizer_file, weights_file) = match which_model {
        WhichModel::QuantizedTiny => (
            "config-tiny.json",
            "tokenizer-tiny.json",
            "model-tiny-q80.gguf",
        ),

        WhichModel::QuantizedTinyEn => (
            "config-tiny-en.json",
            "tokenizer-tiny-en.json",
            "model-tiny-en-q80.gguf",
        ),
        _ => ("config.json", "tokenizer.json", "model.safetensors"),
    };

    Ok((
        config_file,
        tokenizer_file,
        weights_file,
        api_repo,
        cache_repo,
    ))
}

pub fn model_filenames_exist(which_model: WhichModel) -> Result<bool, ApiError> {
    let (config_file, tokenizer_file, weights_file, _, cache_repo) = _model_filenames(which_model)?;

    Ok(cache_repo.get(config_file).is_some()
        && cache_repo.get(tokenizer_file).is_some()
        && cache_repo.get(weights_file).is_some())
}

/// Get the model filenames for the selected model.
pub fn model_filenames(which_model: WhichModel) -> Result<(PathBuf, PathBuf, PathBuf), ApiError> {
    let (config_file, tokenizer_file, weights_file, api_repo, _) = _model_filenames(which_model)?;

    let (config, tokenizer, weights) = (
        api_repo.get(config_file)?,
        api_repo.get(tokenizer_file)?,
        api_repo.get(weights_file)?,
    );

    Ok((config, tokenizer, weights))
}

/// Get the model filenames for the selected model.
pub fn download_model_with_progress<P: Progress + Clone>(
    which_model: WhichModel,
    progress: P,
) -> Result<bool, ApiError> {
    let (config_file, tokenizer_file, weights_file, api_repo, _) = _model_filenames(which_model)?;

    if model_filenames_exist(which_model)? {
        Ok(false)
    } else {
        let _ = api_repo.download_with_progress(config_file, progress.clone())?;
        let _ = api_repo.download_with_progress(tokenizer_file, progress.clone())?;
        let _ = api_repo.download_with_progress(weights_file, progress.clone())?;

        Ok(true)
    }
}

/// Create a set of token setters for the Whisper model.
#[non_exhaustive]
pub enum WithSpace {
    /// Add a space before the token.
    Before,
}

/// Convert a vector of strings into a vector of token IDs.
pub fn vector_into_tokens<T: AsRef<str>>(
    tokenizer: &Tokenizer,
    input: &[T],
    with_space: Option<WithSpace>,
) -> Vec<u32> {
    let encode = |n: &str| -> Vec<u32> {
        tokenizer
            .encode(n, false)
            .unwrap_or_else(|_| panic!("no token-id for {n}"))
            .get_ids()
            .to_vec()
    };

    let spaced: Vec<u32> = if let Some(s) = with_space {
        input
            .iter()
            .flat_map(|n| match s {
                WithSpace::Before => encode(&format!(" {}", n.as_ref())),
            })
            .collect()
    } else {
        vec![]
    };

    input
        .iter()
        .flat_map(|n| encode(n.as_ref()))
        .chain(spaced)
        .collect()
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

/// Whisper model
pub struct Whisper {
    device: Device,
    model: Model,
    config: Config,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,

    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    space_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,

    suppress_tokens: Tensor,
    boost_tokens: Vec<u32>,
    command_tokens: Vec<u32>,
    penalty_tokens: Vec<u32>,

    temperatures: Vec<f64>,

    // Mel
    n_fft: usize,
    hop_lengnth: usize,

    // Boosts & penalties
    repetition_penalty: f32,
    repetition_frequency: usize,
    boost_value: f32,
    command_boost_value: f32,

    // Thresholds
    no_speech_threshold: f64,
    logprob_threshold: f64,
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

impl Whisper {
    // Create a builder method
    pub fn builder(
        device: Device,
        config: Config,
        model: Model,
        tokenizer: Tokenizer,
        language_token: Option<u32>,
    ) -> WhisperBuilder {
        WhisperBuilder::new(device, config, model, tokenizer, language_token)
    }

    /// Convert samples into mel spectrogram.
    pub fn pcm_to_mel(&self, pcm: &[f32]) -> InferenceResult<Tensor> {
        let min_pcm_length = 16000 * 4; // 4 seconds at 16kHz, LargeV3Turbo workaround

        // Create padded PCM
        let padded_pcm = if pcm.len() < min_pcm_length {
            let mut padded = vec![0.0; min_pcm_length];
            // Center the audio in the padded window
            let start_idx = (min_pcm_length - pcm.len()) / 2;
            padded[start_idx..start_idx + pcm.len()].copy_from_slice(pcm);
            padded
        } else {
            pcm.to_vec()
        };

        let pcm_len = padded_pcm.len();
        let mel = audio::pcm_to_mel(
            &self.config,
            &padded_pcm,
            &self.mel_filters,
            self.n_fft,
            self.hop_lengnth,
        );
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
}

#[allow(dead_code)]
fn debug_top_logits(tokenizer: &Tokenizer, logits_vec: &[f32], logits: Tensor) {
    // Create a vector of (logit value, token) pairs
    let mut indexed_logits: Vec<(f32, u32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (logit, idx as u32))
        .collect();

    // Sort in descending order by logit value
    indexed_logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Take top 10 and decode tokens
    println!("Top 10 Logits:");
    for (logit, token) in indexed_logits.iter().take(10) {
        match tokenizer.decode(&[*token], true) {
            Ok(text) => {
                println!(
                    "Token: {}, Logit: {:.4}, Text: '{}'",
                    token,
                    logit,
                    text.trim()
                );
            }
            Err(e) => {
                println!(
                    "Token: {}, Logit: {:.4}, Decode Error: {:?}",
                    token, logit, e
                );
            }
        }
    }

    println!("Logits {:?}", logits);
}

impl SpeechRecognitionDecoder for Whisper {
    fn decode(&mut self, mel: &Tensor, temperature: f64) -> InferenceResult<proto::DecodingResult> {
        // 1. Initial setup
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob: f64 = 0.0;
        let mut no_speech_prob = f64::NAN;

        // 2. Track token frequencies for repetition penalty
        let mut token_frequencies: HashMap<u32, usize> = HashMap::new();

        let mut inc_freq = |token: u32| {
            *token_frequencies.entry(token).or_insert(0) += 1;
        };

        // Token tracking
        let mut last_tokens: [u32; 3] = [0, 0, 0];

        // 3. Create initial token sequence
        // The tokens must be in a specific sequence that matches how the model was trained:
        let mut tokens = vec![
            // - 1. Start with SOT token
            self.sot_token,
            // - 2.? Language token (if multilingual model)
            //
            // - 3. Task token (transcribe)
            self.transcribe_token,
            // - 4. Settings tokens (notimestamps)
            self.no_timestamps_token,
        ];
        inc_freq(self.sot_token);
        inc_freq(self.transcribe_token);
        inc_freq(self.no_timestamps_token);

        if let Some(language_token) = self.language_token {
            tokens.insert(1, language_token);
            inc_freq(language_token);
        }

        // 4. Create combined mask once (not repeatedly)
        let mask = {
            let mut m = self.suppress_tokens.clone();
            let dims1: u32 = m.dims1()? as u32;

            // Suppress space token always
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
                    &Tensor::new(&[self.boost_value], mel.device())?,
                )?;
            }

            // Commands boost
            for &token in self.command_tokens.iter().filter(|t| *t < &dims1) {
                m = m.slice_assign(
                    &[token as usize..=token as usize],
                    &Tensor::new(&[self.command_boost_value], mel.device())?,
                )?;
            }

            m
        };

        // 5. Main decoding loop
        for i in 0..sample_len {
            // Reduced for speech commands
            // Get next token logits
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Handle first iteration special cases
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            // Convert logits to a vector and index
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;

            // debug_top_logits(&self.tokenizer, &logits_vec, logits);

            // Apply repetition penalty only for tokens that repeat at lot
            for (token, frequency) in &token_frequencies {
                // Only penalize if this token matches the last token and has been used
                if *frequency > self.repetition_frequency {
                    let token_idx = *token as usize;
                    if token_idx < logits_vec.len() {
                        // Strong penalty for immediate repetition
                        let penalty = self.repetition_penalty.powi(*frequency as i32);
                        logits_vec[token_idx] = if logits_vec[token_idx] < 0.0 {
                            logits_vec[token_idx] * penalty
                        } else {
                            logits_vec[token_idx] / penalty
                        };
                    }
                }
            }

            // Convert back and apply mask
            let mut masked_logits = Tensor::new(logits_vec.as_slice(), mel.device())?;
            masked_logits = masked_logits.broadcast_add(&mask)?;

            // Get next token
            let next_token = if temperature > 0f64 {
                let prs = softmax(&(&masked_logits / temperature)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                let logits_v: Vec<f32> = masked_logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            };

            // Check stopping conditions first
            if next_token == self.eot_token
            // Model halucinating the same token, e.g.: back backward back backward back
            || (last_tokens[0] == next_token &&  last_tokens[2] == next_token)
            {
                break;
            }

            last_tokens[0] = last_tokens[1];
            last_tokens[1] = last_tokens[2];

            if last_tokens[2] == next_token {
                break;
            }

            last_tokens[2] = next_token;

            // Add token if valid
            if self.penalty_tokens.contains(&next_token) {
                debug!(
                    "Skipping penalty token {next_token} decoded '{}'",
                    self.tokenizer.decode(&[next_token], true)?
                );

                continue;
            } else {
                tokens.push(next_token);
                *token_frequencies.entry(next_token).or_insert(0) += 1; // Add this line

                let prob = softmax(&masked_logits, candle_core::D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()? as f64;
                sum_logprob += prob.ln();
            }
        }

        // 6. Final processing
        let text = self.tokenizer.decode(&tokens, true)?.trim().to_lowercase();

        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
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
                    let needs_fallback = dr.avg_logprob < self.logprob_threshold;
                    if !needs_fallback || dr.no_speech_prob > self.no_speech_threshold {
                        return Ok(dr);
                    } else {
                        warn!("Decoding with temperature {t} failed to pass no_speech or logprob, retry with next temperature {:#?}", dr);
                    }
                }
                Err(err) => {
                    error!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn segments(&mut self, mel: &Tensor) -> InferenceResult<Vec<proto::Segment>> {
        let (_, _, content_frames) = mel.dims3()?;

        let mut seek = 0;
        let mut segments = vec![];

        // Max chunk size for Whisper is 3000 frames
        let max_chunk_size = audio_params::N_FRAMES_MAX;

        let segment_size = usize::min(content_frames, max_chunk_size);

        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, segment_size);

            debug!(
                "content_frames {} seek {} segment_size {}",
                content_frames, seek, segment_size
            );

            let mel_segment = mel.narrow(2, seek, segment_size)?;

            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;

            // no > 0.6 && avg < -1.0
            if dr.no_speech_prob > self.no_speech_threshold
                && dr.avg_logprob < self.logprob_threshold
            {
                debug!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                timestamp: None,
                duration: (segment_size * self.hop_lengnth) as f64
                    / audio_params::SAMPLE_RATE as f64,
                dr,
            };
            // println!("{:.1}s: {}", segment.duration, segment.dr.text,);

            // println!("{seek}: {segment:?}");
            segments.push(segment)
        }
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn get_test_tokenizer() -> Tokenizer {
        let (_, tokenizer_filename, _) = model_filenames(WhichModel::TinyEn).unwrap();

        Tokenizer::from_file(tokenizer_filename).unwrap()
    }
    #[test]
    fn test_vector_into_tokens_penalty_tokens() {
        let tokenizer = get_test_tokenizer();

        let input = &[
            // Punctuation tokens
            "!", "'", "\\", "`", "(", ")", "*", ",", ".", "..", "...", ":", ";", "?", "[", "]", "_",
            "{", "|", "}", "~",
        ];

        let tokens = vector_into_tokens(&tokenizer, input, None);
        assert!(!tokens.is_empty());
        assert_eq!(tokens.len(), input.len());
        assert!(tokens.contains(&0)); // !
        assert!(!tokens.contains(&220)); // <space>
    }

    #[test]
    fn test_vector_into_tokens() {
        let tokenizer = get_test_tokenizer();

        // Test basic tokenization without spaces
        let input = vec!["hello", "world"];
        let tokens = vector_into_tokens(&tokenizer, &input, None);
        assert!(!tokens.is_empty());

        // Test with space before
        let tokens_space_before = vector_into_tokens(&tokenizer, &input, Some(WithSpace::Before));
        assert!(!tokens_space_before.is_empty());
        assert!(
            tokens_space_before.len() > tokens.len(),
            "Space-prefixed tokens should be longer than basic tokens"
        );

        // Test with space after
        let tokens_space_after = vector_into_tokens(&tokenizer, &input, Some(WithSpace::Before));
        assert!(!tokens_space_after.is_empty());
        assert!(
            tokens_space_after.len() > tokens.len(),
            "Space-suffixed tokens should be longer than basic tokens"
        );

        // Test with space before and after
        let tokens_both_spaces = vector_into_tokens(&tokenizer, &input, Some(WithSpace::Before));
        assert!(!tokens_both_spaces.is_empty());
        assert!(
            tokens_both_spaces.len() > tokens.len(),
            "Space-wrapped tokens should be longer than basic tokens"
        );

        // Verify decoded tokens make sense
        let decoded = tokenizer
            .decode(&tokens, true)
            .expect("Failed to decode tokens");
        assert!(
            decoded.contains("hello"),
            "Decoded text should contain 'hello'"
        );
        assert!(
            decoded.contains("world"),
            "Decoded text should contain 'world'"
        );

        // Test empty input
        let empty_input: Vec<&str> = vec![];
        let empty_tokens = vector_into_tokens(&tokenizer, &empty_input, None);
        assert!(
            empty_tokens.is_empty(),
            "Empty input should produce empty output"
        );

        // Test single token
        let single_input = vec!["test"];
        let single_tokens = vector_into_tokens(&tokenizer, &single_input, None);
        assert!(
            !single_tokens.is_empty(),
            "Single input should produce tokens"
        );
    }

    #[test]
    fn test_vector_into_tokens_special_words() {
        let tokenizer = get_test_tokenizer();

        // Test basic tokenization without spaces
        let input = vec!["git"];
        let tokens = vector_into_tokens(&tokenizer, &input, None);
        assert!(!tokens.is_empty());
    }
}
