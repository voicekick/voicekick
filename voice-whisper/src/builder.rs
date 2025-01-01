use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use inference_candle::{inference_device, InferenceResult};
use tokenizers::Tokenizer;

use crate::{
    model_filenames, multilingual, token_id, vector_into_tokens, Model, WhichModel, Whisper,
    WithSpace,
};

macro_rules! create_token_setters {
    ($field_name:ident, $fn_name_set:ident, $fn_name_add:ident, $fn_name_add_words:ident) => {
        // Setter method to replace tokens
        pub fn $fn_name_set(mut self, tokens: Vec<u32>) -> Self {
            self.$field_name = tokens;
            self
        }

        // Add tokens to the existing field
        pub fn $fn_name_add(mut self, tokens: Vec<u32>) -> Self {
            self.$field_name.extend(tokens);
            self
        }

        /// Add words as tokens to the existing field
        pub fn $fn_name_add_words(mut self, words: &[&str], spacing: Option<WithSpace>) -> Self {
            let tokens = vector_into_tokens(&self.tokenizer, words, spacing);
            self.$field_name.extend(tokens);
            self
        }
    };
}

// Builder struct for Whisper
pub struct WhisperBuilder {
    device: Device,
    config: Config,
    model: Model,
    tokenizer: Tokenizer,
    language_token: Option<u32>,

    boost_tokens: Vec<u32>,
    command_tokens: Vec<u32>,
    penalty_tokens: Vec<u32>,
    temperatures: Vec<f64>,

    // Mel
    n_fft: usize,
    hop_lengnth: usize,

    repetition_penalty: f32,
    repetition_frequency: usize,
    boost_value: f32,
    command_boost_value: f32,
    no_speech_threshold: f64,
    logprob_threshold: f64,
    compression_ratio_threshold: f64,
}

impl WhisperBuilder {
    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0. This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
    ///
    /// Default: 0.0
    pub const DEFAULT_TEMPERATURE: f64 = 0.0;

    /// Init a new WhisperBuilder instance
    pub fn new(
        device: Device,
        config: Config,
        model: Model,
        tokenizer: Tokenizer,
        language_token: Option<u32>,
    ) -> Self {
        #[rustfmt::skip]
        let boost_tokens = vector_into_tokens(
            &tokenizer,
            &[
                "zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            ],
            Some(WithSpace::BeforeAndAfter)
        );

        #[rustfmt::skip]
        let command_tokens = vector_into_tokens(
            &tokenizer,
            &[
                "on", "off", "yes", "no",
                "forward", "backward", "leftward", "rightward",
                "up", "down", "left", "right",
                "start", "stop", "follow", "go", "halt",
            ],
            Some(WithSpace::Before),
        );

        #[rustfmt::skip]
        let mut penalty_tokens = vector_into_tokens(
            &tokenizer,
            &[
                // Punctuation tokens
                "!", "'", "\\", "`", "(", ")", "*", ",", ".", "..", "...",
                ":", ";", "?", "[", "]", "_", "{", "|", "}", "~",

                // Punctuation tokens with space at the end
                ". ", ", ", "! ", "? ", "... ", "; ", ": ",
            ],
            Some(WithSpace::After)
        );

        let number_tokens = vector_into_tokens(
            &tokenizer,
            &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            Some(WithSpace::Before),
        );

        penalty_tokens.extend(number_tokens);

        Self {
            device,
            config,
            model,
            tokenizer,
            language_token,
            boost_tokens,
            command_tokens,
            penalty_tokens,
            temperatures: vec![Self::DEFAULT_TEMPERATURE],
            n_fft: 400,
            hop_lengnth: 160,
            repetition_penalty: 4.2,
            repetition_frequency: 3,
            boost_value: 2.0,
            command_boost_value: 3.0,
            no_speech_threshold: 0.7,
            logprob_threshold: -1.5,
            compression_ratio_threshold: 2.4,
        }
    }

    pub fn infer(
        which_model: WhichModel,
        language: Option<&str>,
    ) -> InferenceResult<WhisperBuilder> {
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
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
            };
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

        Ok(WhisperBuilder::new(
            device,
            config,
            model,
            tokenizer,
            language_token,
        ))
    }

    create_token_setters!(
        boost_tokens,
        boost_tokens,
        add_boost_tokens,
        add_boost_words
    );

    create_token_setters!(
        command_tokens,
        command_tokens,
        add_command_tokens,
        add_command_words
    );

    create_token_setters!(
        penalty_tokens,
        penalty_tokens,
        add_penalty_tokens,
        add_penalty_words
    );

    /// Set temperatures
    ///
    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0.
    /// This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
    ///
    /// Default: 0.0
    pub fn temperatures(mut self, temps: Vec<f64>) -> Self {
        self.temperatures = temps;
        self
    }

    /// Add temperature to the existing list
    ///
    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0.
    /// This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
    ///
    /// Default: 0.0
    pub fn add_temperature(mut self, temp: f64) -> Self {
        self.temperatures.push(temp);
        self
    }

    /// Adjust this value to control penalty strength
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Adjust this value to control how many times a token must appear before penalty is applied
    pub fn repetition_frequency(mut self, frequency: usize) -> Self {
        self.repetition_frequency = frequency;
        self
    }

    /// Adjust this value to control how much we favor word numbers
    pub fn boost_value(mut self, value: f32) -> Self {
        self.boost_value = value;
        self
    }

    /// Adjust this value to control how much we favor commands
    pub fn command_boost_value(mut self, value: f32) -> Self {
        self.command_boost_value = value;
        self
    }

    /// Adjust this value to control the threshold for no speech
    pub fn no_speech_threshold(mut self, threshold: f64) -> Self {
        self.no_speech_threshold = threshold;
        self
    }

    /// Adjust this value to control the threshold for logprob
    pub fn logprob_threshold(mut self, threshold: f64) -> Self {
        self.logprob_threshold = threshold;
        self
    }

    /// Adjust this value to control the threshold for compression ratio
    pub fn compression_ratio_threshold(mut self, threshold: f64) -> Self {
        self.compression_ratio_threshold = threshold;
        self
    }

    /// For each FFT operation:
    /// - Takes 400 samples window
    /// - Overlaps by (400-128) = 272 samples with previous window
    /// - Gives ~75% overlap between consecutive windows
    /// - Results in 16000/128 ≈ 125 frames per second
    /// Each window:
    ///
    /// |----400 samples----|
    ///       |----400 samples----|
    ///             |----400 samples----|
    /// Moving by 128 samples each time
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// The overlap (N_FFT - HOP_LENGTH) helps capture smooth transitions and
    /// transient features in the audio signal.
    ///
    /// Original: (400-128)/400 = 68% overlap
    /// - 200;  // 50% overlap
    /// - 300;  // 25% overlap
    /// - 360;  // 10% overlap
    ///
    /// Tradeoff: larger HOP_LENGTH = faster processing but may miss audio transitions
    /// Minimum recommended overlap is 25%
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_lengnth = hop_length;
        self
    }

    // Build method to create the Whisper instance
    pub fn build(self) -> InferenceResult<Whisper> {
        // Create a new Whisper instance
        let no_timestamps_token = token_id(&self.tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..self.model.config().vocab_size as u32)
            .map(|i| {
                if self.model.config().suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();

        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &self.device)?;
        let sot_token = token_id(&self.tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&self.tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&self.tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&self.tokenizer, token).ok());

        let no_speech_token = match no_speech_token {
            // TODO: replace panic with bail! ?
            None => panic!("unable to find any non-speech token"),
            Some(n) => n,
        };

        let mel_bytes = match self.config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => panic!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0.0; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let space_token = vector_into_tokens(&self.tokenizer, &[" "], None)[0];

        let whisper = Whisper {
            device: self.device,
            model: self.model,
            config: self.config,
            mel_filters,
            tokenizer: self.tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            space_token,
            no_speech_token,
            language_token: self.language_token,
            no_timestamps_token,
            boost_tokens: self.boost_tokens,
            command_tokens: self.command_tokens,
            penalty_tokens: self.penalty_tokens,
            temperatures: self.temperatures,
            n_fft: self.n_fft,
            hop_lengnth: self.hop_lengnth,
            repetition_penalty: self.repetition_penalty,
            repetition_frequency: self.repetition_frequency,
            boost_value: self.boost_value,
            command_boost_value: self.command_boost_value,
            no_speech_threshold: self.no_speech_threshold,
            logprob_threshold: self.logprob_threshold,
            compression_ratio_threshold: self.compression_ratio_threshold,
        };

        Ok(whisper)
    }
}
