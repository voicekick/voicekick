use candle_core::Device;
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
}

impl WhisperBuilder {
    /// A good starting temperature for the Whisper model is typically around 0.7 to 1.0. This range allows for a balance between deterministic and diverse outputs:
    /// - Lower temperatures (e.g., 0.5 or below) produce more consistent and predictable outputs, suitable for structured tasks where clarity is key.
    /// - Higher temperatures (e.g., 1.0 and above) introduce more variability, allowing the model to generate more creative or varied responses.
    ///
    /// Start at 0.7 for balanced behavior and adjust based on whether you want more stability or diversity in your output.
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

    pub fn temperatures(mut self, temps: Vec<f64>) -> Self {
        self.temperatures = temps;
        self
    }

    /// Add temperature to the existing list
    pub fn add_temperature(mut self, temp: f64) -> Self {
        self.temperatures.push(temp);
        self
    }

    // Build method to create the Whisper instance
    pub fn build(self) -> InferenceResult<Whisper> {
        // Create a new Whisper instance
        let mut whisper = Whisper::new(
            self.device,
            self.config,
            self.model,
            self.tokenizer,
            self.language_token,
        )?;

        whisper.boost_tokens = self.boost_tokens;

        whisper.command_tokens = self.command_tokens;

        whisper.penalty_tokens = self.penalty_tokens;

        whisper.temperatures = self.temperatures;

        Ok(whisper)
    }
}
