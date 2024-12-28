use tracing::debug;
use voice_activity_detector::VoiceActivityDetector as SileroVoiceActivityDetector;

use crate::{VoiceInputResult, SAMPLE_RATE};

/// Silero VAD requires SAMPLE_RATE / CHUNK_SIZE > 31.25 (16000 / 512 = 31.25)
pub const SILERO_VAD_CHUNK_SIZE: usize = 512;

/// Silero VAD voice threshold
pub const SILERO_VAD_VOICE_THRESHOLD: f32 = 0.5;

/// Voice detection using Silero VAD.
/// Requires either 8000Hz or 16000 Hz sample rate.
/// Assumes incoming samples are mono channel
///
/// Silero VAD
///  - requires 8000 Hz or 16000 Hz sample rate.
///  - chunk size must be at least 512 when 16000 Hz or more is used.
///  - chunk size must be at least 256 when 8000 Hz is used.
///  - voice threshold is set to 0.1 by default.
pub struct VoiceDetection {
    samples_buffer: Vec<f32>,
    padding_buffer: Vec<f32>, // Add buffer for padding

    silero_vad: SileroVoiceActivityDetector,
    silero_vad_voice_threshold: f32,
}

impl Default for VoiceDetection {
    fn default() -> Self {
        Self::new(
            SAMPLE_RATE,
            SILERO_VAD_CHUNK_SIZE,
            SILERO_VAD_VOICE_THRESHOLD,
        )
        .expect("Valid default")
    }
}

impl VoiceDetection {
    //  - These values are recommended for optimal performance, but are not required.
    //  - The only requirement imposed by the underlying model is the sample rate must be no larger than 31.25 times the chunk size.
    pub fn new(
        incoming_sample_rate: usize,
        silero_chunk_size: usize,
        silero_vad_voice_threshold: f32,
    ) -> VoiceInputResult<Self> {
        match incoming_sample_rate {
            8000 => {
                assert!(
                    silero_chunk_size >= 256,
                    "Sample rate 8000 Hz requires chunk size >= 256"
                );
            }
            16000 => {
                assert!(
                    silero_chunk_size >= 512,
                    "Sample rate 16000 Hz requires chunk size >= 512"
                );
            }
            _ => {
                panic!("Sample rate must be 8000 or 16000 Hz");
            }
        }

        let silero_vad = SileroVoiceActivityDetector::builder()
            .sample_rate(incoming_sample_rate as i64)
            .chunk_size(silero_chunk_size)
            .build()
            .expect("valid Silero VAD configuration");

        let padding_size = (incoming_sample_rate as f32 * 0.1) as usize; // 100ms padding

        Ok(Self {
            samples_buffer: Vec::with_capacity(silero_chunk_size * 2),
            padding_buffer: Vec::with_capacity(padding_size),

            silero_vad,
            silero_vad_voice_threshold,
        })
    }

    /// Set the Silero VAD voice threshold.
    pub fn with_silero_vad_voice_threshold(mut self, threshold: f32) -> Self {
        self.silero_vad_voice_threshold = threshold;
        self
    }

    /// Add voice samples to the buffer and return the buffer if it is full and no voice is detected.
    /// Assumes 16000 or 8000 Hz sample rate.
    ///
    /// 1. Typical Pauses Between Words: In conversational speech, pauses between words or short phrases are often between 200-500 milliseconds.
    /// 2. Longer Pauses Between Sentences or Phrases: Pauses between sentences can be around 500-1000 milliseconds.
    ///
    /// Assuming the VAD operates at 16 kHz with a processing window of around 20-30 milliseconds per frame (a common setup),
    /// you could use the following approximations:
    /// 1. For Continuous Speech:
    /// - A threshold of 6-15 frames (~120-300 milliseconds) is often enough to ignore small gaps between words while capturing short pauses naturally.
    /// 2. For Detecting Sentence Boundaries or Longer Pauses:
    /// - A threshold of 25-30 frames (~500-600 milliseconds) may help capture more substantial pauses, which are typical at the end of sentences.
    ///
    /// 10-15 frames for word pauses (~300-450 milliseconds).
    /// 20-30 frames for sentence or longer pauses (~600-900 milliseconds).
    pub fn add_samples(&mut self, samples: Vec<f32>) -> Option<Vec<f32>> {
        let predict = self.silero_vad.predict(samples.clone());
        let is_voice = predict > self.silero_vad_voice_threshold;

        debug!(
            "add_samples[{}] silero predict {:.4}",
            if is_voice { "VOICE+" } else { "VOICE-" },
            predict
        );

        match is_voice {
            true => {
                // If this is the start of voice, prepend padding
                if self.samples_buffer.is_empty() && !self.padding_buffer.is_empty() {
                    self.samples_buffer.extend(&self.padding_buffer);
                }
                self.samples_buffer.extend(samples.clone());
                self.padding_buffer = samples; // Update padding buffer
                None
            }
            false => {
                if self.samples_buffer.is_empty() {
                    // Keep updating padding buffer for potential voice start
                    self.padding_buffer = samples;
                    None
                } else {
                    // Add padding to the end before returning
                    let mut result = self.samples_buffer.split_off(0);
                    result.extend(&self.padding_buffer);
                    self.padding_buffer = samples;
                    Some(result)
                }
            }
        }
    }

    /// Silero VAD prediction for a single frame
    ///
    /// Predicts the existence of speech in a single iterable of audio.
    ///
    /// The samples iterator will be padded if it is too short, or truncated if it is
    /// too long.
    ///
    /// Chunk size >= 512 when 16000 Hz or more is used
    /// Chunk size => 256 for 8000 Hz or more used
    pub fn silero_vad_prediction(&mut self, input: Vec<f32>) -> f32 {
        self.silero_vad.predict(input)
    }

    /// Silero VAD is voice?
    /// Predicts the existence of speech in a single iterable of audio.
    ///
    /// The samples iterator will be padded if it is too short, or truncated if it is
    /// too long.
    ///
    /// Chunk size >= 512 when 16000 Hz or more is used
    /// Chunk size => 256 for 8000 Hz or more used
    pub fn silero_vad_is_voice(&mut self, input: Vec<f32>) -> bool {
        self.silero_vad_prediction(input) > self.silero_vad_voice_threshold
    }
}
