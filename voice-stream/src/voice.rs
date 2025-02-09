use std::time::Duration;
use tracing::{debug, warn};
use voice_activity_detector::VoiceActivityDetector as SileroVoiceActivityDetector;

use crate::{VoiceInputResult, SAMPLE_RATE};

/// Default values for voice detection configuration
pub const SILERO_VAD_CHUNK_SIZE: usize = 512;
pub const SILERO_VAD_VOICE_THRESHOLD: f32 = 0.5;

/// Voice detection configuration parameters
/// The default values are based on typical human speech patterns where:
/// - Word boundaries typically have 50-200ms pauses
/// - Sentence boundaries typically have 500-1000ms pauses
/// - Speech sounds rarely need more than 150ms padding to capture their full articulation
#[derive(Debug, Clone)]
pub struct VoiceDetectionConfig {
    /// Minimum duration of silence to consider speech ended (in milliseconds)
    pub silence_duration_threshold: u64,
    /// Duration of audio to keep before speech starts (in milliseconds)
    /// - Pre-speech padding (150ms):
    /// - Captures speech onset including plosive sounds (like 'p', 'b', 't')
    /// - Helps preserve the natural beginning of utterances
    /// - Typical plosive onset is 50-100ms, so 150ms gives some margin
    pub pre_speech_padding: u64,
    /// Duration of audio to keep after speech ends (in milliseconds)
    /// Post-speech padding (150ms):
    // - Equal to pre-speech for symmetry
    // - Sufficient to capture trailing sounds and natural decay
    // - Avoids cutting off final consonants
    pub post_speech_padding: u64,
    /// Threshold for voice activity detection (0.0 to 1.0)
    pub voice_threshold: f32,
    /// Maximum duration of a single speech segment (in milliseconds)
    pub max_speech_duration: u64,
}

impl Default for VoiceDetectionConfig {
    fn default() -> Self {
        Self {
            silence_duration_threshold: 800, // Longer pause detection (800ms)
            pre_speech_padding: 150,         // Keep 150ms before speech
            post_speech_padding: 200,        // Keep 200ms after speech
            voice_threshold: SILERO_VAD_VOICE_THRESHOLD,
            max_speech_duration: 30_000, // 30 seconds max per segment
        }
    }
}

/// Enhanced voice detection using Silero VAD with configurable parameters
/// Requires either 8000Hz or 16000 Hz sample rate.
/// Assumes incoming samples are mono channel
///
/// Silero VAD
///  - requires 8000 Hz or 16000 Hz sample rate.
///  - chunk size must be at least 512 when 16000 Hz or more is used.
///  - chunk size must be at least 256 when 8000 Hz is used.
///  - voice threshold is set to 0.1 by default.
pub struct VoiceDetection {
    // Voice detection state
    active_speech: Vec<f32>,
    padding_buffer: Vec<f32>,
    silence_frames: usize,

    // Silero VAD
    silero_vad: SileroVoiceActivityDetector,

    // Configuration
    config: VoiceDetectionConfig,
    sample_rate: usize,
    frames_per_chunk: usize,

    // Derived values
    silence_frames_threshold: usize,
    max_speech_frames: usize,
    padding_frames: usize,
}

impl Default for VoiceDetection {
    fn default() -> Self {
        Self::new(SAMPLE_RATE, SILERO_VAD_CHUNK_SIZE, None).expect("Valid default")
    }
}

impl VoiceDetection {
    pub fn new(
        sample_rate: usize,
        chunk_size: usize,
        config: Option<VoiceDetectionConfig>,
    ) -> VoiceInputResult<Self> {
        // Validate sample rate requirements
        match sample_rate {
            8000 => assert!(chunk_size >= 256, "8kHz requires chunk size >= 256"),
            16000 => assert!(chunk_size >= 512, "16kHz requires chunk size >= 512"),
            _ => panic!("Sample rate must be 8000 or 16000 Hz"),
        }

        let config = config.unwrap_or_default();

        // Initialize Silero VAD
        let silero_vad = SileroVoiceActivityDetector::builder()
            .sample_rate(sample_rate as i64)
            .chunk_size(chunk_size)
            .build()
            .expect("Valid Silero VAD configuration");

        // Calculate frame-based thresholds
        let frames_per_second = sample_rate as f32 / chunk_size as f32;
        let silence_frames_threshold =
            ((config.silence_duration_threshold as f32 / 1000.0) * frames_per_second) as usize;
        let max_speech_frames =
            ((config.max_speech_duration as f32 / 1000.0) * frames_per_second) as usize;
        let padding_frames = ((config.pre_speech_padding.max(config.post_speech_padding) as f32
            / 1000.0)
            * sample_rate as f32) as usize;

        Ok(Self {
            active_speech: Vec::with_capacity(sample_rate * 2), // 2 seconds initial capacity
            padding_buffer: Vec::with_capacity(padding_frames),
            silence_frames: 0,
            silero_vad,
            config,
            sample_rate,
            frames_per_chunk: chunk_size,
            silence_frames_threshold,
            max_speech_frames,
            padding_frames,
        })
    }

    /// Process new audio samples and detect voice activity
    ///
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
    pub fn add_samples(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let is_voice = self.silero_vad.predict(samples.to_vec()) > self.config.voice_threshold;

        debug!(
            "process_samples[{}] confidence={:.3}, silence_frames={}, buffer_size={}",
            if is_voice { "VOICE+" } else { "VOICE-" },
            self.silero_vad.predict(samples.to_vec()),
            self.silence_frames,
            self.active_speech.len()
        );

        if is_voice {
            self.handle_voice_detected(samples)
        } else {
            self.handle_silence_detected(samples)
        }
    }

    fn handle_voice_detected(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        // Reset silence counter since we detected voice
        self.silence_frames = 0;

        // If this is the start of speech, include pre-speech padding
        if self.active_speech.is_empty() && !self.padding_buffer.is_empty() {
            // Add pre-speech padding up to padding_frames
            let padding_samples = self.padding_frames.min(self.padding_buffer.len());
            self.active_speech
                .extend(&self.padding_buffer[..padding_samples]);
        }

        // Add new samples to active speech buffer
        self.active_speech.extend(samples);
        self.padding_buffer = samples.to_vec();

        // Check if we've exceeded maximum speech duration
        if self.active_speech.len() >= self.max_speech_frames * self.frames_per_chunk {
            warn!("Maximum speech duration exceeded, forcing segment break");
            let speech = std::mem::take(&mut self.active_speech);
            return Some(speech);
        }

        None
    }

    fn handle_silence_detected(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.silence_frames += 1;

        // If we have active speech, check if silence duration exceeds threshold
        if !self.active_speech.is_empty() {
            if self.silence_frames >= self.silence_frames_threshold {
                // Include post-speech padding before returning
                let mut speech = std::mem::take(&mut self.active_speech);
                speech.extend(&self.padding_buffer);
                self.padding_buffer = samples.to_vec();
                self.silence_frames = 0;
                return Some(speech);
            }
            // Continue buffering during silence period
            self.active_speech.extend(samples);
        }

        // Update padding buffer for potential future speech
        self.padding_buffer = samples.to_vec();
        None
    }

    /// Get the current voice detection configuration
    pub fn config(&self) -> &VoiceDetectionConfig {
        &self.config
    }

    /// Update voice detection configuration
    pub fn update_config(&mut self, config: VoiceDetectionConfig) {
        // Recalculate frame-based thresholds
        let frames_per_second = self.sample_rate as f32 / self.frames_per_chunk as f32;
        self.silence_frames_threshold =
            ((config.silence_duration_threshold as f32 / 1000.0) * frames_per_second) as usize;
        self.max_speech_frames =
            ((config.max_speech_duration as f32 / 1000.0) * frames_per_second) as usize;

        self.config = config;
    }

    /// Get duration of current active speech segment
    pub fn active_speech_duration(&self) -> Duration {
        Duration::from_secs_f64(self.active_speech.len() as f64 / self.sample_rate as f64)
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
        self.silero_vad_prediction(input) > self.config.voice_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_detection_config() {
        let config = VoiceDetectionConfig {
            silence_duration_threshold: 1000,
            pre_speech_padding: 200,
            post_speech_padding: 300,
            voice_threshold: 0.6,
            max_speech_duration: 20_000,
        };

        let detector = VoiceDetection::new(16000, 512, Some(config)).unwrap();
        assert_eq!(detector.config().silence_duration_threshold, 1000);
        assert_eq!(detector.config().voice_threshold, 0.6);
    }

    #[test]
    fn test_speech_duration_tracking() {
        let mut detector = VoiceDetection::new(16000, 512, None).unwrap();

        // Add 1 second of "speech" samples
        let samples = vec![0.1; 16000];
        detector.active_speech.extend(samples);

        let duration = detector.active_speech_duration();
        assert_eq!(duration.as_secs(), 1);
    }
}
