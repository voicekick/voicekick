use earshot::{
    VoiceActivityDetector as WebRtcVoiceActivityDetector,
    VoiceActivityProfile as WebRtcVoiceActivityProfile,
};
use voice_activity_detector::VoiceActivityDetector as SileroVoiceActivityDetector;

use crate::{traits::IntoI16, Resampler, VoiceInputResult, SAMPLE_RATE};

/// Silero VAD requires SAMPLE_RATE / CHUNK_SIZE > 31.25 (16000 / 512 = 31.25)
pub const SILERO_VAD_CHUNK_SIZE: usize = 512;

/// Silero VAD voice threshold
pub const SILERO_VAD_VOICE_THRESHOLD: f32 = 0.1;

const WEBRTC_SAMPLE_RATE: usize = 8000;

// Incoming input of ~341 gets resampled into 512 samples
// if we were to split samples into 240 chunks we would get ~3 samples
// by empirically testing that all 3 samples return true as noise
// sometimes 3 (last sample) returns false as it is padded by 0
// it's good enough to take just first sample hence .take(240)
const WEBRTC_CHUNK_SIZE: usize = 240;

pub struct VoiceDetection {
    samples_buffer: Vec<f32>,
    sample_rate: usize,

    webrtc_vad: WebRtcVoiceActivityDetector,
    webrtc_resampler: Resampler,

    silero_vad: SileroVoiceActivityDetector,
    silero_vad_voice_threshold: f32,
    silero_predict_buffer: Vec<f32>,
}

impl Default for VoiceDetection {
    fn default() -> Self {
        Self::new(
            SAMPLE_RATE,
            WebRtcVoiceActivityProfile::VERY_AGGRESSIVE,
            SILERO_VAD_CHUNK_SIZE,
            SILERO_VAD_VOICE_THRESHOLD,
        )
        .expect("Valid default")
    }
}

impl VoiceDetection {
    /// Create a new voice detection instance.
    /// - Incoming sample rate must be 16000 Hz.
    /// - Expecting incoming samples to be mono channel.
    ///
    /// Silero VAD expects chunk size to be at least 512 when 16000 Hz or more is used
    //  - The model is trained using chunk sizes of 256, 512, and 768 samples for an 8000 hz sample rate.
    //  - It is trained using chunk sizes of 512, 768, 1024 samples for a 16,000 hz sample rate. These values are recommended for optimal performance, but are not required. The only requirement imposed by the underlying model is the sample rate must be no larger than 31.25 times the chunk size.
    pub fn new(
        incoming_sample_rate: usize,
        webrtc_vad_profile: WebRtcVoiceActivityProfile,
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

        // Mono channel at this point
        let channel = 1;

        let silero_vad = SileroVoiceActivityDetector::builder()
            .sample_rate(incoming_sample_rate as i64)
            .chunk_size(silero_chunk_size)
            .build()
            .expect("valid Silero VAD configuration");

        let webrtc_resampler = Resampler::new(
            incoming_sample_rate as f64,
            WEBRTC_SAMPLE_RATE as f64,
            if incoming_sample_rate == WEBRTC_SAMPLE_RATE {
                Some(silero_chunk_size)
            } else {
                Some(silero_chunk_size / 2)
            },
            channel,
        )?;

        Ok(Self {
            samples_buffer: Vec::with_capacity(silero_chunk_size * 2),
            sample_rate: incoming_sample_rate,

            webrtc_resampler,
            webrtc_vad: WebRtcVoiceActivityDetector::new(webrtc_vad_profile),

            silero_vad,
            silero_vad_voice_threshold,
            silero_predict_buffer: Vec::with_capacity(silero_chunk_size),
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
        // Step 1: Convert 16 kHz samples to 8 kHz for WebRTC VAD
        let is_noise = self.webrtc_vad_is_noise(&samples);

        // Step 2: Use WebRTC VAD to check for sound (any non-silence) and Silero VAD for voice detection
        let predict = self.silero_vad_prediction(samples.clone());
        let is_voice = predict > self.silero_vad_voice_threshold;

        // println!(
        //     "add_samples is_noise {} is_voice {} predict {:.7} SILENCE PREDICT {:.7}",
        //     if is_noise { 1 } else { 0 },
        //     if is_voice { 1 } else { 0 },
        //     predict,
        //     self.silero_predict_buffer.iter().sum::<f32>()
        //         / self.silero_predict_buffer.len() as f32,
        // );

        // Step 3: Match on the (has_sound, is_voice) tuple to handle cases
        match (is_noise, is_voice) {
            (true, true) => {
                println!(
                    "add_samples TT predict {:.7} SILENCE PREDICT {:.7}",
                    predict,
                    self.silero_predict_buffer.iter().sum::<f32>()
                        / self.silero_predict_buffer.len() as f32,
                );

                // Detected both sound and voice; accumulate samples as voice
                self.samples_buffer.extend(samples);
                None
            }
            (true, false) => {
                println!(
                    "add_samples TF predict {:.7} SILENCE PREDICT {:.7}",
                    predict,
                    self.silero_predict_buffer.iter().sum::<f32>()
                        / self.silero_predict_buffer.len() as f32,
                );

                // Detected sound but not voice; treat as noise and clear buffer
                self.samples_buffer.clear();
                None
            }
            (false, _) => {
                self.silero_predict_buffer.push(predict);

                // Silence detected (no sound); if voice samples exist, return them
                if self.samples_buffer.is_empty() {
                    None
                } else {
                    Some(self.samples_buffer.split_off(0))
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

    /// Run VAD prediction on a single frame of 16 KHz signed 16-bit mono PCM audio. Returns `Ok(true)` if the model
    /// predicts that this frame contains speech.
    ///
    /// 16000 Hz
    /// The frame must be 10ms (160 samples), 20ms (320 samples), or 30ms (480 samples) in length. An `Err` is returned
    /// if the frame size is invalid.
    ///
    /// 8000 Hz
    /// The frame must be 10ms (80 samples), 20ms (160 samples), or 30ms (240 samples) in length. An `Err` is returned
    /// if the frame size is invalid.
    ///
    /// Chunk size: 160, 320, 480
    pub fn webrtc_vad_is_noise(&mut self, input: &[f32]) -> bool {
        let mut samples_8khz: Vec<i16> = if self.sample_rate == WEBRTC_SAMPLE_RATE {
            input.to_vec()
        } else {
            self.webrtc_resampler.process(input)
        }
        .into_iter()
        .take(WEBRTC_CHUNK_SIZE)
        .map(|sample| sample.into_i16())
        .collect();

        // Pad with 0s if the frame is too short
        samples_8khz.resize(WEBRTC_CHUNK_SIZE, 0);

        self.webrtc_vad
            .predict_8khz(&samples_8khz)
            .expect("safety: valid frame size multiplier")
    }
}
