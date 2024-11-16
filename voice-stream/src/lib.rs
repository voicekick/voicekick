#![doc = include_str!("../README.md")]

// Re-export
pub use cpal;

// Re-export
pub use earshot::{
    VoiceActivityDetector as WebRtcVoiceActivityDetector,
    VoiceActivityProfile as WebRtcVoiceActivityProfile,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, PauseStreamError, PlayStreamError, SampleFormat, StreamConfig, SupportedStreamConfig,
};
use std::{
    sync::mpsc::{self, Receiver, Sender},
    time::Duration,
};
use traits::IntoF32;
use voice::{VoiceDetection, SILERO_VAD_VOICE_THRESHOLD};

use std::error::Error as StdError;

pub type InputSoundSender = Sender<Vec<f32>>;
pub type InputSoundReceiver = Receiver<Vec<f32>>;
pub type InputSoundChannel = (InputSoundSender, InputSoundReceiver);

pub type VoiceInputError = Box<dyn StdError + Send + Sync>;

pub type VoiceInputResult<T> = std::result::Result<T, VoiceInputError>;

/// Traits
pub mod traits;

/// Voice detection
pub mod voice;

/// Resampler
mod resampler;
pub use resampler::Resampler;

mod sound;
use sound::SoundStream;

/// Default sample rate
pub const SAMPLE_RATE: usize = 16000;

/// Convert number type to f32
pub fn sample_to_f32<T>(input: &[T]) -> Vec<f32>
where
    T: cpal::Sample + IntoF32,
{
    input.iter().map(|sample| sample.into_f32()).collect()
}

/// Duration from samples
pub fn samples_to_duration(samples: usize, sample_rate: Option<f64>) -> Duration {
    let seconds = samples as f64 / sample_rate.unwrap_or(SAMPLE_RATE as f64);
    Duration::from_secs_f64(seconds)
}

/// Voice input stream builder
pub struct VoiceStreamBuilder {
    supported_config: SupportedStreamConfig,
    device: Device,
    tx: InputSoundSender,
    sound_stream_samples_buffer_size: usize,

    voice_detection_webrtc_profile: WebRtcVoiceActivityProfile,
    voice_detection_silero_threshold: f32,
}

impl VoiceStreamBuilder {
    /// Create a new voice input stream builder
    pub fn new(
        supported_config: SupportedStreamConfig,
        device: Device,
        tx: InputSoundSender,
    ) -> Self {
        Self {
            supported_config,
            device,
            tx,
            sound_stream_samples_buffer_size: 512,
            voice_detection_webrtc_profile: WebRtcVoiceActivityProfile::VERY_AGGRESSIVE,
            voice_detection_silero_threshold: SILERO_VAD_VOICE_THRESHOLD,
        }
    }

    pub fn with_sound_buffer_until_size(mut self, buffer_size: usize) -> Self {
        match self.supported_config.sample_rate().0 {
            0..8000 => {
                unimplemented!("Silero VAD does not support sample rates below 8khz");
            }
            8000..16000 => {
                assert!(
                    buffer_size >= 256,
                    "Silero VAD requires buffer size >= 256 at 8khz to 16khz"
                );
            }
            16000.. => {
                assert!(
                    buffer_size >= 512,
                    "Silero VAD requires buffer size >= 512 at 16khz or more"
                );
            }
        }

        self.sound_stream_samples_buffer_size = buffer_size;
        self
    }

    pub fn with_voice_detection_silero_voice_threshold(mut self, threshold: f32) -> Self {
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "Silero VAD voice threshold must be between 0.0 and 1.0"
        );
        self.voice_detection_silero_threshold = threshold;

        self
    }

    pub fn with_voice_detection_webrtc_profile(
        mut self,
        profile: WebRtcVoiceActivityProfile,
    ) -> Self {
        self.voice_detection_webrtc_profile = profile;
        self
    }

    /// Build a voice input stream
    pub fn build(self) -> Result<VoiceStream, VoiceInputError> {
        let outgoing_sample_rate = SAMPLE_RATE;

        let voice_detection = VoiceDetection::new(
            outgoing_sample_rate,
            self.voice_detection_webrtc_profile,
            self.sound_stream_samples_buffer_size,
            self.voice_detection_silero_threshold,
        )?;

        let sound_stream = SoundStream::new(
            self.supported_config.sample_rate().0 as usize,
            outgoing_sample_rate,
            self.supported_config.channels() as usize,
            self.sound_stream_samples_buffer_size,
            voice_detection,
        )?;

        let input_stream =
            create_stream(sound_stream, self.supported_config, self.device, self.tx)?;

        // Unfortunately, CPAL does not have a built-in mechanism for creating an input stream in a paused state.
        // This is workaround for the limitation of CPAL explicitly calling pause on the stream to ensure
        // it is paused rather than working-around lazy initialization of the stream with locks or w/e.
        let _ = input_stream.pause()?;

        Ok(VoiceStream { input_stream })
    }
}

/// Voice handler
///
/// 1. Captures audio from the given input device
/// 2. Resamples the audio to the desired sample rate (16kHz default)
/// 3. Detects voice activity using WebRTC VAD and Silero VAD
/// 4. Sends the voice data to the receiver channel in chunks
/// 5. The receiver channel can expect to receive voice data in chunks of 512<= samples
pub struct VoiceStream {
    input_stream: Box<dyn StreamTrait>,
}

impl VoiceStream {
    /// Initialize input sound handler
    pub fn default_device() -> VoiceInputResult<(Self, InputSoundReceiver)> {
        let host = cpal::default_host();

        // Set up the input device and stream with the default input config.
        let device = host
            .default_input_device()
            .expect("failed to find default input device");

        let config = device
            .default_input_config()
            .expect("Failed to get default input config");

        let (tx, receiver) = mpsc::channel();

        Ok((
            VoiceStreamBuilder::new(config, device, tx).build()?,
            receiver,
        ))
    }
}

impl StreamTrait for VoiceStream {
    fn play(&self) -> Result<(), PlayStreamError> {
        self.input_stream.play()
    }

    fn pause(&self) -> Result<(), PauseStreamError> {
        self.input_stream.pause()
    }
}

fn create_stream(
    mut sound_stream: SoundStream,
    supported_config: SupportedStreamConfig,
    device: Device,
    sender: InputSoundSender,
) -> VoiceInputResult<Box<dyn StreamTrait>> {
    let sample_format = supported_config.sample_format();
    let config: StreamConfig = supported_config.into();

    let err_fn = move |err| {
        eprintln!("An error occurred on stream: {}", err);
    };

    let stream = match sample_format {
        SampleFormat::I8 => device.build_input_stream(
            &config,
            move |data, _: &_| sound_stream.process_input_data::<i8>(data, &sender),
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data, _: &_| sound_stream.process_input_data::<i16>(data, &sender),
            err_fn,
            None,
        )?,
        SampleFormat::I32 => device.build_input_stream(
            &config,
            move |data, _: &_| sound_stream.process_input_data::<i32>(data, &sender),
            err_fn,
            None,
        )?,
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data, _: &_| sound_stream.process_input_data::<f32>(data, &sender),
            err_fn,
            None,
        )?,
        sample_format => {
            unimplemented!(
                "cpal::SampleFormat unsupported sample format: {:?}",
                sample_format
            );
        }
    };

    Ok(Box::new(stream))
}
