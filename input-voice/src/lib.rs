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

/// Voice handler
pub struct VoiceInputStream {
    input_stream: Box<dyn StreamTrait>,
}

impl TryFrom<(SupportedStreamConfig, Device, InputSoundSender)> for VoiceInputStream {
    type Error = VoiceInputError;

    fn try_from(
        (config, device, tx): (SupportedStreamConfig, Device, InputSoundSender),
    ) -> Result<Self, Self::Error> {
        let input_stream = create_stream(config, device, tx)?;

        // Unfortunately, CPAL does not have a built-in mechanism for creating an input stream in a paused state.
        // This is workaround for the limitation of CPAL explicitly calling pause on the stream to ensure
        // it is paused rather than working-around lazy initialization of the stream with locks or w/e.
        let _ = input_stream.pause()?;

        Ok(Self { input_stream })
    }
}

impl VoiceInputStream {
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

        Ok((Self::try_from((config, device, tx))?, receiver))
    }
}

impl StreamTrait for VoiceInputStream {
    fn play(&self) -> Result<(), PlayStreamError> {
        self.input_stream.play()
    }

    fn pause(&self) -> Result<(), PauseStreamError> {
        self.input_stream.pause()
    }
}

/// Voice input stream
pub struct SoundStream {
    buffer: Vec<f32>,
    chunk_size: usize,

    resampler: Resampler,

    voice_detection: VoiceDetection,
}

impl SoundStream {
    pub fn new(incoming_sample_rate: usize, channels: usize) -> VoiceInputResult<Self> {
        let outgoing_sample_rate = SAMPLE_RATE;
        let chunk_size = 512;

        let resampler = Resampler::new(
            incoming_sample_rate as f64,
            outgoing_sample_rate as f64,
            Some(1024),
            channels,
        )?;

        Ok(Self {
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,

            resampler,
            voice_detection: VoiceDetection::new(
                outgoing_sample_rate,
                WebRtcVoiceActivityProfile::VERY_AGGRESSIVE,
                chunk_size,
                SILERO_VAD_VOICE_THRESHOLD,
            )
            .expect("Valid default"),
        })
    }

    fn process_input_data<T>(&mut self, input: &[T], sender: &InputSoundSender)
    where
        T: cpal::Sample + IntoF32,
    {
        // Pre-process the incoming audio samples by converting to f32,
        let samples = self.resampler.process(&input);

        self.buffer.extend(samples);

        if self.buffer.len() >= self.chunk_size {
            let buffer = self.buffer.split_off(0);

            if let Some(voice_buffer) = self.voice_detection.add_samples(buffer) {
                if let Err(e) = sender.send(voice_buffer) {
                    eprintln!("Failed to send voice data to channel: {:?}", e);
                }
            }
        }
    }
}

fn create_stream(
    supported_config: SupportedStreamConfig,
    device: Device,
    sender: InputSoundSender,
) -> VoiceInputResult<Box<dyn StreamTrait>> {
    let sample_format = supported_config.sample_format();
    let config: StreamConfig = supported_config.clone().into();

    let err_fn = move |err| {
        eprintln!("An error occurred on stream: {}", err);
    };

    let mut sound_stream = SoundStream::new(
        supported_config.sample_rate().0 as usize,
        supported_config.channels() as usize,
    )?;

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
