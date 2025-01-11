use dioxus::{
    hooks::{use_context, UnboundedReceiver},
    signals::{ReadableVecExt, WritableVecExt},
};
use futures_util::StreamExt;
use tokio::sync::mpsc::{self, error::TryRecvError};
use voice_stream::{
    cpal::{
        self,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    },
    default_input_device,
    voice::SILERO_VAD_VOICE_THRESHOLD,
    InputSoundSender, VoiceStream, VoiceStreamBuilder,
};

use crate::states::VoiceState;

pub fn new_voice_stream(
    selected_device: &str,
    sender: InputSoundSender,
    threshold: f32,
) -> Result<VoiceStream, String> {
    let host = cpal::default_host();

    let device = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .find(|x| x.name().map(|y| y == selected_device).unwrap_or(false))
        // fall back to default device
        .or(host.default_input_device())
        .ok_or("failed to find input device")?;

    let config = device.default_input_config().map_err(|e| e.to_string())?;

    VoiceStreamBuilder::new(config, device, sender.clone())
        .with_voice_detection_silero_voice_threshold(threshold)
        .build()
        .map_err(|e| e.to_string())
}

#[derive(Debug)]
pub enum VoiceCommand {
    Record,
    Pause,
    SetInputDevice(String),
    SetSileroVoiceThreshold(f32),
}

pub async fn voicekick_service(mut rx: UnboundedReceiver<VoiceCommand>) {
    let (samples_tx, mut samples_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let mut voice_state = use_context::<VoiceState>();

    let mut current_device: String = default_input_device().unwrap_or("".to_string());

    let mut silero_voice_threshold = SILERO_VAD_VOICE_THRESHOLD;

    let mut voice_stream =
        new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
            .expect("TODO: fix");

    tokio::spawn(async move {
        const BUFFER_SIZE: usize = 10;

        loop {
            match samples_rx.try_recv() {
                Ok(new_samples) => {
                    voice_state.raw_samples.push(new_samples);

                    if voice_state.raw_samples.len() > BUFFER_SIZE {
                        let recent_samples = voice_state
                            .raw_samples
                            .split_off(voice_state.raw_samples.len() - BUFFER_SIZE);
                        voice_state.raw_samples.clear();
                        voice_state.raw_samples.extend(recent_samples);
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }
    });

    while let Some(command) = rx.next().await {
        match command {
            VoiceCommand::Record => {
                voice_stream.play().expect("TODO: fix");
            }
            VoiceCommand::Pause => {
                voice_stream.pause().expect("TODO: fix");
            }
            VoiceCommand::SetInputDevice(new_device) => {
                if new_device == current_device {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                current_device = new_device;
                voice_stream =
                    new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
                        .expect("TODO: fix");
            }
            VoiceCommand::SetSileroVoiceThreshold(new_threshold) => {
                if new_threshold == silero_voice_threshold {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                silero_voice_threshold = new_threshold;
                voice_stream =
                    new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
                        .expect("TODO: fix");
            }
        }
    }
}
