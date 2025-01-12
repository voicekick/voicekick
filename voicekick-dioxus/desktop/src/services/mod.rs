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
use voice_whisper::{WhichModel, Whisper, WhisperBuilder};

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

fn new_whisper(model: WhichModel, lang: &str) -> Whisper {
    let language = model.is_multilingual().then(|| lang);

    WhisperBuilder::infer(model, language)
        .expect("TODO: fix")
        .build()
        .expect("TODO: fix")
}

#[derive(Debug)]
pub enum VoiceKickCommand {
    Record,
    Pause,
    SetInputDevice(String),
    SetSileroVoiceThreshold(f32),
    SetWhisperModel(WhichModel),
    SetWhisperLanguage(String),
}

pub async fn voicekick_service(mut rx: UnboundedReceiver<VoiceKickCommand>) {
    let (samples_tx, mut samples_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let mut voice_state = use_context::<VoiceState>();

    let mut current_device: String = default_input_device().unwrap_or("".to_string());

    let mut silero_voice_threshold = SILERO_VAD_VOICE_THRESHOLD;

    let mut voice_stream =
        new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
            .expect("TODO: fix");

    let mut current_model = WhichModel::default();
    let mut current_language = String::new();

    let mut whisper = new_whisper(current_model, &current_language);

    tokio::spawn(async move {
        const MAX_RAW_SAMPLES: usize = 10;
        const MAX_SEGMENTS: usize = 30;

        loop {
            match samples_rx.try_recv() {
                Ok(new_samples) => {
                    if new_samples.is_empty() {
                        voice_state.raw_samples.push(new_samples);
                        continue;
                    }

                    // Cloning here to propogate raw_samples as fast as possible to
                    // visualize them on waveform component before actual transcribing
                    // will begin which will take some time
                    voice_state.raw_samples.push(new_samples.clone());

                    if voice_state.raw_samples.len() > MAX_RAW_SAMPLES {
                        let recent_samples = voice_state
                            .raw_samples
                            .split_off(voice_state.raw_samples.len() - MAX_RAW_SAMPLES);
                        voice_state.raw_samples.clear();
                        voice_state.raw_samples.extend(recent_samples);
                    }

                    match whisper.with_mel_segments(&new_samples) {
                        Ok(new_segments) => {
                            if !new_segments
                                .first()
                                .map(|segment| segment.dr.text.is_empty())
                                .unwrap_or(false)
                            {
                                voice_state.segments.extend(new_segments);
                                // Optionally limit number of stored segments
                                if voice_state.segments.len() > MAX_SEGMENTS {
                                    let segments = voice_state
                                        .segments
                                        .split_off(voice_state.segments.len() - MAX_SEGMENTS);
                                    voice_state.segments.clear();
                                    voice_state.segments.extend(segments);
                                }
                            }
                        }
                        Err(e) => {
                            print!("with_mel_segments error {:?}", e);
                        }
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }
    });

    while let Some(command) = rx.next().await {
        match command {
            VoiceKickCommand::Record => {
                voice_stream.play().expect("TODO: fix");
            }
            VoiceKickCommand::Pause => {
                voice_stream.pause().expect("TODO: fix");
            }
            VoiceKickCommand::SetInputDevice(new_device) => {
                if new_device == current_device {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                current_device = new_device;
                voice_stream =
                    new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
                        .expect("TODO: fix");
            }
            VoiceKickCommand::SetSileroVoiceThreshold(new_threshold) => {
                if new_threshold == silero_voice_threshold {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                silero_voice_threshold = new_threshold;
                voice_stream =
                    new_voice_stream(&current_device, samples_tx.clone(), silero_voice_threshold)
                        .expect("TODO: fix");
            }
            VoiceKickCommand::SetWhisperModel(new_model) => {
                if new_model == current_model {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                current_model = new_model;
                whisper = new_whisper(current_model, &current_language);
            }
            VoiceKickCommand::SetWhisperLanguage(new_language) => {
                if current_language == new_language {
                    continue;
                }
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                current_language = new_language;
                whisper = new_whisper(current_model, &current_language);
            }
        }
    }
}
