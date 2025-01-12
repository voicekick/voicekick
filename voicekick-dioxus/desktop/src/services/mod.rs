use std::sync::Arc;

use dioxus::{
    hooks::{use_context, UnboundedReceiver},
    signals::{Readable, ReadableVecExt, WritableVecExt},
};
use futures_util::StreamExt;
use tokio::sync::{
    mpsc::{self, error::TryRecvError},
    Mutex,
};
use voice_stream::{
    cpal::{
        self,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    },
    InputSoundSender, VoiceStream, VoiceStreamBuilder,
};
use voice_whisper::{Whisper, WhisperBuilder};

use crate::states::{VoiceState, WhisperConfigState};

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
pub enum VoiceKickCommand {
    Record,
    Pause,
    UpdateVoiceStream,
    UpdateWhisper,
}

pub async fn voicekick_service(mut rx: UnboundedReceiver<VoiceKickCommand>) {
    let (samples_tx, mut samples_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let mut voice_state = use_context::<VoiceState>();
    let whisper_config_state = use_context::<WhisperConfigState>();

    let mut voice_stream = new_voice_stream(
        &voice_state.selected_input_device.read(),
        samples_tx.clone(),
        voice_state.silero_voice_threshold.read().clone(),
    )
    .expect("TODO: fix");

    let new_whisper = || -> Whisper {
        let current_model = *whisper_config_state.current_model.read();
        let language = current_model
            .is_multilingual()
            .then(|| whisper_config_state.current_language.read().clone());

        println!("WHISPER STATE {:?}", whisper_config_state);

        WhisperBuilder::infer(
            *whisper_config_state.current_model.read(),
            language.as_deref(),
        )
        .expect("TODO: fix")
        .temperatures(vec![whisper_config_state.temperature.read().clone()])
        .repetition_penalty(whisper_config_state.repetition_penalty.read().clone())
        .repetition_frequency(whisper_config_state.repetition_frequency.read().clone())
        .boost_value(whisper_config_state.boost_value.read().clone())
        .command_boost_value(whisper_config_state.command_boost_value.read().clone())
        .no_speech_threshold(whisper_config_state.no_speech_threshold.read().clone())
        .logprob_threshold(whisper_config_state.logprob_threshold.read().clone())
        .build()
        .expect("TODO: fix")
    };

    let whisper = Arc::new(Mutex::new(new_whisper()));
    let whisper_clone = whisper.clone();

    tokio::spawn(async move {
        const MAX_RAW_SAMPLES: usize = 10;
        const MAX_SEGMENTS: usize = 30;

        loop {
            match samples_rx.try_recv() {
                Ok(new_samples) => {
                    if new_samples.is_empty() {
                        voice_state.raw_samples.push(vec![]); // push empty vector to keep track of silence
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

                    match whisper_clone.lock().await.with_mel_segments(&new_samples) {
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
            VoiceKickCommand::UpdateVoiceStream => {
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                voice_stream = new_voice_stream(
                    &voice_state.selected_input_device.read(),
                    samples_tx.clone(),
                    voice_state.silero_voice_threshold.read().clone(),
                )
                .expect("TODO: fix");
            }
            VoiceKickCommand::UpdateWhisper => {
                voice_stream.pause().expect("TODO: fix"); // pause the current stream

                *whisper.lock().await = new_whisper();
            }
        }
    }
}
