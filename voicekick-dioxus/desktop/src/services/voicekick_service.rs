use dioxus::{
    hooks::{use_context, use_coroutine_handle, UnboundedReceiver},
    signals::{Readable, ReadableVecExt, WritableVecExt},
};
use futures_util::StreamExt;
use inference_candle::proto::Segment;
use tokio::sync::mpsc::{self};
use tracing::error;

use crate::states::{VoiceConfigState, VoiceState};
use voice_stream::{
    cpal::{
        self,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    },
    InputSoundSender, VoiceStream, VoiceStreamBuilder,
};
use voice_whisper::{Whisper, WhisperBuilder};

fn new_voice_stream(
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
    UpdateASR,
}

const MAX_RAW_SAMPLES: usize = 10;

pub async fn voicekick_service(mut rx: UnboundedReceiver<VoiceKickCommand>) {
    let (samples_tx, mut samples_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let mut voice_state = use_context::<VoiceState>();
    let voice_config_state = use_context::<VoiceConfigState>();
    let segment_task = use_coroutine_handle::<Segment>();

    let mut voice_stream = new_voice_stream(
        &voice_config_state.selected_input_device.read(),
        samples_tx.clone(),
        *voice_config_state.silero_voice_threshold.read(),
    )
    .expect("TODO: fix");

    let new_whisper = || -> Whisper {
        let current_model = *voice_config_state.current_model.read();
        let language = current_model
            .is_multilingual()
            .then(|| voice_config_state.current_language.read().clone());

        WhisperBuilder::infer(
            *voice_config_state.current_model.read(),
            language.as_deref(),
        )
        .expect("TODO: fix")
        .temperatures(vec![*voice_config_state.temperature.read()])
        .repetition_penalty(*voice_config_state.repetition_penalty.read())
        .repetition_frequency(*voice_config_state.repetition_frequency.read())
        .boost_value(*voice_config_state.boost_value.read())
        .command_boost_value(*voice_config_state.command_boost_value.read())
        .no_speech_threshold(*voice_config_state.no_speech_threshold.read())
        .logprob_threshold(*voice_config_state.logprob_threshold.read())
        .add_boost_words(&voice_config_state.boost_words.read(), None)
        .add_command_words(&voice_config_state.command_words.read(), None)
        .add_penalty_words(&voice_config_state.penalty_words.read(), None)
        .build()
        .expect("TODO: fix")
    };

    let mut whisper = new_whisper();

    loop {
        tokio::select! {
            command = rx.next() => {
                match command {
                    Some(VoiceKickCommand::Record) => {
                        voice_stream.play().expect("TODO: fix");
                    }
                    Some(VoiceKickCommand::Pause) => {
                        voice_stream.pause().expect("TODO: fix");
                    }
                    Some(VoiceKickCommand::UpdateVoiceStream) => {
                        voice_stream.pause().expect("TODO: fix"); // pause the current stream

                        voice_stream = new_voice_stream(
                            &voice_config_state.selected_input_device.read(),
                            samples_tx.clone(),
                            *voice_config_state.silero_voice_threshold.read(),
                        )
                        .expect("TODO: fix");
                    }
                    Some(VoiceKickCommand::UpdateASR) => {
                        voice_stream.pause().expect("TODO: fix"); // pause the current stream

                        whisper = new_whisper();
                    }
                    None => {}
                }
            }
            samples = samples_rx.recv() => {
                if let Some(new_samples) = samples {
                    if new_samples.is_empty() {
                        voice_state.raw_samples.push(vec![]); // push empty vector to keep track of silence
                    } else {
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
                                    .last()
                                    .map(|segment| segment.dr.text.is_empty())
                                    .unwrap_or(false)
                                {
                                    for segment in new_segments {
                                        segment_task.send(segment);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("with_mel_segments error {:?}", e);
                            }
                        }
                    }
                }
            }
        }
    }
}
