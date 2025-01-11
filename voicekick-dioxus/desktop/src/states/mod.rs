use dioxus::signals::{Signal, SyncSignal};
use voice_stream::{default_input_device, voice::SILERO_VAD_VOICE_THRESHOLD};
use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

#[derive(Clone, Debug)]
pub struct VoiceState {
    pub raw_samples: SyncSignal<Vec<Vec<f32>>>,
    pub is_recording: Signal<bool>,
    pub selected_input_device: Signal<String>,
    pub silero_voice_threshold: Signal<f32>,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            raw_samples: Default::default(),
            is_recording: Signal::new(false),
            selected_input_device: Signal::new(default_input_device().unwrap_or("".to_string())),
            silero_voice_threshold: Signal::new(SILERO_VAD_VOICE_THRESHOLD),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WhisperConfigState {
    pub current_model: Signal<WhichModel>,
    pub current_language: Signal<String>,
}

impl Default for WhisperConfigState {
    fn default() -> Self {
        Self {
            current_model: Signal::new(WhichModel::default()),
            current_language: Signal::new(SUPPORTED_LANGUAGES[0].0.to_string()),
        }
    }
}
