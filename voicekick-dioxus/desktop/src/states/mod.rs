use dioxus::signals::{Signal, SyncSignal};

use inference_candle::proto::Segment;
use voice_stream::{default_input_device, voice::SILERO_VAD_VOICE_THRESHOLD};
use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

#[derive(Clone, Debug)]
pub struct VoiceState {
    pub raw_samples: SyncSignal<Vec<Vec<f32>>>,
    pub segments: SyncSignal<Vec<Segment>>, // Add this field
    pub is_recording: Signal<bool>,
    pub selected_input_device: Signal<String>,
    pub silero_voice_threshold: Signal<f32>,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            raw_samples: Default::default(),
            segments: Default::default(),
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
    pub temperature: Signal<f64>,
    pub repetition_penalty: Signal<f32>,
    pub repetition_frequency: Signal<usize>,
    pub boost_value: Signal<f32>,
    pub command_boost_value: Signal<f32>,
    pub no_speech_threshold: Signal<f64>,
    pub logprob_threshold: Signal<f64>,
}

impl Default for WhisperConfigState {
    fn default() -> Self {
        Self {
            current_model: Signal::new(WhichModel::default()),
            current_language: Signal::new(SUPPORTED_LANGUAGES[0].0.to_string()),
            temperature: Signal::new(0.0),
            repetition_penalty: Signal::new(4.2),
            repetition_frequency: Signal::new(3),
            boost_value: Signal::new(2.0),
            command_boost_value: Signal::new(3.0),
            no_speech_threshold: Signal::new(0.7),
            logprob_threshold: Signal::new(-1.5),
        }
    }
}
