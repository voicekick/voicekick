use core::fmt;

use dioxus::signals::Signal;

use inference_candle::proto::Segment;
use voice_stream::{default_input_device, voice::SILERO_VAD_VOICE_THRESHOLD};
use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

#[derive(Clone, Debug)]
pub enum VoiceCommandStatus {
    Success,
    Loading,
    Failed,
    NoMatches,
}

impl fmt::Display for VoiceCommandStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::Loading => write!(f, "loading"),
            Self::Failed => write!(f, "failed"),
            Self::NoMatches => write!(f, "no matches"),
        }
    }
}

impl Default for VoiceCommandStatus {
    fn default() -> Self {
        Self::Loading
    }
}

#[derive(Clone, Debug)]
pub struct VoiceCommandSegment {
    pub segment: Segment,
    pub status: Signal<VoiceCommandStatus>,
    pub command_text: Signal<String>,
    pub execution_time: Signal<f64>,
}

#[derive(Clone, Debug)]
pub struct VoiceState {
    pub raw_samples: Signal<Vec<Vec<f32>>>,
    pub segments: Signal<Vec<VoiceCommandSegment>>,
    pub is_recording: Signal<bool>,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            raw_samples: Default::default(),
            segments: Default::default(),
            is_recording: Signal::new(false),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceConfigState {
    pub selected_input_device: Signal<String>,
    pub silero_voice_threshold: Signal<f32>,

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

impl Default for VoiceConfigState {
    fn default() -> Self {
        Self {
            selected_input_device: Signal::new(default_input_device().unwrap_or("".to_string())),
            silero_voice_threshold: Signal::new(SILERO_VAD_VOICE_THRESHOLD),
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

#[derive(Clone, Debug, Default)]
pub struct CommandsBoxState {
    pub selected_namespace: Signal<Option<String>>,
}
