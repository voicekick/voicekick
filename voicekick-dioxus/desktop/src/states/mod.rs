use dioxus::signals::Signal;
use tokio::sync::broadcast;

use inference_candle::proto::Segment;
use voice_stream::{default_input_device, voice::SILERO_VAD_VOICE_THRESHOLD};
use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

#[derive(Debug)]
pub struct VoiceSegmentState {
    pub segment_tx: broadcast::Sender<Segment>,
    pub segment_rx: broadcast::Receiver<Segment>,
}

impl Default for VoiceSegmentState {
    fn default() -> Self {
        let (segment_tx, segment_rx) = broadcast::channel(1000);
        Self {
            segment_tx,
            segment_rx,
        }
    }
}

impl Clone for VoiceSegmentState {
    fn clone(&self) -> Self {
        Self {
            segment_tx: self.segment_tx.clone(),
            segment_rx: self.segment_tx.subscribe(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceState {
    pub raw_samples: Signal<Vec<Vec<f32>>>,
    pub segments: Signal<Vec<Segment>>,
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
