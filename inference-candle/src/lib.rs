use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device, Tensor,
};
use std::error::Error as StdError;
use tracing::debug;

use proto::{DecodingResult, Segment};

pub mod proto;

pub type InferenceError = Box<dyn StdError + Send + Sync>;

pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

/// Model trait
pub trait SpeechRecognitionModel {
    fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> InferenceResult<Tensor>;
    fn decoder_forward(&mut self, x: &Tensor, xa: &Tensor, flush: bool) -> InferenceResult<Tensor>;
    fn decoder_final_linear(&self, x: &Tensor) -> InferenceResult<Tensor>;
}

pub trait SpeechRecognitionDecoder {
    fn decode(&mut self, mel: &Tensor, t: f64) -> InferenceResult<DecodingResult>;
    fn decode_with_fallback(&mut self, segment: &Tensor) -> InferenceResult<DecodingResult>;
    fn segments(&mut self, mel: &Tensor) -> InferenceResult<Vec<Segment>>;
}

pub fn inference_device() -> InferenceResult<Device> {
    if cuda_is_available() {
        debug!("Cuda in use");
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        debug!("Metal in use");
        Ok(Device::new_metal(0)?)
    } else {
        debug!("CPU in use");
        Ok(Device::Cpu)
    }
}
