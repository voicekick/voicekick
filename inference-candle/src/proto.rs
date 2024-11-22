/// DecodingResult represents the result of decoding an audio tensor.
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// The tokens of the decoded text
    pub tokens: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Average log probability of the tokens
    pub avg_logprob: f64,
    /// No speach probability
    pub no_speech_prob: f64,
    /// Temperature
    pub temperature: f64,
    /// Compression ratio
    pub compression_ratio: Option<f64>,
}

/// Represents a segment of the input audio.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time of the segment in ms
    pub start: f64,
    /// Duration of the segment in ms
    pub duration: f64,
    /// Decoding result
    pub dr: DecodingResult,
}
