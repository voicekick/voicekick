use std::env;
use std::error::Error as StdError;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;
use symphonia_core::codecs::CodecParameters;
use voice_stream::{voice::VoiceDetection, Resampler, WebRtcVoiceActivityProfile};

pub type BoxError = Box<dyn StdError + Send + Sync>;

/// Voice datasets path
pub fn voice_datasets_path(path: &str) -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_HOME should be set");
    let (workspace_dir, _) = manifest_dir.rsplit_once("/").unwrap();

    format!("{workspace_dir}/voice-tests/voice-datasets/{path}")
}

/// Matching speech commands path
pub fn matching_speech_commands_path(path: &str) -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_HOME should be set");
    let (workspace_dir, _) = manifest_dir.rsplit_once("/").unwrap();

    format!("{workspace_dir}/voice-tests/matching-speech-commands/{path}")
}

/// Read voice dataset WAV into samples
pub fn read_voice_dataset_wav_into_samples(path: &str) -> (Vec<f32>, CodecParameters) {
    read_wav_into_samples(&voice_datasets_path(path))
}

/// Read WAV into samples
pub fn read_wav_into_samples(path: &str) -> (Vec<f32>, CodecParameters) {
    let (samples, codec_params) = pcm_decode(path).expect("Failed to decode PCM");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    println!(
        "READ samples {} file '{}' sample rate {} channels {}",
        samples.len(),
        path,
        sample_rate,
        channels,
    );

    (samples, codec_params)
}

/// Preprocess samples
pub fn preprocess_samples(
    incoming_sample_rate: usize,
    outgoing_sample_rate: usize,
    channels: usize,
    chunk_size: Option<usize>,
    samples: Vec<f32>,
) -> Vec<f32> {
    let mut resampler = Resampler::new(
        incoming_sample_rate as f64,
        outgoing_sample_rate as f64,
        chunk_size,
        channels,
    )
    .expect("Failed to create Resampler");

    let before = samples.len();
    let samples = resampler.process(&samples);
    let after = samples.len();

    println!(
        "PREPROCESS RATIO {:.1} samples {} -> {}",
        after as f32 / before as f32,
        before,
        after
    );

    samples
}

pub fn new_voice_detection(
    sample_rate: usize,
    chunk_size: usize,
    siler_voice_threshold: f32,
) -> VoiceDetection {
    VoiceDetection::new(
        sample_rate,
        WebRtcVoiceActivityProfile::VERY_AGGRESSIVE,
        chunk_size,
        siler_voice_threshold,
    )
    .expect("Failed to create VoiceDetection")
}

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

pub fn pcm_decode<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(Vec<f32>, CodecParameters), BoxError> {
    // Open the media source.
    let src = std::fs::File::open(path)?;

    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, codec_params))
}
