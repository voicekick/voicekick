use std::{env, sync::Arc, thread, time::Duration};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};

#[allow(unused)]
#[tokio::test]
async fn test_resampler_playback() {
    if env::var("CI").is_ok() {
        return;
    }

    let (samples, codec_params) = read_voice_dataset_wav_into_samples("Harvard list 01.wav");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    let input = preprocess_samples(sample_rate, 16000, channels, samples);
    let input_len = input.len();

    let host = cpal::default_host();

    let output_device = host
        .default_output_device()
        .expect("failed to find output device");

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let input = Arc::new(input);
    let mut sample_index = 0;

    let stream = output_device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let input = input.as_slice();
                for sample in data.iter_mut() {
                    if sample_index >= input.len() {
                        *sample = 0.0;
                    } else {
                        *sample = input[sample_index];
                        sample_index += 1;
                    }
                }
            },
            |err| eprintln!("an error occurred on stream: {}", err),
            None,
        )
        .expect("failed to build output stream");

    stream.play().expect("failed to play stream");

    // Wait for the audio to finish playing
    // let duration = Duration::from_secs_f32(input_len as f32 / 16000.0);
    let duration = Duration::from_secs(10);
    thread::sleep(duration);
}
