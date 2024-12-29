use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};

#[tokio::test]
async fn test_preprocess_samples() {
    let (samples, codec_params) =
        read_voice_dataset_wav_into_samples("24bit-M1F1-int24WE-AFsp.wav");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    let samples = preprocess_samples(sample_rate, 16000, channels, samples);

    // First 10 elements
    assert_eq!(
        &samples[0..10],
        &[
            0.0,
            0.0,
            7.4505806e-8,
            0.0,
            -9.536743e-7,
            0.0,
            5.990267e-6,
            0.0,
            -3.066659e-5,
            -3.0517578e-5
        ]
    );
}
