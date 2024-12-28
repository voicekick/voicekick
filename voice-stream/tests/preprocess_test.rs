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

    let samples = preprocess_samples(sample_rate, 16000, channels, Some(512), samples);

    // First 10 elements
    assert_eq!(
        &samples[0..10],
        &[
            0.0,
            0.0,
            0.0,
            0.0,
            -7.4505806e-8,
            2.2351742e-7,
            0.0,
            0.0,
            7.301569e-7,
            -2.6375055e-6
        ]
    );
}
