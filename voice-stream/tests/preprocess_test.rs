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
            -3.1381132e-6,
            3.675746e-5,
            -8.2479655e-6,
            -3.122663e-5,
            1.1794421e-5,
            -9.7202734e-5,
            4.0539882e-5,
            -5.7998288e-5,
            3.5818877e-5,
            6.75999e-5
        ]
    );
}
