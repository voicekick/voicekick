use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};
use voice_whisper::WhichModel;

#[tokio::test]
async fn test_whisper_voice_dataset_jfk() {
    let (raw_samples, codec_params) = read_voice_dataset_wav_into_samples("jfk.wav");

    let resampled_samples: Vec<f32> = preprocess_samples(
        codec_params.sample_rate.unwrap_or(0) as usize,
        16000,
        codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1),
        raw_samples.clone(),
    );

    let models = vec![WhichModel::TinyEn, WhichModel::BaseEn, WhichModel::SmallEn];

    let expected = "
        and so my fellow americans ask not what
        your country can do for you ask what you can do for your country
    "
    .split_whitespace()
    .collect::<Vec<&str>>()
    .join(" ");

    for model in models {
        let mut whisper = voice_whisper::new(model, None).unwrap();
        let got = whisper.with_mel_segments(&resampled_samples).unwrap();

        assert_eq!(got.len(), 1, "fail {:?}", got);
        similar_asserts::assert_eq!(got[0].dr.text, expected, "fail {:?} {:?}", model, got);
    }
}
