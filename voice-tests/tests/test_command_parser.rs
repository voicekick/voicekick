use command_parser::CommandParserBuilder;
use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};
use voice_whisper::WhichModel;

#[tokio::test]
async fn test_command_parser() {
    let (raw_samples, codec_params) = read_voice_dataset_wav_into_samples("jfk.wav");

    let mut resampled_samples: Vec<f32> = preprocess_samples(
        codec_params.sample_rate.unwrap_or(0) as usize,
        16000,
        codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1),
        raw_samples.clone(),
    );

    // 176000 / 512 = 343

    resampled_samples = resampled_samples
        .chunks(512)
        .skip(50)
        .take(100)
        .flatten()
        .cloned()
        .collect();

    let expected = "americans ask not"
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");

    let mut whisper = voice_whisper::WhisperBuilder::infer(WhichModel::TinyEn, None)
        .unwrap()
        .build()
        .unwrap();
    let got = whisper.with_mel_segments(&resampled_samples).unwrap();

    assert_eq!(got.len(), 1, "fail {:?}", got);
    similar_asserts::assert_eq!(got[0].dr.text, expected, "fail {:?}", got);

    let parser = CommandParserBuilder::new()
        .register_namespace("americans", Some(1)) // Toleration for one character difference
        .register_command("americans", "ask now", |input: Option<&str>| {
            println!("Command: {:?}", input);
            assert!(input == Some("ask not"));
        })
        .unwrap()
        .build();

    match parser.parse(got[0].dr.text.as_str()) {
        Ok((func, arg)) => {
            func(arg.as_deref());

            //
        }
        Err(e) => {
            print!("parser error {:?}", e);
        }
    }
}
