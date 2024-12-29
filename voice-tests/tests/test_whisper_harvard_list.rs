use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};
use voice_whisper::WhichModel;

#[tokio::test]
async fn test_whisper_voice_dataset_harvard_list_01() {
    let (raw_samples, codec_params) = read_voice_dataset_wav_into_samples("Harvard list 01.wav");
    println!(
        "Original samples: {}, rate: {}, channels: {}",
        raw_samples.len(),
        codec_params.sample_rate.unwrap_or(0),
        codec_params.channels.map(|c| c.count()).unwrap_or(1)
    );

    let resampled_samples: Vec<f32> = preprocess_samples(
        codec_params.sample_rate.unwrap_or(0) as usize,
        16000,
        codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1),
        raw_samples.clone(),
    );

    println!("Resampled samples: {}", resampled_samples.len());

    // Check for silence/zero samples at the start and end
    let non_zero_samples = resampled_samples
        .iter()
        .filter(|&&x| x.abs() > 1e-6)
        .count();
    println!("Non-zero samples: {}", non_zero_samples);

    // Harvard list number one
    //
    // The birch canoe slid on the smooth planks.
    // Glue the sheet to the dark blue background.
    // It's easy to tell the depth of a well.
    // These days a chicken leg is a rare dish.
    // Rice is often served in round bowls.
    // The juice of lemons makes fine punch.
    // The box was thrown beside the parked truck.
    // The hogs were fed chopped corn and garbage.
    // Four hours of steady work faced us.
    // A large size in stockings is hard to sell.

    // Expected
    let expected = "
     harvard list number one

     the birch canoe slid on the smooth planks
     glue the sheet to the dark blue background
     it's easy to tell the depth of a well
     these days a chicken leg is a rare dish
     rice is often served in round bowls
     the juice of lemons makes fine punch
     the box was thrown beside the parked truck
     the hogs were fed chopped corn and garbage
     four hours of steady work faced us
     large size and stockings is hard to sell
    "
    .split_whitespace()
    .collect::<Vec<&str>>()
    .join(" ");

    let models = vec![WhichModel::BaseEn];

    for model in models {
        let mut whisper = voice_whisper::WhisperBuilder::infer(model, None)
            .unwrap()
            .add_boost_words(&["was"], Some(voice_whisper::WithSpace::BeforeAndAfter))
            .build()
            .unwrap();
        let segments = whisper.with_mel_segments(&resampled_samples).unwrap();

        for segment in segments.iter() {
            println!("{}", segment.dr.text);
        }

        let input = segments
            .into_iter()
            .map(|segment| segment.dr.text)
            .collect::<Vec<String>>()
            .join(" ");

        similar_asserts::assert_eq!(input, expected);
    }
}
