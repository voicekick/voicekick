use std::{
    collections::BTreeMap,
    env,
    fs::{self},
    path::Path,
};

use voice_tests::{matching_speech_commands_path, preprocess_samples, read_wav_into_samples};
use voice_whisper::WhichModel;

pub fn read_files(dir: &str) -> Vec<String> {
    let path = matching_speech_commands_path(dir);

    let mut files: Vec<String> = fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path().display().to_string())
        .collect();

    files.sort();

    files
}

pub fn speech_commands_path(path: &str) -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_HOME should be set");
    let (workspace_dir, _) = manifest_dir.rsplit_once("/").unwrap();

    format!("{workspace_dir}/voice-tests/speech-commands/{path}")
}

pub fn read_speech_commands_dirs() -> Vec<String> {
    let path = matching_speech_commands_path("");

    let mut files: Vec<String> = fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            entry
                .unwrap()
                .path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
        })
        .filter(|entry| !entry.starts_with("."))
        .collect();

    files.sort();

    files
}

pub fn speech_commands_expectations(limit: Option<usize>) -> BTreeMap<String, Vec<String>> {
    let mut expectations = BTreeMap::new();

    for dir in read_speech_commands_dirs() {
        let files = read_files(&dir);

        let mut expectations_files = Vec::new();

        for file in files.iter().take(limit.unwrap_or(files.len())) {
            expectations_files.push(file.clone());
        }

        expectations.insert(dir, expectations_files);
    }

    expectations
}

async fn test_voice_commands_whisper_model(
    model: WhichModel,
) -> (WhichModel, BTreeMap<String, usize>) {
    let mut whisper = voice_whisper::new(model, None).unwrap();

    let expectations = speech_commands_expectations(Some(3));
    let total_expectations = expectations.values().into_iter().flatten().count();

    println!("Total expectations: {}", total_expectations);

    // let temperatures = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let temperatures = vec![0.0];

    let mut matches: BTreeMap<String, usize> = BTreeMap::new();

    for (expectation, files) in expectations.iter() {
        for temperature in temperatures.iter() {
            whisper.with_temperatures(vec![*temperature]);

            for file in files.iter() {
                let (raw_samples, codec_params) = read_wav_into_samples(file);

                let resampled_samples: Vec<f32> = preprocess_samples(
                    codec_params.sample_rate.unwrap_or(0) as usize,
                    16000,
                    codec_params
                        .channels
                        .map(|channels| channels.count())
                        .unwrap_or(1),
                    raw_samples.clone(),
                );

                let segments = whisper.with_mel_segments(&resampled_samples).unwrap();
                if segments.len() == 0 {
                    continue;
                }
                let dr = &segments[0].dr;

                if dr.text == *expectation || {
                    // Allow backwards + backward to match
                    dr.text.split_whitespace().count() == 1 && expectation.starts_with(&dr.text)
                } {
                    let count = matches.entry(temperature.to_string()).or_insert(0);
                    *count += 1;
                }

                println!(
                    "TEMP {} GOT '{}' expected '{}' file {} no_speec_prop {} avg_log_prob {} temperate {}",
                    temperature,
                    &dr.text, expectation, file, dr.no_speech_prob, dr.avg_logprob, dr.temperature
                );
            }

            // assert_eq!(
            //     matches.entry(temperature.to_string()).or_insert(0),
            //     &files.len(),
            //     "Expected at temperature '{temperature}' with model {:?} to match {} files",
            //     model,
            //     files.len()
            // );
        }
    }

    (model, matches)
}

#[tokio::test]
async fn test_triage_voice_commands_whisper_english_models() {
    let triage_models = tokio::join![
        test_voice_commands_whisper_model(WhichModel::TinyEn),
        test_voice_commands_whisper_model(WhichModel::BaseEn),
        test_voice_commands_whisper_model(WhichModel::SmallEn),
        test_voice_commands_whisper_model(WhichModel::MediumEn),
    ];

    println!("{:#?}", triage_models);
}

#[allow(dead_code)]
fn whisper_dir(paths: Vec<String>, expectation: &str) -> Vec<String> {
    let mut whisper = voice_whisper::new(WhichModel::TinyEn, None).unwrap();
    whisper.with_temperatures(vec![0.5]);

    let mut output = Vec::new();

    for path in paths[0..100].iter() {
        let (raw_samples, codec_params) = read_wav_into_samples(&path);

        let resampled_samples: Vec<f32> = preprocess_samples(
            codec_params.sample_rate.unwrap_or(0) as usize,
            16000,
            codec_params
                .channels
                .map(|channels| channels.count())
                .unwrap_or(1),
            raw_samples.clone(),
        );

        let segments = whisper.with_mel_segments(&resampled_samples).unwrap();
        if segments.len() == 0 {
            continue;
        }

        let dr = &segments[0].dr;

        if dr.text == expectation {
            let to = path.replace("speech-commands", "matching-speech-commands");

            let path = Path::new(&path);
            let filename = path.file_name().unwrap().to_str().unwrap();
            output.push(format!("{expectation}/{filename}"));

            fs::rename(path, to).unwrap();
        }
    }

    println!("{expectation} => {:?}", output);

    output
}

#[tokio::test]
async fn test_whisper_triage_whisper() {
    // let dirs = vec![
    //     "four", "off", "forward", "five", "on", "yes", "six", "down", "house", "two", "marvin",
    //     "visual", "up", "seven", "zero", "bird", "one", "sheila", "three", "stop", "left", "nine",
    //     "follow", "wow", "no", "dog", "go", "happy", "bed", "tree", "learn", "backward", "cat",
    //     "eight", "right",
    // ];

    // let mut children = vec![];

    // for dirs in dirs
    //     .into_iter()
    //     .map(|d| d.to_string())
    //     .collect::<Vec<String>>()
    //     .chunks(6)
    //     .into_iter()
    //     .map(|d| d.to_vec())
    //     .collect::<Vec<Vec<String>>>()
    // {
    //     // Spin up another thread
    //     children.push(thread::spawn(move || {
    //         for dir in dirs.iter() {
    //             let _ = fs::create_dir(matching_speech_commands_path(dir));

    //             whisper_dir(read_files(dir), dir);
    //         }
    //     }));
    // }

    // for child in children {
    //     // Wait for the thread to finish. Returns a result.
    //     let _ = child.join();
    // }
}
