use std::sync::Arc;

use command_parser::{
    async_trait, CommandAction, CommandArgs, CommandOutput, CommandParser, CommandResult,
};
use voice_tests::{preprocess_samples, read_voice_dataset_wav_into_samples};
use voice_whisper::WhichModel;

struct DummyCommand;

#[async_trait]
impl CommandAction for DummyCommand {
    async fn execute(&self, args: CommandArgs) -> CommandResult {
        match args {
            CommandArgs::None => {
                println!("No arguments");
            }
            CommandArgs::Some(input) => {
                assert_eq!(input, "not");
            }
            _ => {}
        }
        Ok(CommandOutput::Ok(None))
    }
}

fn dummy_action() -> Arc<DummyCommand> {
    Arc::new(DummyCommand {})
}

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

    let parser = CommandParser::new();

    parser
        .register_namespace("americans", Some(1))
        .unwrap() // Toleration for one character difference
        .register_command("americans", "ask now", dummy_action())
        .unwrap();

    match parser.parse(got[0].dr.text.as_str()) {
        Ok((cmd, arg)) => {
            cmd.execute(arg);
        }
        Err(e) => {
            print!("parser error {:?}", e);
        }
    }
}
