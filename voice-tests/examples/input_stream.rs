use tokio::sync::mpsc;
use tracing_subscriber::filter::{EnvFilter, LevelFilter};

use tokio::io::{self, AsyncBufReadExt, BufReader};
use voice_stream::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use voice_stream::{VoiceInputError, VoiceStreamBuilder};

#[tokio::main]
async fn main() -> Result<(), VoiceInputError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::DEBUG.into())
                .from_env_lossy(),
        )
        .init();

    let host = cpal::default_host();

    let select_device = "default";

    // Set up the input device and stream with the default input config.
    let device = if select_device == "default" {
        host.default_input_device()
    } else {
        host.input_devices()?
            .find(|x| x.name().map(|y| y == select_device).unwrap_or(false))
    }
    .expect("failed to find input device");

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    println!("Default input config: {:?}", config);

    for supported_config in device.supported_input_configs()? {
        println!("Supported input config: {:?}", supported_config);
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    let input_sound = VoiceStreamBuilder::new(config, device, tx).build()?;

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Some(samples) => {
                    println!("{:?}", samples);
                }
                _ => {}
            }
        }
    });

    // Create an async BufReader from stdin
    let stdin = io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    println!("Enter 'r' to record or 'p' to pause. Type 'q' to quit.");

    let mut is_recording = false;

    // Loop to handle the input from stdin
    while let Some(line) = lines.next_line().await? {
        match line.as_str() {
            "r" => {
                if is_recording {
                    println!("Paused.");
                    input_sound.pause()?;
                    is_recording = false;
                } else {
                    println!("Recording...");
                    input_sound.play()?;
                    is_recording = true;
                }
            }
            "p" => {
                println!("Paused.");
                input_sound.pause()?;
                is_recording = false;
            }
            "q" => {
                println!("Quitting...");
                break;
            }
            _ => {
                println!("Unknown command. Type 'r' to record, 'p' to pause, 'q' to quit.");
            }
        }
    }

    Ok(())
}
