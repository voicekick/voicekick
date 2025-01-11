use hound::{WavSpec, WavWriter};
use std::fs;
use std::sync::{Arc, Mutex};
use tokio::io::{self, AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;
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

    // Buffer for accumulating samples
    let sample_buffer = Arc::new(Mutex::new(Vec::new()));
    let sample_buffer_clone = sample_buffer.clone();

    // In your recording script:
    let spec = WavSpec {
        channels: 1,        // This should match the output format
        sample_rate: 16000, // Fixed output rate
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    fs::create_dir_all("recordings").unwrap_or_else(|e| {
        println!("Warning: couldn't create recordings directory: {}", e);
    });

    let wav_writer = Arc::new(Mutex::new(
        None::<WavWriter<std::io::BufWriter<std::fs::File>>>,
    ));
    let wav_writer_clone = wav_writer.clone();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let input_sound = VoiceStreamBuilder::new(config, device, tx).build()?;

    // Constants for buffer management
    const BUFFER_THRESHOLD: usize = 4096; // Process in larger chunks

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Some(samples) => {
                    let mut buffer = sample_buffer_clone.lock().unwrap();
                    buffer.extend(samples);

                    // Process buffer when it reaches threshold
                    if buffer.len() >= BUFFER_THRESHOLD {
                        if let Some(writer) = wav_writer_clone.lock().unwrap().as_mut() {
                            // Write samples in chunks
                            for chunk in buffer.chunks(256) {
                                for &sample in chunk {
                                    if let Err(e) = writer.write_sample(sample) {
                                        eprintln!("Error writing to WAV file: {}", e);
                                    }
                                }
                            }
                        }
                        buffer.clear(); // Clear buffer after processing
                    }
                }
                _ => {}
            }
        }
    });

    let stdin = io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    println!("Enter 'r' to record or 'p' to pause. Type 'q' to quit.");

    let mut is_recording = false;
    let mut recording_count = 0;

    while let Some(line) = lines.next_line().await? {
        match line.as_str() {
            "r" => {
                if is_recording {
                    println!("Paused.");
                    input_sound.pause()?;
                    is_recording = false;

                    // Flush remaining samples
                    if let Some(writer) = wav_writer.lock().unwrap().as_mut() {
                        let remaining = sample_buffer.lock().unwrap();
                        for &sample in remaining.iter() {
                            if let Err(e) = writer.write_sample(sample) {
                                eprintln!("Error writing to WAV file: {}", e);
                            }
                        }
                    }
                } else {
                    recording_count += 1;
                    let filename = format!("recordings/recording_{}.wav", recording_count);

                    // Clear buffer before starting new recording
                    sample_buffer.lock().unwrap().clear();

                    match WavWriter::create(filename.clone(), spec) {
                        Ok(writer) => {
                            *wav_writer.lock().unwrap() = Some(writer);
                            println!("Recording... Output will be saved to {}", filename);
                            input_sound.play()?;
                            is_recording = true;
                        }
                        Err(e) => {
                            eprintln!("Error creating WAV file: {}", e);
                        }
                    }
                }
            }
            "q" => {
                println!("Quitting...");
                // Flush any remaining samples before quitting
                if let Some(writer) = wav_writer.lock().unwrap().as_mut() {
                    let remaining = sample_buffer.lock().unwrap();
                    for &sample in remaining.iter() {
                        if let Err(e) = writer.write_sample(sample) {
                            eprintln!("Error writing to WAV file: {}", e);
                        }
                    }
                }
                break;
            }
            _ => {
                println!("Unknown command. Type 'r' to record, 'p' to pause, 'q' to quit.");
            }
        }
    }

    Ok(())
}
