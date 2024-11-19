use std::sync::mpsc;

use cpal::StreamConfig;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use tokio::io::{self, AsyncBufReadExt, BufReader};
use voice_stream::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use voice_stream::{VoiceInputError, VoiceStreamBuilder};

#[tokio::main]
async fn main() -> Result<(), VoiceInputError> {
    let host = cpal::default_host();

    let select_device = "default";

    // Set up the input device and stream with the default input config.
    let input_device = if select_device == "default" {
        host.default_input_device()
    } else {
        host.input_devices()?
            .find(|x| x.name().map(|y| y == select_device).unwrap_or(false))
    }
    .expect("failed to find input device");

    let output_device = if select_device == "default" {
        host.default_output_device()
    } else {
        host.output_devices()?
            .find(|x| x.name().map(|y| y == select_device).unwrap_or(false))
    }
    .expect("failed to find output device");

    println!("Using input device: \"{}\"", input_device.name()?);
    println!("Using output device: \"{}\"", output_device.name()?);

    let supported_config = input_device
        .default_input_config()
        .expect("Failed to get default input config");

    println!("Default input config: {:?}", supported_config);

    let config: StreamConfig = supported_config.clone().into();

    for supported_config in input_device.supported_input_configs()? {
        println!("Supported input config: {:?}", supported_config);
    }

    let latency = 1000.0;

    let mut output_config = config.clone();
    output_config.sample_rate = cpal::SampleRate(16_000);
    // output_config.buffer_size = cpal::BufferSize::Fixed(512);

    // Create a delay in case the input and output devices aren't synced.
    let latency_frames = (latency / 1_000.0) * output_config.sample_rate.0 as f32;
    let latency_samples = latency_frames as usize * output_config.channels as usize;

    let ring = HeapRb::<f32>::new(latency_samples * 100);
    let (mut producer, mut consumer) = ring.split();

    // Fill the samples with 0.0 equal to the length of the delay.
    for _ in 0..latency_samples {
        // The ring buffer has twice as much space as necessary to add latency here,
        // so this should never fail
        producer.try_push(0.0).unwrap();
    }

    let (tx, rx) = mpsc::channel();
    let input_sound = VoiceStreamBuilder::new(supported_config, input_device, tx).build()?;

    tokio::spawn(async move {
        let mut output_fell_behind = false;

        loop {
            match rx.recv() {
                Ok(samples) => {
                    for sample in samples {
                        if let Err(e) = producer.try_push(sample) {
                            output_fell_behind = true;
                            break;
                        }
                    }
                }
                Err(e) => {
                    print!("Error = {:?}", e);
                }
            }

            if output_fell_behind {
                eprintln!("output stream fell behind: try increasing latency");
            }
        }
    });

    fn err_fn(err: cpal::StreamError) {
        eprintln!("an error occurred on stream: {}", err);
    }

    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let mut input_fell_behind = false;
        for sample in data {
            *sample = match consumer.try_pop() {
                Some(s) => s,
                None => {
                    input_fell_behind = true;
                    0.0
                }
            };
        }
        if input_fell_behind {
            eprintln!("input stream fell behind: try increasing latency");
        }
    };

    let output_stream =
        output_device.build_output_stream(&output_config, output_data_fn, err_fn, None)?;
    output_stream.pause()?;

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
                    output_stream.play()?;
                } else {
                    println!("Recording...");
                    input_sound.play()?;
                    is_recording = true;
                    output_stream.pause()?;
                }
            }
            "p" => {
                println!("Paused.");
                input_sound.pause()?;
                is_recording = false;
                output_stream.play()?;
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
