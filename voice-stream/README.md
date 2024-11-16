# Voice Stream

A Rust library for real-time voice activity detection and audio stream processing.
This library provides a high-level interface for capturing audio input, performing voice detection using both WebRTC VAD and Silero VAD, and processing audio streams.

## Features

- Real-time audio capture from input devices
- Audio resampling to desired sample rate (default 16kHz)
- Dual voice activity detection using:
  - WebRTC VAD
  - Silero VAD
- Configurable buffer sizes and voice detection parameters
- Channel-based audio data transmission
- Support for multiple sample formats (I8, I16, I32, F32)
- Conversion from multi channel to mono channel

## Usage

```rust,no_run
use voice_stream::VoiceStream;
use voice_stream::cpal::traits::StreamTrait;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a default voice stream with receiver
    let (voice_stream, receiver) = VoiceStream::default_device().unwrap();

    // Start capturing audio
    voice_stream.play().unwrap();

    // Receive voice data chunks
    for voice_data in receiver {
        // Process voice data (Vec<f32>)
        println!("Received voice data chunk of size: {}", voice_data.len());
    }

    Ok(())
}
```

## Advanced Configuration

The library provides a builder pattern for advanced configuration:

```rust,no_run
use voice_stream::{VoiceStreamBuilder, WebRtcVoiceActivityProfile};
use voice_stream::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

let (tx, rx) = std::sync::mpsc::channel();

let host = cpal::default_host();

let select_device = "default";

// Set up the input device and stream with the default input config.
let device = if select_device == "default" {
    host.default_input_device()
} else {
    host.input_devices()
        .expect("Failed to get input devices")
        .find(|x| x.name().map(|y| y == select_device).unwrap_or(false))
}
.expect("failed to find input device");

let config = device
    .default_input_config()
    .expect("Failed to get default input config");

let voice_stream = VoiceStreamBuilder::new(config, device, tx)
    .with_sound_buffer_until_size(1024)
    .with_voice_detection_silero_voice_threshold(0.5)
    .with_voice_detection_webrtc_profile(WebRtcVoiceActivityProfile::AGGRESSIVE)
    .build()
    .unwrap();
```
