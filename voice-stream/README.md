# Voice Stream

A Rust library for real-time voice activity detection and audio stream processing.
This library provides a high-level interface for capturing audio input,
performing voice detection using Silero VAD, and processing audio streams.

## Features

- Real-time audio capture from input devices
- Audio resampling to desired sample rate (default 16kHz)
- Voice activity detection using:
  - Silero VAD
- Configurable buffer sizes and voice detection parameters
- Channel-based audio data transmission
- Support for multiple sample formats (I8, I16, I32, F32)
- Conversion from multi channel to mono channel

## Usage

```rust,no_run
use voice_stream::VoiceStream;
use voice_stream::cpal::traits::StreamTrait;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a default voice stream with receiver
    let (voice_stream, mut rx) = VoiceStream::default_device().unwrap();

    // Start capturing audio
    voice_stream.play().unwrap();

    // Receive voice data chunks
    loop {
        match rx.recv().await {
            Some(samples) => {
              // Process voice data (Vec<f32>)
              println!("Received voice data chunk of size: {}", samples.len());
            }
            _ => {}
        }

    }

    Ok(())
}
```

### Diagram

```mermaid
flowchart TD
    Start --> Capture[Capture audio input from device mono/multi channels at various sample rates]
    Capture --> Convert
    IntoMono --> TakeBuffer[Buffer f32 samples to at least 512 size]
    TakeBuffer --> Step1[Split off samples buffer when >= 512]

    subgraph Resampler
        %% Nodes
        Convert[Convert i8, i16, i32 or f32 samples to f32]
        Resample[Resample to target sample rate 16,000 Hz]
        IntoMono[Convert multi channel sound to mono]

        %% Flow connections
        Convert --> Resample --> IntoMono
    end

    subgraph Voice Detection
        %% Nodes
        Step1[Get predict from silero_vad_prediction]
        silero[predict = silero_vad_prediction samples]
        is_voice[is_voice = predict > silero_vad_voice_threshold]
        Decision{Match is_noise, is_voice}

        %% Subgraphs for Each Case
        subgraph CaseTrueTrue ["Case: is_noise and is_voice"]
            ActionTT[Accumulate samples into samples_buffer]
            ReturnTT[Return None]
        end

        subgraph CaseTrueFalse ["Case: is_noise and !is_voice"]
            ActionTF[Clear samples_buffer]
            ReturnTF[Return None]
        end

        subgraph CaseFalse ["Case: is_noise"]
            ActionF[Push predict to silero_predict_buffer]
            BufferEmpty{Is samples_buffer empty?}
            ReturnNone[Return None]
            ReturnSamples[Return all voice samples]
        end

        %% Flow connections
        Step1 --> silero --> is_voice --> Decision

        %% Decision branches
        Decision -->|is_noise = true and is_voice = true| CaseTrueTrue
        CaseTrueTrue --> ReturnTT

        Decision -->|is_noise = true and is_voice = false| CaseTrueFalse
        CaseTrueFalse --> ReturnTF

        Decision -->|is_noise = false| CaseFalse
        CaseFalse --> BufferEmpty
        BufferEmpty -->|Yes| ReturnNone
        BufferEmpty -->|No| ReturnSamples
    end

    %%Nodes
    ProcessVoiceDetectionSamples{Process voice detection}
    ChannelSendData{Channel send}
    NoiseDiscard[Disregarded into noise void]
    User[User channel receiver]

    ReturnNone -->|None| ProcessVoiceDetectionSamples
    ReturnTT -->|None| ProcessVoiceDetectionSamples
    ReturnTF -->|None| ProcessVoiceDetectionSamples
    ReturnSamples -->|Some| ProcessVoiceDetectionSamples
    ChannelSendData --> User

    %% ChannelSendData branches
    ProcessVoiceDetectionSamples -->|Some voice| ChannelSendData

    ProcessVoiceDetectionSamples -->|No voice| NoiseDiscard
```

## Advanced Configuration

The library provides a builder pattern for advanced configuration:

```rust,no_run
use voice_stream::VoiceStreamBuilder;
use voice_stream::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::unbounded_channel();

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
    .build()
    .unwrap();
```
