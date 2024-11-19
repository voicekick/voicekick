# Voice Testing Utilities

This library provides utilities for processing and testing voice audio files, particularly focused on speech command recognition and voice activity detection. It uses the Symphonia audio processing library for decoding various audio formats and includes tools for working with open source voice datasets.

## Features

- Audio file decoding (WAV and other formats) using Symphonia
- Voice activity detection using WebRTC VAD
- Audio sample preprocessing and resampling
- Support for managing speech command datasets
- Utilities for reading and organizing voice test files

## Voice Datasets

The library is designed to work with open source voice datasets. It expects datasets to be organized in the following directory structure:

```
voice-tests/
├── voice-datasets/      # General voice samples
└── matching-speech-commands/  # Organized speech commands for testing
```

Used datasets:

 - Voice commands https://github.com/sankalp2610/Speech_Command_Recognition
 - Obama voice samples https://huggingface.co/datasets/RaysDipesh/obama-voice-samples-283
 - WAVE Sample Files https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples.html
 - JFK https://huggingface.co/datasets/Xenova/transformers.js-docs/tree/main
 - Hardvard Sentences https://www.cs.columbia.edu/~hgs/audio/harvard.html

### Compatible Datasets

You can use this library with various open source voice datasets, such as:

- Google Speech Commands Dataset
- Mozilla Common Voice
- VoxForge
- LibriSpeech

## Usage

The library provides several key functions for working with voice data:

```rust
// Read and process WAV files
let (samples, codec_params) = read_wav_into_samples("path/to/audio.wav");

// Preprocess audio samples (resampling)
let processed = preprocess_samples(
    44100,  // input sample rate
    16000,  // output sample rate
    1,      // channels
    None,   // chunk size
    samples
);

// Initialize voice detection
let vad = new_voice_detection(
    16000,  // sample rate
    480,    // chunk size
    0.5     // voice threshold
);
```

## Dependencies

- `symphonia` - Audio decoding and processing
- `voice_stream` - Voice activity detection and audio processing

## License

This project is available under the MIT License. Note that while this code is open source, when using voice datasets, make sure to comply with their respective licenses and terms of use.
