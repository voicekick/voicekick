use crate::{
    traits::IntoF32, voice::VoiceDetection, InputSoundSender, Resampler, VoiceInputResult,
};

/// Sound stream with VAD
pub(crate) struct SoundStream {
    buffer: Vec<f32>,
    buffer_size: usize,

    resampler: Resampler,

    voice_detection: VoiceDetection,
}

impl SoundStream {
    pub(crate) fn new(
        incoming_sample_rate: usize,
        outgoing_sample_rate: usize,
        channels: usize,
        buffer_size: usize,
        voice_detection: VoiceDetection,
    ) -> VoiceInputResult<Self> {
        let resampler = Resampler::new(
            incoming_sample_rate as f64,
            outgoing_sample_rate as f64,
            Some(1024),
            channels,
        )?;

        Ok(Self {
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,

            resampler,
            voice_detection,
        })
    }

    pub(crate) fn process_input_data<T>(&mut self, input: &[T], sender: &InputSoundSender)
    where
        T: cpal::Sample + IntoF32,
    {
        // Pre-process the incoming audio samples by converting to f32,
        let samples = self.resampler.process(&input);

        self.buffer.extend(samples);

        if self.buffer.len() >= self.buffer_size {
            let buffer = self.buffer.split_off(0);

            if let Some(voice_buffer) = self.voice_detection.add_samples(buffer) {
                // TODO: improve error handling
                if let Err(e) = sender.send(voice_buffer) {
                    eprintln!("Failed to send voice data to channel: {:?}", e);
                }
            }
        }
    }
}
