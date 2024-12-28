use crate::{
    sample_to_f32, traits::IntoF32, voice::VoiceDetection, InputSoundSender, Resampler,
    VoiceInputResult,
};

pub(crate) struct SoundStream {
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    buffer_size: usize,
    resampler: Resampler,
    voice_detection: VoiceDetection,
    channels: usize,
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
            Some(buffer_size),
            channels,
        )?;

        Ok(Self {
            input_buffer: Vec::with_capacity(buffer_size),
            output_buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            resampler,
            voice_detection,
            channels,
        })
    }

    pub(crate) fn process_input_data<T>(&mut self, raw_input: &[T], sender: &InputSoundSender)
    where
        T: cpal::Sample + IntoF32,
    {
        let input = sample_to_f32(raw_input);

        // If input is much larger than our buffer size, process it in larger chunks
        if input.len() > self.buffer_size * 10 {
            // Arbitrary threshold for "bulk" processing
            let chunk_size = self.buffer_size * self.channels;
            for chunk in input.chunks(chunk_size) {
                let resampled = self.resampler.process(chunk);

                for voice_chunk in resampled.chunks(self.buffer_size) {
                    if let Some(detected_voice) =
                        self.voice_detection.add_samples(voice_chunk.to_vec())
                    {
                        if let Err(e) = sender.send(detected_voice) {
                            eprintln!("Failed to send voice data: {:?}", e);
                        }
                    }
                }
            }
        } else {
            // Standard streaming process for smaller chunks
            self.input_buffer.extend(input);

            while self.input_buffer.len() >= self.buffer_size * self.channels {
                let chunk: Vec<_> = self
                    .input_buffer
                    .drain(..self.buffer_size * self.channels)
                    .collect();
                let resampled = self.resampler.process(&chunk);
                self.output_buffer.extend(resampled);

                while self.output_buffer.len() >= self.buffer_size {
                    let voice_chunk: Vec<_> =
                        self.output_buffer.drain(..self.buffer_size).collect();
                    if let Some(detected_voice) = self.voice_detection.add_samples(voice_chunk) {
                        if let Err(e) = sender.send(detected_voice) {
                            eprintln!("Failed to send voice data: {:?}", e);
                        }
                    }
                }
            }
        }
    }
}
