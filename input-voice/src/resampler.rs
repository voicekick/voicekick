use rubato::{make_buffer, FastFixedIn, Resampler as _};

use crate::{traits::IntoF32, VoiceInputResult};

/// Convert number type to f32
pub fn sample_to_f32<T>(input: &[T]) -> Vec<f32>
where
    T: cpal::Sample + IntoF32,
{
    input.iter().map(|sample| sample.into_f32()).collect()
}

/// Voice input stream
pub struct Resampler {
    channels: usize,

    resampler: FastFixedIn<f32>,
    resampler_buffer: Vec<Vec<f32>>,
    resampler_chunk_size: usize,
}

impl Resampler {
    pub fn new(
        input_sample_ratio: f64,
        output_sample_ratio: f64,
        resampler_chunk_size: usize,
        channels: usize,
    ) -> VoiceInputResult<Self> {
        // https://github.com/HEnquist/rubato?tab=readme-ov-file#resampling-a-given-audio-clip
        let resample_ratio = output_sample_ratio / input_sample_ratio;
        let max_resample_ratio_relative = 10.0;

        println!(
            "Resampler new channels {} input_sample_ratio {} output_sample_ratio {} resample_ratio {}",
            channels,
            input_sample_ratio, output_sample_ratio, resample_ratio
        );

        let resampler = FastFixedIn::<f32>::new(
            resample_ratio,
            max_resample_ratio_relative,
            // Degree of the polynomial used for interpolation.
            // A higher degree gives a higher quality result, while taking longer to compute.
            rubato::PolynomialDegree::Septic,
            resampler_chunk_size,
            channels,
        )?;

        let resampler_buffer = resampler.output_buffer_allocate(true);

        Ok(Self {
            channels,
            resampler_buffer,
            resampler,
            resampler_chunk_size,
        })
    }

    /// Pre-process the incoming audio samples by converting to f32,
    /// averaging across channels, and re-sampling to 16 kHz.
    pub fn process<T>(&mut self, input: &[T]) -> Vec<f32>
    where
        T: cpal::Sample + IntoF32,
    {
        // Convert the incoming audio data to f32
        let input = sample_to_f32(input);

        // When there are more than 1 channels in the audio, process the samples in chunks
        // each chunk must be a multiple of the number of channels
        // since the samples are interleaved by channel in the input buffer
        // and the resampler expects samples in chunks of `resampler_chunk_size`
        let chunks = input.chunks_exact(self.resampler_chunk_size * self.channels);

        // The last chunk may be incomplete, so we need to handle it separately
        // allocating to vector because chunks gets consumed
        let incomplete_chunk: Vec<f32> = chunks.remainder().to_vec();

        let channels = self.channels;
        let chunk_size = self.resampler_chunk_size;

        let into_mono_samples = |raw_samples: &[f32]| -> Vec<Vec<f32>> {
            if channels > 1 {
                let mut mono_samples = make_buffer(channels, chunk_size, false);

                for (i, sample) in raw_samples.iter().enumerate() {
                    mono_samples[i % channels].push(*sample);
                }

                mono_samples
            } else {
                vec![raw_samples.to_vec()]
            }
        };

        let mut output: Vec<f32> = chunks
            // No need to pad samples in both if branches because of chunks_exact guarantee
            .flat_map(|samples| {
                let samples = into_mono_samples(samples);

                self.resample_audio(&samples)
            })
            .collect();

        if incomplete_chunk.len() > 0 {
            let samples = into_mono_samples(&incomplete_chunk);

            let remainder: Vec<f32> = self.resample_partial_audio(&samples);

            output.extend(remainder);
        }

        output
    }

    fn resample_partial_audio(&mut self, input: &[Vec<f32>]) -> Vec<f32> {
        let (_chunk_size, output_length) = self
            .resampler
            .process_partial_into_buffer(Some(&input), &mut self.resampler_buffer, None)
            .expect("valid buffer");

        // Truncate the output buffer to the actual length of the re-sampled audio
        // note: slice is used to avoid cloning the entire buffer
        self.resampler_buffer
            .iter()
            .flat_map(|channel| &channel[0..output_length])
            .cloned()
            .collect()
    }

    // Re-sample the audio from input_sample_rate to output_sample_rate
    // The variation in output length (170 vs. 171) is due to the re-sampling process and how it handles fractional sample rates.
    // The rubato re-sampling algorithm rounds fractional output sample lengths, and this can vary slightly depending on
    // the input phase and how the re-sampling window aligns with the data. This is normal behavior for re-sampling algorithms,
    // especially with non-integer ratios like 48 kHz to 16 kHz.
    fn resample_audio(&mut self, input: &[Vec<f32>]) -> Vec<f32> {
        let (_chunk_size, output_length) = self
            .resampler
            .process_into_buffer(&input, &mut self.resampler_buffer, None)
            .expect("valid buffer");

        // Truncate the output buffer to the actual length of the re-sampled audio
        // note: slice is used to avoid cloning the entire buffer
        self.resampler_buffer
            .iter()
            .flat_map(|channel| &channel[0..output_length])
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_vector_resize_overwrite() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        input.resize(10, 0.0);

        assert_eq!(
            input,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        input.resize(3, 0.0);

        assert_eq!(input, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_with_push() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        input.resize(10, 0.0);

        assert_eq!(
            input,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        input.push(5.0);

        assert_eq!(
            input,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
        );
    }
}
