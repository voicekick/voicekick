use rubato::{make_buffer, FastFixedIn, Resampler as _};
use tracing::trace;

use crate::{traits::IntoF32, VoiceInputResult};

/// Convert number type to f32
pub fn sample_to_f32<T>(input: &[T]) -> Vec<f32>
where
    T: cpal::Sample + IntoF32,
{
    input.iter().map(|sample| sample.into_f32()).collect()
}

/// Resampler for audio samples
/// - default polynomial degree: Septic
/// - default max_resample_ratio_relative: 10.0
/// - default chunk_size: 1024
pub struct Resampler {
    channels: usize,

    resampler: FastFixedIn<f32>,
    resampler_chunk_size: usize,
    resample_ratio: f64,
}

impl Resampler {
    /// Create a new Resampler
    /// input_sample_ratio: input sample rate
    /// output_sample_ratio: output sample rate
    /// chunk_size: number of samples to process at a time, default 1024
    /// channels: number of channels in the audio
    pub fn new(
        input_sample_ratio: f64,
        output_sample_ratio: f64,
        chunk_size: Option<usize>,
        channels: usize,
    ) -> VoiceInputResult<Self> {
        // https://github.com/HEnquist/rubato?tab=readme-ov-file#resampling-a-given-audio-clip
        let resample_ratio = output_sample_ratio / input_sample_ratio;
        let max_resample_ratio_relative = 10.0;

        trace!(
            "Resampler new channels {} input_sample_ratio {} output_sample_ratio {} resample_ratio {}",
            channels,
            input_sample_ratio, output_sample_ratio, resample_ratio
        );

        let resampler_chunk_size = chunk_size.unwrap_or(1024);

        let resampler = FastFixedIn::<f32>::new(
            resample_ratio,
            max_resample_ratio_relative,
            // Degree of the polynomial used for interpolation.
            // A higher degree gives a higher quality result, while taking longer to compute.
            rubato::PolynomialDegree::Septic,
            resampler_chunk_size,
            channels,
        )?;

        Ok(Self {
            channels,
            resampler,
            resampler_chunk_size,
            resample_ratio,
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

        // No need to resample if the input and output sample rates are the same
        // and channel is already mono
        if self.resample_ratio == 1.0 && self.channels == 1 {
            return input;
        }

        // assert!(
        //     input.len() % self.channels == 0,
        //     "Number of samples must be a multiple of channels"
        // );

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

        let into_channeled_samples = |raw_samples: &[f32]| -> Vec<Vec<f32>> {
            if channels > 1 {
                let mut channeled_samples = make_buffer(channels, chunk_size, false);

                for (i, raw_sample) in raw_samples.iter().enumerate() {
                    channeled_samples[i % channels].push(*raw_sample);
                }

                channeled_samples
            } else {
                // 1 channel / mono channel
                vec![raw_samples.to_vec()]
            }
        };

        let mut output: Vec<f32> = chunks
            // No need to pad samples in both if branches because of chunks_exact guarantee
            .flat_map(|samples| {
                let samples = into_channeled_samples(samples);

                self.resample_audio(&samples)
            })
            .collect();

        if !incomplete_chunk.is_empty() {
            let samples = into_channeled_samples(&incomplete_chunk);

            // Padding of the last chunk is handled by the process_partial_into_buffer
            let remainder: Vec<f32> = self.resample_partial_audio(&samples);

            output.extend(remainder);
        }

        output
    }

    fn resample_partial_audio(&mut self, input: &[Vec<f32>]) -> Vec<f32> {
        let resampled_samples = self
            .resampler
            .process_partial(Some(input), None)
            .expect("valid partial buffer");

        samples_into_mono_channel(resampled_samples)
    }

    // Re-sample the audio from input_sample_rate to output_sample_rate
    // The variation in output length (170 vs. 171) is due to the re-sampling process and how it handles fractional sample rates.
    // The rubato re-sampling algorithm rounds fractional output sample lengths, and this can vary slightly depending on
    // the input phase and how the re-sampling window aligns with the data. This is normal behavior for re-sampling algorithms,
    // especially with non-integer ratios like 48 kHz to 16 kHz.
    fn resample_audio(&mut self, input: &[Vec<f32>]) -> Vec<f32> {
        // Perform resampling on the multi-channel input
        let resampled_samples = self.resampler.process(input, None).expect("valid buffer");

        samples_into_mono_channel(resampled_samples)
    }
}

fn samples_into_mono_channel(resampled_samples: Vec<Vec<f32>>) -> Vec<f32> {
    // Convert multi-channel output to mono by averaging across channels
    let output_length = resampled_samples
        .first()
        .map(|channel| channel.len())
        .unwrap_or(0);
    // Assume all channels have the same length
    let mut mono_output = Vec::with_capacity(output_length);

    for i in 0..output_length {
        let sum: f32 = resampled_samples.iter().map(|channel| channel[i]).sum();
        mono_output.push(sum / resampled_samples.len() as f32);
    }
    mono_output
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_resampler_process_when_resample_ratio_is_one_and_mono_channel() {
        let mut resampler = Resampler::new(16000.0, 16000.0, Some(512), 1).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];

        let output = resampler.process(&input);

        assert_eq!(input, output);
    }

    #[test]
    fn test_resampler_process_when_resample_ratio_is_one_and_two_channels() {
        let mut resampler = Resampler::new(16000.0, 16000.0, Some(4), 2).unwrap();

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let output = resampler.process(&input);

        assert_eq!(output, vec![0.0, 0.0]);
    }
}
