use rubato::{make_buffer, FastFixedIn, Resampler as _, SincFixedIn, SincFixedOut};
use tracing::debug;

use crate::VoiceInputResult;

/// Resampler for audio samples
/// - default polynomial degree: Septic
/// - default max_resample_ratio_relative: 10.0
/// - default chunk_size: 1024
pub struct Resampler {
    channels: usize,

    resampler: SincFixedOut<f32>,
    resampler_chunk_size: usize,
    resample_ratio: f64,
}

impl Resampler {
    pub fn new(
        input_sample_ratio: f64,
        output_sample_ratio: f64,
        chunk_size: Option<usize>,
        channels: usize,
    ) -> VoiceInputResult<Self> {
        // Correct ratio for downsampling (high rate to low rate)
        let resample_ratio = output_sample_ratio / input_sample_ratio;

        let max_resample_ratio_relative = 1.1;

        let resampler_chunk_size = chunk_size.unwrap_or(1024);

        debug!(
            "Creating resampler: input rate = {}, output rate = {}, ratio = {}",
            input_sample_ratio, output_sample_ratio, resample_ratio
        );

        let sinc_len = 128;
        let window = rubato::WindowFunction::Blackman2;
        let f_cutoff = rubato::calculate_cutoff(sinc_len, window);

        // Adjusted parameters for more precise resampling
        let params = rubato::SincInterpolationParameters {
            sinc_len,
            f_cutoff,
            interpolation: rubato::SincInterpolationType::Cubic,
            oversampling_factor: 512,
            window,
        };

        let resampler = SincFixedOut::<f32>::new(
            resample_ratio,
            max_resample_ratio_relative,
            params,
            resampler_chunk_size,
            channels,
        )?;

        debug!(
            "Next input frames needed: {}",
            resampler.input_frames_next()
        );
        debug!("Max output frames: {}", resampler.output_frames_max());

        Ok(Self {
            channels,
            resampler,
            resampler_chunk_size,
            resample_ratio,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        // Early return if no resampling needed
        if self.resample_ratio == 1.0 && self.channels == 1 {
            return input.to_vec();
        }

        debug!("Input samples: {}", input.len());

        // Create separate channel vectors
        let mut indata: Vec<Vec<f32>> = vec![Vec::new(); self.channels];
        let frames = input.len() / self.channels;

        debug!("Input frames per channel: {}", frames);

        // Fill channel vectors
        for frame in 0..frames {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                indata[ch].push(input[idx]);
            }
        }

        // Create output buffer with enough capacity
        let out_frames = (frames as f64 * self.resample_ratio).ceil() as usize;
        let mut outdata = vec![Vec::with_capacity(out_frames); self.channels];

        // Create slices for input
        let mut indata_slices: Vec<&[f32]> = indata.iter().map(|v| v.as_slice()).collect();

        // Create output buffer for processing
        let mut outbuffer = vec![vec![0.0f32; self.resampler.output_frames_max()]; self.channels];
        let mut input_frames_next = self.resampler.input_frames_next();

        // Process full chunks
        let mut output = Vec::new();

        while indata_slices[0].len() >= input_frames_next {
            if let Ok((nbr_in, nbr_out)) =
                self.resampler
                    .process_into_buffer(&indata_slices, &mut outbuffer, None)
            {
                // Append the processed frames
                for ch in 0..self.channels {
                    outdata[ch].extend_from_slice(&outbuffer[ch][..nbr_out]);
                }
                // Update the input slices
                for ch in 0..self.channels {
                    indata_slices[ch] = &indata_slices[ch][nbr_in..];
                }
            }
            input_frames_next = self.resampler.input_frames_next();
        }

        // Process remaining samples if any
        if !indata_slices[0].is_empty() {
            if let Ok((_nbr_in, nbr_out)) = self.resampler.process_partial_into_buffer(
                Some(&indata_slices),
                &mut outbuffer,
                None,
            ) {
                for ch in 0..self.channels {
                    outdata[ch].extend_from_slice(&outbuffer[ch][..nbr_out]);
                }
            }
        }

        // Convert channel buffers back to interleaved format
        let out_frames = outdata[0].len();
        for frame in 0..out_frames {
            for ch in 0..self.channels {
                output.push(outdata[ch][frame]);
            }
        }

        debug!("Output frames per channel: {}", outdata[0].len());
        debug!("Total output samples: {}", output.len());

        output
    }
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

        assert_eq!(
            output,
            vec![
                1.3045013, 2.2875674, 2.641381, 3.6544528, 5.425432, 6.4173446, 6.5281663, 7.530723
            ]
        );
    }
}
