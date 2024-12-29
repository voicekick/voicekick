use rubato::{FastFixedIn, Resampler as _};
use tracing::{debug, error};

use crate::VoiceInputResult;

/// Resampler for audio samples
/// - default polynomial degree: Septic
/// - default max_resample_ratio_relative: 10.0
/// - default chunk_size: 512
pub struct Resampler {
    channels: usize,

    resampler: FastFixedIn<f32>,
    resample_ratio: f64,
}

impl Resampler {
    pub fn new(
        input_sample_ratio: f64,
        output_sample_ratio: f64,
        chunk_size: Option<usize>,
        channels: usize,
    ) -> VoiceInputResult<Self> {
        let resample_ratio = output_sample_ratio / input_sample_ratio;
        let resampler_chunk_size = chunk_size.unwrap_or(512);

        debug!(
            "Creating resampler: input rate = {}, output rate = {}, ratio = {}",
            input_sample_ratio, output_sample_ratio, resample_ratio
        );

        // Use FastFixedIn for better real-time performance
        let resampler = FastFixedIn::<f32>::new(
            resample_ratio,
            1.1,
            rubato::PolynomialDegree::Septic, // Higher quality interpolation
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
            resample_ratio,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        if self.resample_ratio == 1.0 {
            // If single channel, return as-is
            if self.channels == 1 {
                return input.to_vec();
            }

            // If multi-channel and no resampling, convert to mono
            let frames = input.len() / self.channels;
            let mut mono_output = Vec::with_capacity(frames);
            for frame in 0..frames {
                let mono_sample = (0..self.channels)
                    .map(|ch| input[frame * self.channels + ch])
                    .sum::<f32>()
                    / self.channels as f32;
                mono_output.push(mono_sample);
            }
            return mono_output;
        }

        // Rest of the existing resampling logic...
        let frames = input.len() / self.channels;
        let required_frames = self.resampler.input_frames_next();

        let mut indata: Vec<Vec<f32>> = vec![vec![0.0; required_frames]; self.channels];
        for (frame, chunk) in input
            .chunks(self.channels)
            .enumerate()
            .take(required_frames)
        {
            for (ch, &sample) in chunk.iter().enumerate() {
                indata[ch][frame] = sample;
            }
        }

        let mut output = Vec::new();
        let indata_slices: Vec<&[f32]> = indata.iter().map(|v| v.as_slice()).collect();
        let mut outbuffer = vec![vec![0.0f32; self.resampler.output_frames_max()]; self.channels];

        // Process chunk
        match self
            .resampler
            .process_into_buffer(&indata_slices, &mut outbuffer, None)
        {
            Ok((_, nbr_out)) => {
                let output_frames = ((frames as f64 * self.resample_ratio) as usize).min(nbr_out);

                // Convert to mono if multi-channel
                if self.channels > 1 {
                    for frame in 0..output_frames {
                        let mono_sample = (0..self.channels)
                            .map(|ch| outbuffer[ch][frame])
                            .sum::<f32>()
                            / self.channels as f32;
                        output.push(mono_sample);
                    }
                } else {
                    // Keep original channel format for single channel
                    for frame in 0..output_frames {
                        output.push(outbuffer[0][frame]);
                    }
                }
            }
            Err(e) => {
                error!("Error processing buffer: {:?}", e);
            }
        }

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
    fn test_resampler_process_when_resample_ratio_is_same_and_one_channel() {
        let mut resampler = Resampler::new(16000.0, 16000.0, Some(4), 1).unwrap();

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let output = resampler.process(&input);

        assert_eq!(
            output,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
    }

    #[test]
    fn test_resampler_process_when_resample_ratio_is_higher_and_two_channels() {
        let mut resampler = Resampler::new(16000.0, 24000.0, Some(512), 2).unwrap();

        let input = vec![
            1.0, 2.0, // Frame 1: left=1.0, right=2.0
            3.0, 4.0, // Frame 2: left=3.0, right=4.0
            5.0, 6.0, // Frame 3: left=5.0, right=6.0
            7.0, 8.0, // Frame 4: left=7.0, right=8.0
            9.0, 10.0, // Frame 5: left=9.0, right=10.0
            11.0, 12.0, // Frame 6: left=11.0, right=12.0
            13.0, 14.0, // Frame 7: left=13.0, right=14.0
            15.0, 16.0, // Frame 8: left=15.0, right=16.0
        ];

        let output = resampler.process(&input);

        assert_eq!(
            output,
            vec![
                -0.0033531473,
                0.022227304,
                0.0,
                -0.110095,
                0.35053092,
                1.5,
                2.8261445,
                4.172255,
                5.5,
                6.8323174,
                8.166667,
                9.5
            ]
        );
    }
}
