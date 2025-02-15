// Audio processing code, adapted from whisper.cpp
// https://github.com/ggerganov/whisper.cpp
// - constrained to f32
// - improved remainder padding handling

use candle_core::utils::get_num_threads;
use std::f32::consts::PI;
use std::sync::Arc;
use std::thread;
use tracing::debug;

const TWO_PI: f32 = PI + PI;
const ZERO: f32 = 0.0;
const HALF: f32 = 0.5;
const ONE: f32 = 1.0;
const FOUR: f32 = 4.0;

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 1 {
        return vec![inp[0], ZERO];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![ZERO; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let n_t = n as f32;
    for k in 0..n / 2 {
        let k_t = k as f32;
        let theta = TWO_PI * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn dft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();

    let mut out = Vec::with_capacity(2 * n);
    let n_t = n as f32;
    for k in 0..n {
        let k_t = k as f32;
        let mut re = ZERO;
        let mut im = ZERO;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = j as f32;
            let angle = TWO_PI * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn log_mel_spectrogram_w(
    ith: usize,
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    speed_up: bool,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<f32> {
    let n_fft = if speed_up {
        1 + fft_size / 4
    } else {
        1 + fft_size / 2
    };

    let mut fft_in = vec![ZERO; fft_size];
    let mut mel = vec![ZERO; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);

    for i in (ith..end).step_by(n_threads) {
        let offset = i * fft_step;

        // apply Hanning window
        for j in 0..std::cmp::min(fft_size, n_samples - offset) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with ZEROs
        if n_samples - offset < fft_size {
            fft_in[n_samples - offset..].fill(ZERO);
        }

        // FFT
        let mut fft_out: Vec<f32> = fft(&fft_in);

        // Calculate modulus^2 of complex numbers
        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j];
            fft_out[j] += v;
        }

        if speed_up {
            // scale down in the frequency domain results in a speed up in the time domain
            for j in 0..n_fft {
                fft_out[j] = HALF * (fft_out[2 * j] + fft_out[2 * j + 1]);
            }
        }

        // mel spectrogram
        for j in 0..n_mel {
            let mut sum = ZERO;
            let mut k = 0;
            // Unroll loop
            while k < n_fft.saturating_sub(3) {
                sum += fft_out[k] * filters[j * n_fft + k]
                    + fft_out[k + 1] * filters[j * n_fft + k + 1]
                    + fft_out[k + 2] * filters[j * n_fft + k + 2]
                    + fft_out[k + 3] * filters[j * n_fft + k + 3];
                k += 4;
            }
            // Handle remainder
            while k < n_fft {
                sum += fft_out[k] * filters[j * n_fft + k];
                k += 1;
            }
            mel[j * n_len + i] = f32::max(sum, 1e-10).log10();
        }
    }
    mel
}

fn log_mel_spectrogram(
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
    speed_up: bool,
) -> Vec<f32> {
    let fft_size_t = fft_size as f32;

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| HALF * (ONE - ((TWO_PI * i as f32) / fft_size_t).cos()))
        .collect();
    let n_len = samples.len() / fft_step;

    // Minimal padding - just enough to complete the last FFT window
    let remainder = samples.len() % fft_step;
    let pad = if remainder > 0 {
        fft_step - remainder
    } else {
        0
    };

    debug!(
        "samples {} fft_size {fft_size} fft_step {fft_step} n_len {} remainder {} pad {}",
        samples.len(),
        n_len,
        remainder,
        pad
    );

    let samples = if pad > 0 {
        let mut samples_padded = samples.to_vec();
        samples_padded.extend(std::iter::repeat(ZERO).take(pad));
        samples_padded
    } else {
        samples.to_vec()
    };

    // ensure that the number of threads is even and less than 12
    let n_threads = std::cmp::min(get_num_threads() - get_num_threads() % 2, 12);

    let hann = Arc::new(hann);
    let samples = Arc::new(samples);
    let filters = Arc::new(filters);

    // use scope to allow for non static references to be passed to the threads
    // and directly collect the results into a single vector
    let all_outputs = thread::scope(|s| {
        (0..n_threads)
            // create threads and return their handles
            .map(|thread_id| {
                let hann = Arc::clone(&hann);
                let samples = Arc::clone(&samples);
                let filters = Arc::clone(&filters);
                // spawn new thread and start work
                s.spawn(move || {
                    log_mel_spectrogram_w(
                        thread_id, &hann, &samples, &filters, fft_size, fft_step, speed_up, n_len,
                        n_mel, n_threads,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            // wait for each thread to finish and collect their results
            .map(|handle| handle.join().expect("Thread failed"))
            .collect::<Vec<_>>()
    });

    let l = all_outputs[0].len();
    let mut mel = vec![ZERO; l];

    // iterate over mel spectrogram segments, dividing work by threads.
    for segment_start in (0..l).step_by(n_threads) {
        // go through each thread's output.
        for thread_output in all_outputs.iter() {
            // add each thread's piece to our mel spectrogram.
            for offset in 0..n_threads {
                let mel_index = segment_start + offset; // find location in mel.
                if mel_index < mel.len() {
                    // Make sure we don't go out of bounds.
                    mel[mel_index] += thread_output[mel_index];
                }
            }
        }
    }

    let mmax = mel
        .iter()
        .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Greater))
        .copied()
        .unwrap_or(ZERO)
        - 8.0;
    for m in mel.iter_mut() {
        let v = f32::max(*m, mmax);
        *m = v / FOUR + ONE
    }
    mel
}

/// Convert PCM samples to Mel spectrogram
pub fn pcm_to_mel(
    cfg: &super::Config,
    samples: &[f32],
    filters: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    log_mel_spectrogram(samples, filters, n_fft, hop_length, cfg.num_mel_bins, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft() {
        let input: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];
        let output = fft(&input);
        assert_eq!(
            output,
            vec![1.0, 0.0, -4.371139e-8, -1.0, -1.0, 0.0, 4.371139e-8, 1.0]
        );
    }

    #[test]
    fn test_dft() {
        let input = vec![0.0, 1.0, 0.0, 0.0];
        let output = dft(&input);
        assert_eq!(
            output,
            vec![
                1.0,
                0.0,
                -4.371139e-8,
                -1.0,
                -1.0,
                8.742278e-8,
                1.1924881e-8,
                1.0
            ]
        );
    }

    #[test]
    fn test_log_mel_spectrogram() {
        let samples = vec![0.0; 1000];
        let filters = vec![0.0; 1000];
        let output = log_mel_spectrogram(&samples, &filters, 100, 10, 10, false);
        assert_eq!(output.len(), 1000);
    }

    #[test]
    fn test_tiny_log_mel_spectrogram() {
        let samples = vec![0.0; 100];
        let filters = vec![0.0; 100];
        let output = log_mel_spectrogram(&samples, &filters, 20, 2, 2, false);
        assert_eq!(output.len(), 100);
    }
}
