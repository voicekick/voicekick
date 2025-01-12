use dioxus::prelude::*;

use crate::states::VoiceState;

const WAVEFORM_CSS: Asset = asset!("/assets/waveform.css");
const WHISPER_CSS: Asset = asset!("/assets/whisper.css");
const VISUALIZATION_WIDTH: usize = 100;

#[component]
pub fn WaveformComponent() -> Element {
    let voice_state = use_context::<VoiceState>();

    // Downsample the audio data to visualization width
    let downsampled: Vec<f32> = if voice_state.raw_samples.is_empty() {
        vec![0.0; VISUALIZATION_WIDTH]
    } else {
        // First, flatten all sample vectors into one
        let flat_samples: Vec<f32> = voice_state
            .raw_samples
            .read()
            .iter()
            .flatten()
            .copied()
            .collect();

        if !flat_samples.is_empty() {
            let samples_per_point = flat_samples.len() / VISUALIZATION_WIDTH;
            (0..VISUALIZATION_WIDTH)
                .map(|i| {
                    let start = i * samples_per_point;
                    let end = (start + samples_per_point).min(flat_samples.len());
                    if start < flat_samples.len() {
                        // Take max absolute value in this chunk to preserve peaks
                        flat_samples[start..end]
                            .iter()
                            .map(|x| x.abs())
                            .fold(0f32, |a, b| a.max(b))
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            vec![0.0; VISUALIZATION_WIDTH]
        }
    };

    let samples_rendered = downsampled.iter().enumerate().map(|(i, value)| {
        let height = value * 1000.0;

        rsx! {
            div {
                key: "{i}",
                class: if *value >= 0.0 { "sample-bar positive" } else { "sample-bar negative" },
                style: "left: {i}%; width: 1%; height: {height}%;",
            }
        }
    });

    let segments_rendered = voice_state
        .segments
        .iter()
        .enumerate()
        .map(|(i, segment)| {
            let duration_ms = (segment.duration * 1000.0) as i64;

            rsx! {
                div {
                    key: "{i}",
                    class: "segment",
                    p {
                        class: "text",
                        "{&segment.dr.text}"
                    }
                    div {
                        class: "metadata",
                        span {
                            class: "duration segment-metric",
                            "Duration: {duration_ms}ms"
                        }
                        span {
                            class: "nospeech segment-metric",
                            "No speech: {(1.0 - segment.dr.no_speech_prob):.2}"
                        }
                        span {
                            class: "avglogprob segment-metric",
                            "AVG log prob: {(1.0 - segment.dr.avg_logprob):.2}"
                        }
                        span {
                            class: "temperature segment-metric",
                            "Temperature: {segment.dr.temperature}"
                        }
                    }
                }
            }
        })
        .collect::<Vec<Element>>();

    rsx! {
        document::Link { rel: "stylesheet", href: WAVEFORM_CSS}
        document::Link { rel: "stylesheet", href: WHISPER_CSS}
        div {
            class: "waveform-container",

            div {
                class: "waveform-visualization",
                // Center line (zero line)
                div {
                    class: "center-line",
                }

                // Draw bars for each sample
                {samples_rendered}
            }
        }
        div {
            class: "whisper-container",
            div {
                class: "transcription-box",
                {segments_rendered.into_iter().rev()}
            }
        }
    }
}
