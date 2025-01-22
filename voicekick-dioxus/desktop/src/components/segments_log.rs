use dioxus::prelude::*;

use crate::states::VoiceState;

const CSS: Asset = asset!("/assets/whisper.css");

#[component]
pub fn SegmentsLogComponent() -> Element {
    let voice_state = use_context::<VoiceState>();

    let segments_rendered = voice_state
        .segments
        .iter()
        .enumerate()
        .map(|(i, vc)| {
            let segment = &vc.segment;

            let duration_ms = (segment.duration * 1000.0) as i64;

            rsx! {
                div {
                    key: "{i}",
                    class: "segment",
                    div {
                        class: "voicepart",
                        p {
                            class: "text",
                            "{&segment.dr.text}"
                        }
                        div {
                            class: "metadata",
                            span {
                                class: "segment-metric",
                                "Duration: {duration_ms}ms"
                            }
                            span {
                                class: "segment-metric",
                                "No speech: {(1.0 - segment.dr.no_speech_prob):.2}"
                            }
                            span {
                                class: "segment-metric",
                                "AVG log prob: {(1.0 - segment.dr.avg_logprob):.2}"
                            }
                            span {
                                class: "segment-metric",
                                "Temperature: {segment.dr.temperature:.2}"
                            }
                        }
                    }
                    div {
                        class: "commandpart",
                        p {
                            class: "text",
                            "Output: {vc.command_text}"
                        }
                        div {
                            class: "command",
                            span {
                                class: "command-metric {vc.status}",
                                "Status: {vc.status}"
                            }
                            span {
                                class: "command-metric",
                                "Execution time: {vc.execution_time:.4}ms"
                            }
                        }
                    }
                }
            }
        })
        .collect::<Vec<Element>>();

    rsx! {
        document::Link { rel: "stylesheet", href: CSS}
        div {
            class: "whisper-container",
            div {
                class: "transcription-box",
                {segments_rendered.into_iter().rev()}
            }
        }
    }
}
