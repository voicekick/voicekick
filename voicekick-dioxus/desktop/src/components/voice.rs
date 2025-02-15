use dioxus::prelude::*;

use voice_stream::input_devices;

use crate::services::VoiceKickCommand;
use crate::states::{VoiceConfigState, VoiceState};

#[component]
pub fn VoiceComponent() -> Element {
    let mut voice_state = use_context::<VoiceState>();
    let mut voice_config_state = use_context::<VoiceConfigState>();
    let voice_command_task = use_coroutine_handle::<VoiceKickCommand>();
    let devices = use_signal(|| input_devices().unwrap_or_default());

    let handle_device_change = move |evt: Event<FormData>| {
        let new_value = evt.value();
        if *voice_config_state.selected_input_device.read() != new_value {
            voice_config_state.selected_input_device.set(new_value);
            voice_command_task.send(VoiceKickCommand::UpdateVoiceStream);
        }
    };

    let options_rendered = devices.iter().map(|device| {
        rsx! {
            option {
                value: "{device}",
                selected: *device == voice_config_state.selected_input_device.read().as_str(),
                "{device}"
            }
        }
    });

    let handle_threshold_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f32>() {
            if *voice_config_state.silero_voice_threshold.read() != value {
                voice_config_state.silero_voice_threshold.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateVoiceStream);
            }
        }
    };

    let toggle_recording = move |_| {
        if *voice_state.is_recording.read() {
            voice_state.is_recording.set(false);
            voice_command_task.send(VoiceKickCommand::Pause);
        } else {
            voice_state.is_recording.set(true);
            voice_command_task.send(VoiceKickCommand::Record);
        }
    };

    rsx! {
        div {
            class: "input-device-selector",
            label {
                r#for: "device-select",
                "Input device "
            }
            select {
                disabled: *voice_state.is_recording.read(),
                id: "device-select",
                value: "{voice_config_state.selected_input_device}",
                onchange: handle_device_change,
                {options_rendered}
            }

            button {
                disabled: voice_config_state.selected_input_device.read().is_empty(),
                onclick: toggle_recording,
                div {
                    class: "record-button-content",
                    // Recording indicator dot
                    div {
                        class: "record-indicator"
                    }
                    // Button text
                    span {
                        if *voice_state.is_recording.read() { "⏸︎" } else { "▶" }
                    }
                }
            }

        }
        div {
                class: "threshold-selector",
                label {
                    r#for: "threshold-range",
                    "Voice detection threshold: {voice_config_state.silero_voice_threshold:.2}"
                }
                input {
                    disabled: *voice_state.is_recording.read(),
                    r#type: "range",
                    id: "threshold-range",
                    min: "0",
                    max: "1",
                    step: "0.01",
                    value: "{voice_config_state.silero_voice_threshold}",
                    onchange: handle_threshold_change
                }
        }
    }
}
