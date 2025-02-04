use dioxus::prelude::*;

use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

use crate::{
    services::VoiceKickCommand,
    states::{VoiceConfigState, VoiceState},
};

#[component]
pub fn WhisperComponent() -> Element {
    let voice_state = use_context::<VoiceState>();
    let mut voice_config_state = use_context::<VoiceConfigState>();
    let voice_command_task = use_coroutine_handle::<VoiceKickCommand>();

    let handle_model_change = move |evt: Event<FormData>| {
        let model: WhichModel = evt.value().as_str().into();

        if *voice_config_state.current_model.read() != model {
            voice_config_state.current_model.set(model);
            voice_command_task.send(VoiceKickCommand::UpdateASR);
        }
    };

    let handle_language_change = move |evt: Event<FormData>| {
        let language: String = evt.value();

        if *voice_config_state.current_language.read() != language {
            voice_config_state.current_language.set(language.clone());
            voice_command_task.send(VoiceKickCommand::UpdateASR);
        }
    };

    let handle_temperature_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f64>() {
            if *voice_config_state.temperature.read() != value {
                voice_config_state.temperature.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_repetition_penalty_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f32>() {
            if *voice_config_state.repetition_penalty.read() != value {
                voice_config_state.repetition_penalty.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_repetition_frequency_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<usize>() {
            if *voice_config_state.repetition_frequency.read() != value {
                voice_config_state.repetition_frequency.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_boost_value_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f32>() {
            if *voice_config_state.boost_value.read() != value {
                voice_config_state.boost_value.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_command_boost_value_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f32>() {
            if *voice_config_state.command_boost_value.read() != value {
                voice_config_state.command_boost_value.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_no_speech_threshold_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f64>() {
            if *voice_config_state.no_speech_threshold.read() != value {
                voice_config_state.no_speech_threshold.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let handle_logprob_threshold_change = move |evt: Event<FormData>| {
        if let Ok(value) = evt.value().parse::<f64>() {
            if *voice_config_state.logprob_threshold.read() != value {
                voice_config_state.logprob_threshold.set(value);
                voice_command_task.send(VoiceKickCommand::UpdateASR);
            }
        }
    };

    let current_model = *voice_config_state.current_model.read();

    let models_rendered = WhichModel::iter().map(|model| {
        let current_model_string = current_model.to_string();
        rsx! {
            option {
                value: "{model}",
                selected: *model == current_model_string,
                "{model}"
            }
        }
    });

    let languages_rendered = SUPPORTED_LANGUAGES.iter().map(|(code, lang)| {
        rsx! {
            option {
                value: "{code}",
                selected: *code == voice_config_state.current_language.read().as_str(),
                "{lang}"
            }
        }
    });

    let is_disabled = *voice_state.is_recording.read();
    rsx! {
        div {
            class: "input-model-selector",
            label {
                r#for: "model-select",
                "Whisper model "
            }
            select {
                disabled: is_disabled,
                id: "model-select",
                value: "{current_model}",
                onchange: handle_model_change,
                {models_rendered}
            }
        }

        div {
            class: "input-language-selector",
            label {
                r#for: "language-select",
                "Language "
            }
            select {
                disabled: !current_model.is_multilingual() || is_disabled,
                id: "language-select",
                value: "{voice_config_state.current_language.read()}",
                onchange: handle_language_change,
                {languages_rendered}
            }
        }

        div {
            label {
                r#for: "temperature",
                "Temperature ({voice_config_state.temperature.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "temperature",
                min: "0.0",
                max: "1.0",
                step: "0.1",
                value: "{voice_config_state.temperature.read()}",
                onchange: handle_temperature_change,
            }
        }

        div {
            label {
                r#for: "repetition_penalty",
                "Repetition Penalty ({voice_config_state.repetition_penalty.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "repetition_penalty",
                min: "0.0",
                max: "2.0",
                step: "0.1",
                value: "{voice_config_state.repetition_penalty.read()}",
                onchange: handle_repetition_penalty_change,
            }
        }

        div {
            label {
                r#for: "repetition_frequency",
                "Repetition Frequency ({voice_config_state.repetition_frequency.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "number",
                id: "repetition_frequency",
                min: "0",
                max: "100",
                value: "{voice_config_state.repetition_frequency.read()}",
                onchange: handle_repetition_frequency_change,
            }
        }

        div {
            label {
                r#for: "boost_value",
                "Boost Value ({voice_config_state.boost_value.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "boost_value",
                min: "0.0",
                max: "5.0",
                step: "0.1",
                value: "{voice_config_state.boost_value.read()}",
                onchange: handle_boost_value_change,
            }
        }

        div {
            label {
                r#for: "command_boost_value",
                "Command Boost Value ({voice_config_state.command_boost_value.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "command_boost_value",
                min: "0.0",
                max: "5.0",
                step: "0.1",
                value: "{voice_config_state.command_boost_value.read()}",
                onchange: handle_command_boost_value_change,
            }
        }

        div {
            label {
                r#for: "no_speech_threshold",
                "No Speech Threshold ({voice_config_state.no_speech_threshold.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "no_speech_threshold",
                min: "0.0",
                max: "1.0",
                step: "0.01",
                value: "{voice_config_state.no_speech_threshold.read()}",
                onchange: handle_no_speech_threshold_change,
            }
        }

        div {
            label {
                r#for: "logprob_threshold",
                "Logprob Threshold ({voice_config_state.logprob_threshold.read()})"
            }
            input {
                disabled: is_disabled,
                r#type: "range",
                id: "logprob_threshold",
                min: "-5.0",
                max: "0.0",
                step: "0.1",
                value: "{voice_config_state.logprob_threshold.read()}",
                onchange: handle_logprob_threshold_change,
            }
        }
    }
}
