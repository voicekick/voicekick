use dioxus::prelude::*;

use voice_whisper::{WhichModel, SUPPORTED_LANGUAGES};

use crate::states::{VoiceState, WhisperConfigState};

#[component]
pub fn WhisperComponent() -> Element {
    let voice_state = use_context::<VoiceState>();
    let mut whisper_config_state = use_context::<WhisperConfigState>();

    let handle_model_change = move |evt: Event<FormData>| {
        whisper_config_state
            .current_model
            .set(evt.value().as_str().into());
    };

    let handle_language_change = move |evt: Event<FormData>| {
        whisper_config_state
            .current_language
            .set(evt.value().clone());
    };

    let models_rendered = WhichModel::iter().map(|model| {
        rsx! {
            option {
                value: "{model}",
                "{model}"
            }
        }
    });

    let languages_rendered = SUPPORTED_LANGUAGES.iter().map(|(code, lang)| {
        rsx! {
            option {
                value: "{code}",
                "{lang}"
            }
        }
    });

    rsx! {
        div {
            class: "input-model-selector",
            label {
                r#for: "model-select",
                "Whisper model "
            }
            select {
                id: "model-select",
                value: "{whisper_config_state.current_model.read()}",
                onchange: handle_model_change,
                disabled: *voice_state.is_recording.read(),
                {models_rendered}
            }
        }

        div {
            class: "input-language-selector",
            label {
                r#for: "language-select",
                "Whisper language "
            }
            select {
                disabled: !whisper_config_state.current_model.read().is_multilingual() || *voice_state.is_recording.read(),
                id: "language-select",
                value: "{whisper_config_state.current_language.read()}",
                onchange: handle_language_change,
                {languages_rendered}
            }
        }
    }
}
