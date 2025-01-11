use dioxus::prelude::*;

use crate::components::WhisperComponent;

#[component]
pub fn Whisper() -> Element {
    rsx! {
        WhisperComponent {}
    }
}
