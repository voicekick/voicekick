use dioxus::prelude::*;

use crate::components::{VoiceComponent, WaveformComponent};

#[component]
pub fn Home() -> Element {
    rsx! {
        VoiceComponent {}

        WaveformComponent {}
    }
}
