use dioxus::prelude::*;

use crate::components::{SegmentsLogComponent, VoiceComponent, WaveformComponent};

#[component]
pub fn Home() -> Element {
    rsx! {
        VoiceComponent {}

        WaveformComponent {}

        SegmentsLogComponent {}

    }
}
