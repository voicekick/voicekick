use dioxus::prelude::*;

use crate::components::{
    CommandsBoxComponent, SegmentsLogComponent, VoiceComponent, WaveformComponent,
};

#[component]
pub fn Home() -> Element {
    rsx! {
        VoiceComponent {}

        WaveformComponent {}

        CommandsBoxComponent {}

        SegmentsLogComponent {}
    }
}
