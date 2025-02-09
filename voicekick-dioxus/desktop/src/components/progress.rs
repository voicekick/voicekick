use dioxus::prelude::*;
use voice_whisper::Progress;

/// Custom download progress.
#[derive(Default, Clone, Debug)]
pub struct DownloadProgress {
    pub current: usize,
    pub total: usize,
}

#[derive(Default, Clone, Debug)]
pub struct GlobalProgress;

impl Progress for GlobalProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        // let mut writer = CURRENT_PROGRESS.write();
        // writer.current = 10;
        // writer.total = size + 100;
        // // println!("Writer {:?}", *writer);
    }

    fn update(&mut self, size: usize) {
        // let mut writer = CURRENT_PROGRESS.write();

        // writer.current += size;
        // // println!("Writer {:?}", *writer);
    }

    fn finish(&mut self) {
        println!("Done !");
    }
}

pub static CURRENT_PROGRESS: GlobalSignal<DownloadProgress> =
    Signal::global(DownloadProgress::default);

#[component]
pub fn ProgressComponent() -> Element {
    rsx! {
        div {
            class: "progress-container",
            div {
                class: "progress-bar",
                style: "width: {CURRENT_PROGRESS.read().current.to_string()}%",
                {"10"}
            }
            div {
                class: "progress-text",
                {"101"}
            }
        }
    }
}
