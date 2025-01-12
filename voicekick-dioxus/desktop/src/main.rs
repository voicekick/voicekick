use dioxus::prelude::*;
use tracing::Level;

mod components;
mod services;
mod states;
mod views;

use states::{VoiceState, WhisperConfigState};
use ui::Navbar;
use views::{Home, Whisper};

#[derive(Debug, Clone, Routable, PartialEq)]
enum Route {
    #[layout(DesktopNavbar)]
    #[route("/")]
    Home {},
    #[route("/whisper")]
    Whisper {},
}

const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus_logger::init(Level::INFO).expect("failed to init logger");
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Build cool things ✌️
    use_coroutine(services::voicekick_service);
    use_context_provider(|| VoiceState::default());
    use_context_provider(|| WhisperConfigState::default());

    rsx! {
        // Global app resources
        document::Link { rel: "stylesheet", href: MAIN_CSS }

        Router::<Route> {}
    }
}

/// A desktop-specific Router around the shared `Navbar` component
/// which allows us to use the desktop-specific `Route` enum.
#[component]
fn DesktopNavbar() -> Element {
    rsx! {
        Navbar {
            Link {
                to: Route::Home {},
                "Voice"
            }
            Link {
                to: Route::Whisper {  },
                "Whisper"
            }
        }

        Outlet::<Route> {}
    }
}
