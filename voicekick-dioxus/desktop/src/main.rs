use dioxus::prelude::*;
use tracing::Level;

mod commands;
mod components;
mod services;
mod states;
mod views;

use states::{CommandsBoxState, VoiceConfigState, VoiceState};
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

const CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus_logger::init(Level::INFO).expect("failed to init logger");

    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    tokio::spawn(server::serve());

    // Build cool things ✌️
    use_coroutine(services::voicekick_service);
    use_coroutine(services::segments_service);
    use_context_provider(|| VoiceState::default());
    use_context_provider(|| VoiceConfigState::default());
    use_context_provider(|| commands::all());
    use_context_provider(|| CommandsBoxState::default());

    rsx! {
        // Global app resources
        document::Link { rel: "stylesheet", href: CSS }

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
