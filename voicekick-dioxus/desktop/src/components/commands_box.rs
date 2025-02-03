use dioxus::prelude::*;

use crate::{services::AVAILABLE_COMMANDS, states::CommandsBoxState};

const CSS: Asset = asset!("/assets/commands_box.css");

#[component]
pub fn CommandsBoxComponent() -> Element {
    let mut command_box_state = use_context::<CommandsBoxState>();

    let commands_rendered = (AVAILABLE_COMMANDS.read()).clone().into_iter().map(|(namespace, cmds)| {
        let is_selected = if command_box_state
            .selected_namespace
            .read()
            .as_ref()
            .map(|s| s == namespace.as_str())
            .unwrap_or(false)
        {
            "selected"
        } else {
            ""
        };

        let render_cmds = cmds.iter().map(|cmd| {
            rsx! { div { class: "command-item", "{cmd}" } }
        });

        rsx! {
            div {
                class: "namespace-container {is_selected}",
                onclick: move |_| command_box_state.selected_namespace.set(Some(namespace.clone())),
                div {
                    class: "namespace-header",
                    "{namespace}"
                }
                div { class: "namespace-content",
                    {render_cmds}
                }
            }
        }
    });

    rsx! {
    document::Link { rel: "stylesheet", href: CSS }
     div {
        class: "commands-outer-container",
            p {
                "Available commands"
            }

            div {
                class: "commands-grid",
                {commands_rendered}
            }
        }
    }
}
