use dioxus::prelude::*;

const CSS: Asset = asset!("/assets/styling/navbar.css");

#[component]
pub fn Navbar(children: Element) -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: CSS }

        div {
            id: "navbar",
            {children}
        }
    }
}
