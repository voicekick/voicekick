use dioxus::prelude::*;

#[component]
pub fn Navbar(children: Element) -> Element {
    rsx! {
        div {
            id: "navbar",
            {children}
        }
    }
}
