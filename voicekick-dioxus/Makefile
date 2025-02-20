.PHONY: dev
dev:
	dx serve --platform desktop --package desktop --always-on-top false

.PHONY: build
build:
	dx build --platform desktop --package desktop --release
	cp Info.plist ../voicekick-dioxus/target/dx/desktop/release/macos/Desktop.app/Contents/Info.plist
	rm -rf ../voicekick-dioxus/target/dx/desktop/release/macos/Voicekick.app
	mv ../voicekick-dioxus/target/dx/desktop/release/macos/Desktop.app ../voicekick-dioxus/target/dx/desktop/release/macos/Voicekick.app

.PHONE: setup
setup:
	cargo install dioxus-cli
