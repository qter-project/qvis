#![warn(clippy::pedantic)]
#![allow(clippy::must_use_candidate, clippy::wildcard_imports, clippy::too_many_lines)]

pub mod app;
pub mod take_picture;
pub mod video;
#[cfg(feature = "ssr")]
pub mod calibration_ui;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(crate::app::App);
}
