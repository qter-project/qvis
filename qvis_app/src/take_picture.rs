use leptos::{
    prelude::*,
    server_fn::codec::{GetUrl, Json},
};
use log::warn;
use puzzle_theory::{permutations::Permutation, puzzle_geometry::parsing::puzzle};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ssr")]
mod ssr_imports {
    pub use crate::calibration_ui;
    pub use leptos::logging::log;
    pub use leptos_ws::ChannelSignal;
    pub use std::sync::Mutex;
}

pub const TAKE_PICTURE_CHANNEL: &str = "take_picture_channel";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TakePictureMessage {
    // Request
    TakePicture,
    // Response
    PermutationResult(Permutation),
    NeedsCalibration,
}

#[server(
  endpoint = "take_picture",
  input = GetUrl,
  output = Json
)]
pub async fn take_picture() -> Result<Permutation, ServerFnError> {
    use ssr_imports::*;

    let channel = ChannelSignal::new(TAKE_PICTURE_CHANNEL).map_err(ServerFnError::new)?;

    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Mutex::new(Some(tx));

    channel
        .on_server(move |message: &TakePictureMessage| {
            log!("Recieved message {message:#?}");
            match message {
                TakePictureMessage::PermutationResult(permutation) => {
                    let sender = tx.lock().unwrap().take().expect("No sender available");
                    sender.send(permutation.clone()).unwrap();
                }
                TakePictureMessage::NeedsCalibration => {
                    let puzzle_geometry = puzzle("3x3").into_inner();
                    calibration_ui::calibration_ui(puzzle_geometry).unwrap();
                }
                TakePictureMessage::TakePicture => {
                    warn!("Received TakePictureMessage::TakePicture on server, which should not happen");
                }
            }
        })
        .map_err(ServerFnError::new)?;

    channel
        .send_message(TakePictureMessage::TakePicture)
        .map_err(ServerFnError::new)?;

    rx.await.map_err(ServerFnError::new)
}
