use crate::{
    take_picture::{TAKE_PICTURE_CHANNEL, TakePictureMessage},
    video::Video,
};
use leptos::{logging::log, prelude::*};
use leptos_ws::ChannelSignal;
use puzzle_theory::puzzle_geometry::parsing::puzzle;
use qvis::CVProcessor;

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Cube Vision</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <link rel="stylesheet" id="leptos" href="/pkg/qvis-app.css" />
          <AutoReload options=options.clone() />
          <HydrationScripts options />
        </head>
        <body>
          <App />
        </body>
      </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    leptos_ws::provide_websocket();

    let take_picture_channel = ChannelSignal::new(TAKE_PICTURE_CHANNEL).unwrap();

    let puzzle_geometry = puzzle("3x3").into_inner();
    let cv = CVProcessor::new(puzzle_geometry, 0);

    take_picture_channel
        .clone()
        .on_client(move |msg: &TakePictureMessage| {
            log!("Recieved message {msg:#?}");
            let TakePictureMessage::TakePicture = msg else {
                return;
            };
            let picture = cv.process_image(Box::new([])).0;
            take_picture_channel
                .send_message(TakePictureMessage::PictureResult(Ok(picture)))
                .unwrap();
        })
        .unwrap();

    view! {
      // <div class="p-2 m-2 rounded border border-gray-300">
      // <h2>"Echo Channel Signal Example"</h2>
      // <input
      // type="text"
      // prop:value=move || echo_input.get()
      // on:input=move |ev| {
      // set_echo_input.set(event_target_value(&ev));
      // }
      // placeholder="Type something to echo..."
      // />
      // <button on:click=move |_| {
      // let msg = echo_input.get();
      // if !msg.trim().is_empty() {
      // take_picture_channel.send_message(msg).ok();
      // set_echo_input.set(String::new());
      // }
      // }>"Send Echo"</button>
      // </div>
      <Video />
    }
}
