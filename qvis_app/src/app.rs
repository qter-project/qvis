use crate::{
    take_picture::{TAKE_PICTURE_CHANNEL, TakePictureMessage},
    video::Video,
};
use leptos::prelude::*;
use leptos_ws::ChannelSignal;
use log::{LevelFilter, Log, Metadata, Record, info};
use std::sync::atomic::{AtomicU32, Ordering};

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Cube Vision</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <link rel="shortcut icon" href="favicon.ico" type="image/x-icon" />
          <link rel="stylesheet" id="leptos" href="/pkg/qvis_app.css" />
          <AutoReload options=options.clone() />
          <HydrationScripts options />
        </head>
        <body class="text-white bg-black">
          <App />
        </body>
      </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    let (messages, set_messages) = signal(Vec::<(u32, String)>::new());
    let logger = Box::leak(Box::new(MessagesLogger {
        writer: set_messages,
        id: AtomicU32::default(),
    }));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(LevelFilter::Debug);
    }

    leptos_ws::provide_websocket();

    let take_picture_channel = ChannelSignal::new(TAKE_PICTURE_CHANNEL).unwrap();
    let take_picture_channel2 = take_picture_channel.clone();

    let messages_container = NodeRef::<leptos::html::Div>::new();
    let (overflowing, set_overflowing) = signal(true);
    let (take_picture_command, set_take_picture) = signal(());

    let take_picture = move |_| set_take_picture.set(());

    let take_picture_resp = Callback::new(move |resp| {
        take_picture_channel2
            .send_message(resp)
            .unwrap();
    });

    take_picture_channel
        .on_client(move |msg: &TakePictureMessage| {
            info!("Recieved message {msg:#?}");
            let TakePictureMessage::TakePicture = msg else {
                return;
            };
            set_take_picture.set(());
        })
        .unwrap();

    Effect::new(move |_| {
        messages.get();
        let Some(container) = messages_container.get() else {
            return;
        };
        let scroll_height = container.scroll_height();
        let client_height = container.client_height();
        set_overflowing.set(scroll_height > client_height);
        container.set_scroll_top(scroll_height);
    });

    view! {
      <header class="mb-5 font-sans text-4xl font-bold tracking-wider text-center bg-[rgb(47,48,80)] leading-20">
        "QVIS"
      </header>
      <main class="flex flex-col gap-4 justify-center mr-4 ml-4 text-center">
        <Video take_picture_resp take_picture_command />
        <button on:click=take_picture>HERE</button>
        "Messages:"
        <div class="relative h-72 font-mono text-left border-2 border-gray-300">
          <div
            class:hidden=move || !overflowing.get()
            class="absolute top-0 left-0 right-3 from-black to-transparent pointer-events-none h-15 bg-linear-to-b"
          />
          <div
            node_ref=messages_container
            class="overflow-y-auto h-full [&::-webkit-scrollbar]:w-3 [&::-webkit-scrollbar-thumb]:bg-white"
          >
            <ul class="pl-4 list-disc list-inside">
              <For each=move || messages.get() key=|msg| msg.0 let((_, msg))>
                <li>{msg}</li>
              </For>
            </ul>
          </div>
        </div>
      </main>
    }
}

struct MessagesLogger {
    writer: WriteSignal<Vec<(u32, String)>>,
    id: AtomicU32,
}

impl Log for MessagesLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        self.writer.update(|v| {
            v.push((
                self.id.fetch_add(1, Ordering::SeqCst),
                format!("[{}] {}", record.level(), record.args()),
            ));
        });
    }

    fn flush(&self) {}
}
