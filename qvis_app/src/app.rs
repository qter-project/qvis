#![allow(clippy::similar_names, clippy::unused_async)]

use crate::{
    messages_logger::MessagesLogger,
    video::{OnceBarrier, Video, pixel_assignment_command, take_picture_command},
};
use leptos::{html, prelude::*, task::spawn_local};
use leptos_use::{
    ConstraintExactIdeal, FacingMode, UseUserMediaOptions, UseUserMediaReturn,
    VideoTrackConstraints, use_user_media_with_options,
};
use leptos_ws::ChannelSignal;
use log::{LevelFilter, info, warn};
use puzzle_theory::{permutations::Permutation, puzzle_geometry::parsing::puzzle};
use qvis::{CVProcessor, Pixel};
use serde::{Deserialize, Serialize};
use server_fn::codec::{MultipartData, MultipartFormData};
use std::sync::Arc;

pub const TAKE_PICTURE_CHANNEL: &str = "take_picture_channel";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TakePictureMessage {
    // Request
    TakePicture,
    Calibrate(Permutation),
    // Response
    PermutationResult(Permutation, f64),
    Calibrated,
}

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
    let logger = Box::leak(Box::new(MessagesLogger::new(set_messages)));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(LevelFilter::Debug);
    }

    leptos_ws::provide_websocket();
    #[cfg(feature = "hydrate")]
    {
        let context = expect_context::<leptos_ws::ServerSignalWebSocket>();
        context.set_on_connect(move || {
            info!("Established connection with server");
            spawn_local(async move {
                print_ready().await.unwrap();
            });
        });
        context.set_on_disconnect(move || {
            warn!("Lost connection with server; trying reconnect");
        });
        context.set_on_reconnect(move || {
            info!("Re-established connection with server");
            spawn_local(async move {
                print_ready().await.unwrap();
            });
        });
    }

    let use_user_media_return = use_user_media_with_options(UseUserMediaOptions::default().video(
        VideoTrackConstraints::default().facing_mode(ConstraintExactIdeal::ExactIdeal {
            exact: None,
            ideal: Some(FacingMode::Environment),
        }),
    ));

    let messages_container = NodeRef::<leptos::html::Div>::new();
    let video_ref = NodeRef::<html::Video>::new();
    let canvas_ref = NodeRef::<html::Canvas>::new();
    let cv_overlay_ref: NodeRef<html::Canvas> = NodeRef::new();
    let (overflowing, set_overflowing) = signal(true);
    let playing_barrier = OnceBarrier::new();
    let cube3 = puzzle("3x3");
    let (cv_available_tx, cv_available_rx) = tokio::sync::watch::channel(None::<CVProcessor>);

    let take_picture_channel = ChannelSignal::new(TAKE_PICTURE_CHANNEL).unwrap();

    let pixel_assignment_action =
        Action::new_local(|data: &web_sys::FormData| pixel_assignment(data.clone().into()));

    let do_pixel_assignment = {
        let playing_barrier = Arc::clone(&playing_barrier);
        let UseUserMediaReturn {
            enabled: video_enabled,
            set_enabled: set_video_enabled,
            ..
        } = use_user_media_return;
        move || {
            let video_ref = video_ref.get_untracked().unwrap();
            let canvas_ref = canvas_ref.get_untracked().unwrap();
            let playing_barrier = Arc::clone(&playing_barrier);
            spawn_local(async move {
                let blob = pixel_assignment_command(
                    &video_ref,
                    &canvas_ref,
                    video_enabled,
                    set_video_enabled,
                    &playing_barrier,
                )
                .await;
                let form_data = web_sys::FormData::new().unwrap();
                form_data.append_with_blob("qvis_picture", &blob).unwrap();
                pixel_assignment_action.dispatch_local(form_data);
            });
        }
    };
    {
        let cv_available_tx = cv_available_tx.clone();
        let cv_available_rx = cv_available_rx.clone();
        let playing_barrier = Arc::clone(&playing_barrier);
        let do_pixel_assignment = do_pixel_assignment.clone();
        take_picture_channel
            .clone()
            .on_client(move |msg: &TakePictureMessage| {
                let video_ref = video_ref.get_untracked().unwrap();
                let canvas_ref = canvas_ref.get_untracked().unwrap();
                info!("Received message {msg:?}");

                let take_picture_channel = take_picture_channel.clone();
                let mut cv_available_rx = cv_available_rx.clone();
                let cv_available_tx = cv_available_tx.clone();
                let UseUserMediaReturn {
                    enabled: video_enabled,
                    set_enabled: set_video_enabled,
                    ..
                } = use_user_media_return;
                match msg {
                    TakePictureMessage::TakePicture => {
                        let playing_barrier = Arc::clone(&playing_barrier);
                        let do_pixel_assignment = do_pixel_assignment.clone();
                        spawn_local(async move {
                            let pixels = take_picture_command(
                                &video_ref,
                                &canvas_ref,
                                video_enabled,
                                set_video_enabled,
                                &playing_barrier,
                            )
                            .await;
                            if cv_available_rx.borrow_and_update().is_none() {
                                do_pixel_assignment();
                                cv_available_rx.changed().await.unwrap();
                            }
                            let cv_processor = cv_available_rx.borrow_and_update();
                            let cv_processor = cv_processor.as_ref().unwrap();
                            let (permutation, confidence) = cv_processor.process_image(&pixels);
                            info!(
                                "Processed {permutation} with confidence {:.2}",
                                confidence * 100.
                            );
                            take_picture_channel
                                .send_message(TakePictureMessage::PermutationResult(
                                    permutation,
                                    confidence,
                                ))
                                .unwrap();
                        });
                    }
                    TakePictureMessage::Calibrate(permutation) => {
                        let permutation = permutation.clone();
                        let playing_barrier = Arc::clone(&playing_barrier);
                        let do_pixel_assignment = do_pixel_assignment.clone();
                        spawn_local(async move {
                            let pixels = take_picture_command(
                                &video_ref,
                                &canvas_ref,
                                video_enabled,
                                set_video_enabled,
                                &playing_barrier,
                            )
                            .await;
                            if cv_available_rx.borrow().is_none() {
                                do_pixel_assignment();
                                cv_available_rx.changed().await.unwrap();
                            }
                            cv_available_tx.send_modify(|maybe_cv_processor| {
                                let cv_processor = maybe_cv_processor.as_mut().unwrap();
                                cv_processor.calibrate(&pixels, &permutation);
                            });
                            take_picture_channel
                                .send_message(TakePictureMessage::Calibrated)
                                .unwrap();
                        });
                    }
                    m @ (TakePictureMessage::PermutationResult(_, _)
                    | TakePictureMessage::Calibrated) => {
                        warn!("Received {m:?} on client, which should not happen");
                    }
                }
            });
    }

    {
        let cv_available_tx = cv_available_tx.clone();
        let cube3 = Arc::clone(&cube3);
        Effect::new(move |_| {
            let pixel_assignment = pixel_assignment_action.value().get();
            let Some(pixel_assignment) = pixel_assignment else {
                return;
            };
            let Some(pixel_assignment) = pixel_assignment
                .inspect_err(|err| warn!("Pixel assignment failed: {err}"))
                .ok()
            else {
                return;
            };

            let cv_processor =
                CVProcessor::new(Arc::clone(&cube3), pixel_assignment.len(), pixel_assignment);

            cv_available_tx.send_modify(|maybe_cv_processor| {
                *maybe_cv_processor = Some(cv_processor);
            });
        });
    }

    Effect::watch(
        move || messages.get(),
        move |_, _, _| {
            let Some(container) = messages_container.get_untracked() else {
                return;
            };
            let scroll_height = container.scroll_height();
            let client_height = container.client_height();
            set_overflowing.set(scroll_height > client_height);
            container.set_scroll_top(scroll_height);
        },
        false,
    );

    let cv_available_rx2 = cv_available_rx.clone();
    let do_export_cv_processor = move |_| {
        let export_file_name = match web_sys::window().unwrap().prompt_with_message_and_default(
            "Enter file name for CVProcessor export",
            "cv_processor_export.json",
        ) {
            Ok(Some(export_file_name)) if !export_file_name.trim().is_empty() => export_file_name,
            Ok(Some(_)) => {
                warn!("Export cancelled: file name is empty");
                return;
            }
            Ok(None) => {
                warn!("Export cancelled: user cancelled dialog");
                return;
            }
            Err(err) => {
                warn!("Export cancelled: prompt failed: {err:?}");
                return;
            }
        };
        let cv_available_rx2 = cv_available_rx2.borrow();
        let Some(cv_processor2) = cv_available_rx2.as_ref().cloned() else {
            warn!("Export failed: CVProcessor not yet available");
            return;
        };

        let cv_processor2 = leptos::serde_json::to_string(&cv_processor2).unwrap();
        spawn_local(async move {
            if let Err(err) = export_cv_processor(cv_processor2, export_file_name.clone()).await {
                warn!("Failed to export CVProcessor: {err}");
            } else {
                info!("Successfully exported CVProcessor to {export_file_name}");
            }
        });
    };

    let do_import_cv_processor = move |_| {
        let export_file_name = match web_sys::window().unwrap().prompt_with_message_and_default(
            "Enter file name for CVProcessor import",
            "cv_processor_export.json",
        ) {
            Ok(Some(export_file_name)) if !export_file_name.trim().is_empty() => export_file_name,
            Ok(Some(_)) => {
                warn!("Import cancelled: file name is empty");
                return;
            }
            Ok(None) => {
                warn!("Import cancelled: user cancelled dialog");
                return;
            }
            Err(err) => {
                warn!("Import cancelled: prompt failed: {err:?}");
                return;
            }
        };

        let cv_available_tx = cv_available_tx.clone();
        spawn_local(async move {
            match import_cv_processor(export_file_name.clone()).await {
                Ok(cv_processor) => {
                    cv_available_tx.send_modify(|maybe_cv_processor| {
                        *maybe_cv_processor = Some(cv_processor);
                    });
                    info!("Successfully imported CVProcessor from {export_file_name}");
                }
                Err(err) => {
                    warn!("Failed to import CVProcessor: {err}");
                }
            }
        });
    };

    view! {
      <header class="font-sans text-4xl font-bold tracking-wider text-center bg-[rgb(47,48,80)] leading-20">
        <button
          on:click=move |_| {
            location().reload().unwrap();
          }
          class="cursor-pointer"
        >
          "QVIS"
        </button>
      </header>
      <main class="flex flex-col gap-4 justify-center mt-5 mr-4 mb-6 ml-4 text-center">
        <Video video_ref canvas_ref cv_overlay_ref use_user_media_return playing_barrier cv_available_rx />
        // zoom
        // resolution (width)
        // camera device
        <div class="flex h-12">
          <button on:click=move |_| do_pixel_assignment() class="flex-1 border-2 border-white cursor-pointer">
            {move || {
              if pixel_assignment_action.pending().get() {
                "Processing...".to_string()
              } else {
                "Pixel assignment".to_string()
              }
            }}
          </button>
          <button class="flex-1 border-2 border-white cursor-pointer" on:click=do_export_cv_processor>
            "Export CVProcessor"
          </button>
          <button class="flex-1 border-2 border-white cursor-pointer" on:click=do_import_cv_processor>
            "Import CVProcessor"
          </button>
        </div>
        "Messages:"
        <div class="relative h-72 font-mono text-left border-2 border-gray-300">
          <div
            class:hidden=move || !overflowing.get()
            class="absolute top-0 left-0 right-3 h-5 from-black to-transparent pointer-events-none bg-linear-to-b"
          />
          <div
            node_ref=messages_container
            class="overflow-y-auto h-full [&::-webkit-scrollbar]:w-3 [&::-webkit-scrollbar-thumb]:bg-white"
          >
            <ul class="pl-4 list-disc list-inside whitespace-pre-wrap">
              <For each=move || messages.get() key=|msg| msg.0 let((_, msg))>
                <li>{msg}</li>
              </For>
            </ul>
          </div>
        </div>
      </main>
    }
}

#[server]
async fn print_ready() -> Result<(), ServerFnError> {
    leptos::logging::log!("READY");
    Ok(())
}

#[server]
async fn export_cv_processor(
    cv_processor: String,
    export_file_name: String,
) -> Result<(), ServerFnError> {
    let cv_processor: CVProcessor = leptos::serde_json::from_str(&cv_processor)?;
    let export_path = std::env::current_dir().unwrap().join(&export_file_name);
    let export_file = std::fs::File::create(export_path)?;
    leptos::serde_json::to_writer(export_file, &cv_processor)?;
    leptos::logging::log!("Exported CVProcessor to {export_file_name}");
    Ok(())
}

#[server]
async fn import_cv_processor(import_file_name: String) -> Result<CVProcessor, ServerFnError> {
    let import_path = std::env::current_dir().unwrap().join(&import_file_name);
    let import_file = std::fs::File::open(import_path)?;
    let cv_processor = leptos::serde_json::from_reader(import_file)?;
    leptos::logging::log!("Imported CVProcessor from {import_file_name}");
    Ok(cv_processor)
}

#[server(
    input = MultipartFormData,
)]
async fn pixel_assignment(data: MultipartData) -> Result<Box<[Pixel]>, ServerFnError> {
    let mut data = data.into_inner().unwrap();
    let field = data
        .next_field()
        .await
        .map_err(ServerFnError::new)?
        .unwrap();
    let bytes = field.bytes().await?;

    let pixel_assignment_ui_tx = use_context::<
        std::sync::mpsc::Sender<(tokio::sync::oneshot::Sender<Box<[Pixel]>>, bytes::Bytes)>,
    >()
    .unwrap();
    let (pixel_assignment_done_tx, pixel_assignment_done_rx) = tokio::sync::oneshot::channel();
    pixel_assignment_ui_tx
        .send((pixel_assignment_done_tx, bytes))
        .unwrap();
    let pixel_assignment = pixel_assignment_done_rx.await.unwrap();

    Ok(pixel_assignment)
}
