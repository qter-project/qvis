use leptos::{
    logging::{error, log},
    prelude::*,
};
use leptos_use::{
    FacingMode, UseUserMediaOptions, UseUserMediaReturn, VideoTrackConstraints,
    use_user_media_with_options,
};

#[component]
pub fn Video() -> impl IntoView {
    let video_ref = NodeRef::<leptos::html::Video>::new();
    let UseUserMediaReturn {
        stream,
        enabled,
        set_enabled,
        ..
    } = use_user_media_with_options(
        UseUserMediaOptions::default()
            .video(VideoTrackConstraints::default().facing_mode(FacingMode::Environment)),
    );

    Effect::new(move |_| {
        match stream.get() {
            Some(Ok(s)) => {
                video_ref.with(|v| {
                    if let Some(v) = v {
                        v.set_src_object(Some(&s));
                    }
                });
                return;
            }
            Some(Err(e)) => error!("Failed to get media stream: {:?}", e),
            None => log!("No stream yet"),
        }

        video_ref.with(|v| {
            if let Some(v) = v {
                v.set_src_object(None);
            }
        });
    });

    view! {
      <div class="flex flex-col gap-4 text-center">
        <div>
          <video node_ref=video_ref controls=false autoplay=true muted=true class="w-auto h-96"></video>
        </div>
        <button on:click=move |_| {
          set_enabled.set(!enabled.get())
        }>{move || if enabled.get() { "Stop Video" } else { "Start Video" }}</button>
        <button on:click=move |_| {
          set_enabled.set(!enabled.get())
        }>{move || if enabled.get() { "Stop" } else { "Start" }}</button>
      </div>
    }
}
