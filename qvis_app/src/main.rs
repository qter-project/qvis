use axum::{
    Router,
    body::Body as AxumBody,
    extract::{FromRef, Path, RawQuery, State},
    http::{HeaderMap, Request},
    response::{IntoResponse, Response as AxumResponse},
    routing::{get, post},
};
use leptos::{logging::log, prelude::*};
use leptos_axum::{
    AxumRouteListing, LeptosRoutes, file_and_error_handler_with_context,
    generate_route_list_with_exclusions_and_ssg_and_context, handle_server_fns_with_context,
};
use leptos_ws::WsSignals;
use qvis_app::app::*;
use tokio::net::TcpListener;

#[derive(Clone, FromRef)]
pub struct AppState {
    server_signals: WsSignals,
    routes: Option<Vec<AxumRouteListing>>,
    options: LeptosOptions,
}

async fn server_fn_handler(
    State(state): State<AppState>,
    _path: Path<String>,
    _headers: HeaderMap,
    _query: RawQuery,
    request: Request<AxumBody>,
) -> impl IntoResponse {
    handle_server_fns_with_context(
        move || {
            provide_context(state.options.clone());
            provide_context(state.server_signals.clone());
        },
        request,
    )
    .await
}

async fn leptos_routes_handler(state: State<AppState>, req: Request<AxumBody>) -> AxumResponse {
    let state1 = state.0.clone();
    let options1 = state.0.options.clone();
    let handler = leptos_axum::render_route_with_context(
        state.routes.clone().unwrap(),
        move || {
            provide_context(state1.options.clone());
            provide_context(state1.server_signals.clone());
        },
        move || shell(options1.clone()),
    );
    handler(state, req).await.into_response()
}

#[tokio::main]
async fn main() {
    let conf = get_configuration(None).unwrap();
    let leptos_options = conf.leptos_options;
    let addr = leptos_options.site_addr;

    let server_signals = WsSignals::new();
    let mut state = AppState {
        options: leptos_options.clone(),
        routes: None,
        server_signals: server_signals.clone(),
    };
    let state1 = state.clone();
    let state2 = state.clone();

    let (routes, _) = generate_route_list_with_exclusions_and_ssg_and_context(
        || view! { <App /> },
        None,
        move || provide_context(state1.server_signals.clone()),
    );
    state.routes = Some(routes.clone());

    let app = Router::new()
        .route(
            "/api/{*fn_name}",
            post(server_fn_handler).get(server_fn_handler),
        )
        .leptos_routes_with_handler(routes, get(leptos_routes_handler))
        .fallback(file_and_error_handler_with_context::<AppState, _>(
            move || provide_context(state2.server_signals.clone()),
            shell,
        ))
        .with_state(state);

    log!("listening on http://{addr}");
    let listener = TcpListener::bind(&addr)
        .await
        .expect("couldn't bind to address");
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
