use std::net::SocketAddr;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Router};
use error::ServerError;
use events::{Event, EventEmitter, ServerEventsBroadcaster};
use extractors::JsonValidated;
use params::CommandParams;
use tokio::net::TcpListener;
use tracing::info;

mod database;
pub use database::Database;
mod error;
pub mod events;
mod extractors;
mod params;
mod state;

/// Start the server
pub async fn serve(routes: Router) -> Result<(), ServerError> {
    let listen_addr = SocketAddr::from(([0, 0, 0, 0], 1313));

    info!("axum: listening on {}", listen_addr);

    let listener = TcpListener::bind(listen_addr).await?;

    axum::serve(listener, routes.into_make_service())
        .await
        .map_err(Into::into)
}

async fn home() -> String {
    "VoiceKick commands server".to_string()
}

async fn register_command(
    State(state): State<ServerEventsBroadcaster>,
    JsonValidated(params): JsonValidated<CommandParams>,
) -> Result<impl IntoResponse, ServerError> {
    let commandment = params.into();

    let _ = state.emit(Event::RegisterCommandment(commandment)).await?;

    Ok(StatusCode::ACCEPTED)
}

/// Define the routes for the server
pub fn routes(broadcaster: ServerEventsBroadcaster) -> Router {
    let v1 = Router::new().nest(
        "/v1",
        Router::new().route("/command", post(register_command).delete(home)),
    );

    let api = Router::new().nest("/api", v1);

    Router::new()
        .merge(api)
        .fallback(home)
        .with_state(broadcaster)
}
