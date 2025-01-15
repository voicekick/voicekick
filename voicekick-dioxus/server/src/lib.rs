use axum::{routing::get, Router};
use tracing::info;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Start the server
pub async fn serve() -> Result<(), BoxError> {
    use std::net::SocketAddr;

    async fn app_endpoint() -> String {
        // render the rsx! macro to HTML
        "AWd".to_string()
    }

    let listen_addr = SocketAddr::from(([0, 0, 0, 0], 1313));

    info!("axum: listening on {}", listen_addr);

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;

    axum::serve(
        listener,
        Router::new()
            .route("/", get(app_endpoint))
            .into_make_service(),
    )
    .await
    .map_err(Into::into)
}
