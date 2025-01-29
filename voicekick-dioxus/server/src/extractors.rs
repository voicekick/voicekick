use axum::{
    extract::{rejection::JsonRejection, FromRequest, Request},
    Json,
};
use serde::de::DeserializeOwned;
use validator::Validate;

use crate::error::ServerError;

/// Validation helper for JSON extractor
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonValidated<T>(pub T);

impl<T, S> FromRequest<S> for JsonValidated<T>
where
    S: Send + Sync,
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    T: DeserializeOwned + Validate + Send + Sync,
{
    type Rejection = ServerError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state).await?;
        value.validate()?;
        Ok(Self(value))
    }
}
