use std::sync::{PoisonError, RwLockReadGuard, RwLockWriteGuard};

use crate::Namespace;

#[derive(Debug)]
pub enum CommandParserError {
    PoisonError(String),
    InvalidFormat,
    NamespaceNotFound(String),
    CommandNotFound(String),
    NoCloseMatches(String),
}

impl From<PoisonError<RwLockWriteGuard<'_, Vec<Namespace>>>> for CommandParserError {
    fn from(e: PoisonError<RwLockWriteGuard<Vec<Namespace>>>) -> Self {
        CommandParserError::PoisonError(e.to_string())
    }
}

impl From<PoisonError<RwLockReadGuard<'_, Vec<Namespace>>>> for CommandParserError {
    fn from(e: PoisonError<RwLockReadGuard<Vec<Namespace>>>) -> Self {
        CommandParserError::PoisonError(e.to_string())
    }
}
