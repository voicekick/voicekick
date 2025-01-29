use std::sync::Arc;

use crate::database::{Database, MemoryDB};

#[derive(Clone)]
pub struct ServerState {
    pub database: Arc<dyn Database>,
}

impl ServerState {
    pub fn new(database: Arc<dyn Database>) -> Self {
        Self { database }
    }

    pub fn memory() -> Self {
        Self::new(Arc::new(MemoryDB::new()))
    }
}
