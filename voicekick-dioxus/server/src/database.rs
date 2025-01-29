use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use tokio::sync::RwLock;

use proto::{Commandment, Namespace};

#[async_trait]
pub trait Database: Send + Sync {
    async fn all(&self) -> Vec<Commandment>;
    async fn add(&self, commandment: Commandment) -> bool;
}

#[derive(Clone)]
pub struct MemoryDB {
    items: Arc<RwLock<HashMap<Namespace, HashSet<Commandment>>>>,
}

impl MemoryDB {
    pub fn new() -> Self {
        Self {
            items: Default::default(),
        }
    }
}

#[async_trait]
impl Database for MemoryDB {
    async fn all(&self) -> Vec<Commandment> {
        self.items
            .read()
            .await
            .values()
            .flatten()
            .cloned()
            .collect()
    }

    async fn add(&self, commandment: Commandment) -> bool {
        self.items
            .write()
            .await
            .entry(commandment.namespace().clone())
            .or_default()
            .insert(commandment)
    }
}
