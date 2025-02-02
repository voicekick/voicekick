use async_trait::async_trait;
use tokio::sync::broadcast::{
    self,
    error::{RecvError, SendError},
};

use proto::Commandment;
use tracing::{error, info};

#[derive(Debug, Clone)]
pub enum Event {
    RegisterHttpCommandment(Commandment),
}

pub type BroadTx = broadcast::Sender<Event>;
pub type BroadRx = broadcast::Receiver<Event>;
pub type BroadSendError = SendError<Event>;
pub type BroadRecvError = RecvError;

#[async_trait]
pub trait EventEmitter: Send + Sync {
    async fn emit(&self, event: Event) -> Result<usize, BroadSendError>;
    fn subscribe(&self) -> BroadRx;
}

#[derive(Debug, Clone)]
pub struct ServerEventsBroadcaster {
    broad_tx: BroadTx,
}

impl ServerEventsBroadcaster {
    pub fn new(capacity: usize) -> Self {
        let (broad_tx, mut broad_rx) = broadcast::channel(capacity);

        // A bit of a hack to prevent the receiver from being dropped
        // nevertheless, it's a good idea to spawn the receiver in a separate task for logging
        tokio::spawn(async move {
            loop {
                match broad_rx.recv().await {
                    Ok(event) => {
                        info!(
                            "ServerEventsBroadcaster logger: received event: {:?}",
                            event
                        );
                    }
                    Err(BroadRecvError::Closed) => {
                        error!("ServerEventsBroadcaster logger: broadcaster receiver closed");
                        break;
                    }
                    Err(BroadRecvError::Lagged(n)) => {
                        info!("ServerEventsBroadcaster logger: lagged by {} events", n);
                    }
                }
            }
        });

        Self { broad_tx }
    }
}

#[async_trait]
impl EventEmitter for ServerEventsBroadcaster {
    async fn emit(&self, event: Event) -> Result<usize, SendError<Event>> {
        self.broad_tx.send(event)
    }

    fn subscribe(&self) -> BroadRx {
        self.broad_tx.subscribe()
    }
}
