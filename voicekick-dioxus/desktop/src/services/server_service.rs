use std::sync::Arc;

use command_parser::CommandParser;
use commands::HttpCommand;
use dioxus::hooks::{use_context, UnboundedReceiver};
use futures_util::StreamExt;
use server::events::{BroadRecvError, Event, EventEmitter, ServerEventsBroadcaster};
use tracing::{error, info, warn};

#[derive(Debug)]
#[non_exhaustive]
pub enum ServerCommand {}

pub async fn server_service(mut rx: UnboundedReceiver<ServerCommand>) {
    let server_events_broadcaster_state = use_context::<ServerEventsBroadcaster>();
    let command_parser_state = use_context::<CommandParser>();

    let mut server_events_rx = server_events_broadcaster_state.subscribe();

    loop {
        tokio::select! {
            event = server_events_rx.recv() => {
                match event {
                    Ok(event) => {
                        match event {
                            Event::RegisterCommandment(commandment) => {
                                info!("Server event received: {:?}", commandment);

                                // TODO: fix unwrap
                                command_parser_state.register_namespace(commandment.namespace().as_ref(), None).unwrap();
                                // TODO: fix unwrap
                                command_parser_state.register_command(
                                    commandment.namespace().as_ref(),
                                    commandment.command().as_ref(),
                                    Arc::new(HttpCommand::new(commandment.clone()))
                                ).unwrap();

                            }
                        }
                    },
                    Err(BroadRecvError::Closed) => {
                        error!("Server events channel closed");
                        break;
                    }
                    Err(BroadRecvError::Lagged(skipped_messages)) => {
                        warn!("Server events Skipped {} messages", skipped_messages);
                        break;
                    }
                }
            }
            command = rx.next() => {
              info!("Server command received: {:?}", command);
            }
        }
    }
}
