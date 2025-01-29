use std::sync::Arc;

use command_parser::{CommandParser, CommandParserError};
use commands::{
    llm::{
        AnthropicChatCommand, DeepseekChatCommand, GoogleChatCommand, OllamaChatCommand,
        OpenaiChatCommand,
    },
    HttpCommand, VoiceLogCommand,
};
use dioxus::hooks::{use_context, UnboundedReceiver};
use futures_util::StreamExt;
use server::events::{BroadRecvError, Event, EventEmitter, ServerEventsBroadcaster};
use tracing::{error, info, warn};

#[derive(Debug)]
#[non_exhaustive]
pub enum ServerCommand {}

fn init_commands(command_parser: &CommandParser) -> Result<(), CommandParserError> {
    command_parser
        .register_namespace("test", Some(1))?
        .register_command("test", "voice log", Arc::new(VoiceLogCommand::default()))?;

    command_parser
        .register_namespace("anthropic", Some(1))?
        .register_command(
            "anthropic",
            "chat",
            Arc::new(AnthropicChatCommand::default()),
        )?;

    command_parser
        .register_namespace("openai", Some(1))?
        .register_command("openai", "chat", Arc::new(OpenaiChatCommand::default()))?;

    command_parser
        .register_namespace("ollama", Some(1))?
        .register_command("ollama", "chat", Arc::new(OllamaChatCommand::default()))?;

    command_parser
        .register_namespace("deepseek", Some(1))?
        .register_command("deepseek", "chat", Arc::new(DeepseekChatCommand::default()))?;

    command_parser
        .register_namespace("google", Some(1))?
        .register_command("google", "chat", Arc::new(GoogleChatCommand::default()))?;

    Ok(())
}

pub async fn server_service(mut rx: UnboundedReceiver<ServerCommand>) {
    let server_events_broadcaster_state = use_context::<ServerEventsBroadcaster>();
    let command_parser_state = use_context::<CommandParser>();

    let mut server_events_rx = server_events_broadcaster_state.subscribe();

    init_commands(&command_parser_state).expect("TODO: fix");

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
