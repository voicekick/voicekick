use std::sync::Arc;

use dioxus::{
    hooks::{use_context, use_coroutine_handle, UnboundedReceiver},
    signals::{GlobalSignal, Signal, WritableVecExt},
};
use futures_util::StreamExt;
use server::events::{BroadRecvError, Event, EventEmitter, ServerEventsBroadcaster};
use tracing::{error, info, warn};

use command_parser::{CommandParser, CommandParserError};
use commands::{
    llm::{
        AnthropicChatCommand, DeepseekChatCommand, GoogleChatCommand, OllamaChatCommand,
        OpenaiChatCommand,
    },
    HttpCommand, VoiceLogCommand,
};

use crate::states::VoiceConfigState;

use super::VoiceKickCommand;

#[derive(Debug)]
#[non_exhaustive]
pub enum ServerCommand {}

pub fn init_base_commands(command_parser: &CommandParser) -> Result<(), CommandParserError> {
    command_parser
        .register_namespace("test", Some(1))?
        .register_command("test", "voice log", Arc::new(VoiceLogCommand))?;

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

pub static AVAILABLE_COMMANDS: GlobalSignal<Vec<(String, Vec<String>)>> = Signal::global(Vec::new);

pub async fn server_service(mut rx: UnboundedReceiver<ServerCommand>) {
    let server_events_broadcaster_state = use_context::<ServerEventsBroadcaster>();
    let command_parser_state = use_context::<CommandParser>();
    let voice_command_task = use_coroutine_handle::<VoiceKickCommand>();
    let mut voice_config_state = use_context::<VoiceConfigState>();

    let mut server_events_rx = server_events_broadcaster_state.subscribe();

    let mut update_whisper_commands_boosts = || {
        let commands: Vec<(String, Vec<String>)> = command_parser_state
            .commands()
            .unwrap_or_default()
            .clone()
            .into_iter()
            .collect();

        let boost_commands: Vec<String> = commands
            .clone()
            .into_iter()
            .flat_map(|(k, mut v)| {
                v.push(k);
                v
            })
            .collect();

        voice_config_state.command_words.extend(boost_commands);
        voice_command_task.send(VoiceKickCommand::UpdateWhisper);
        *AVAILABLE_COMMANDS.write() = commands.clone();
    };

    init_base_commands(&command_parser_state).unwrap();
    update_whisper_commands_boosts();

    loop {
        tokio::select! {
            event = server_events_rx.recv() => {
                match event {
                    Ok(event) => {
                        match event {
                            Event::RegisterHttpCommandment(commandment) => {
                                info!("Server event received: {:?}", commandment);

                                // TODO: fix unwrap
                                command_parser_state.register_namespace(commandment.namespace().as_ref(), None).unwrap();
                                // TODO: fix unwrap
                                command_parser_state.register_command(
                                    commandment.namespace().as_ref(),
                                    commandment.command().as_ref(),
                                    Arc::new(HttpCommand::new(commandment.clone()))
                                ).unwrap();

                                update_whisper_commands_boosts();
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
