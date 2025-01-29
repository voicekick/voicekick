use std::sync::Arc;

use command_parser::{CommandParser, CommandParserError};
use commands::{
    llm::{
        AnthropicChatCommand, DeepseekChatCommand, GoogleChatCommand, OllamaChatCommand,
        OpenaiChatCommand,
    },
    VoiceLogCommand,
};

pub fn init(command_parser: &CommandParser) -> Result<(), CommandParserError> {
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
