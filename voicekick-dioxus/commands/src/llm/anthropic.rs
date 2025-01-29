use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use llm::{
    backends::anthropic::Anthropic,
    chat::{ChatMessage, ChatProvider, ChatRole, Tool},
};

use super::voicekick_var;

pub struct AnthropicChatCommand {
    backend: Anthropic,
}

impl Default for AnthropicChatCommand {
    fn default() -> Self {
        let api_key = voicekick_var("ANTHROPIC_API_KEY");
        let model = "claude-3-5-sonnet-20240620";
        let max_tokens = Some(512);
        let temperature = Some(0.7);
        let timeout_seconds = Some(5);
        let system_prompt: Option<String> = None;
        let should_stream: Option<bool> = None;
        let top_p: Option<f32> = None;
        let top_k: Option<u32> = None;
        let tools: Option<Vec<Tool>> = None;

        let backend = Anthropic::new(
            api_key.expect("TODO: fix"),
            Some(model.into()),
            max_tokens,
            temperature,
            timeout_seconds,
            system_prompt,
            should_stream,
            top_p,
            top_k,
            tools,
        );

        Self { backend }
    }
}

#[async_trait]
impl CommandAction for AnthropicChatCommand {
    async fn execute(&self, args: CommandArgs) -> CommandResult {
        match args {
            CommandArgs::Some(content) => {
                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    content,
                }];

                let text = self.backend.chat(&messages).await?;

                Ok(CommandOutput::Ok(Some(text)))
            }
            CommandArgs::None => Ok(CommandOutput::Ok(None)),
            _ => unimplemented!(),
        }
    }
}
