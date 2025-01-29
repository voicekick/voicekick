use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use llm::{
    backends::deepseek::DeepSeek,
    chat::{ChatMessage, ChatProvider, ChatRole},
};

use super::voicekick_var;

pub struct DeepseekChatCommand {
    backend: DeepSeek,
}

impl Default for DeepseekChatCommand {
    fn default() -> Self {
        let api_key = voicekick_var("DEEPSEEK_API_KEY");
        let model = "deepseek-reasoner";
        let max_tokens = Some(512);
        let temperature = Some(0.7);
        let timeout_seconds = Some(5);
        let system_prompt: Option<String> = None;
        let should_stream: Option<bool> = None;

        let backend = DeepSeek::new(
            api_key.unwrap_or("TODO: fix".into()),
            Some(model.into()),
            max_tokens,
            temperature,
            timeout_seconds,
            system_prompt,
            should_stream,
        );

        Self { backend }
    }
}

#[async_trait]
impl CommandAction for DeepseekChatCommand {
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
