use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use llm::{
    backends::google::Google,
    chat::{ChatMessage, ChatProvider, ChatRole},
};

use super::voicekick_var;

pub struct GoogleChatCommand {
    backend: Google,
}

impl Default for GoogleChatCommand {
    fn default() -> Self {
        let api_key = voicekick_var("GOOGLE_API_KEY");
        let model = "gemini-2.0-flash-exp";
        let max_tokens = Some(8512);
        let temperature = Some(0.7);
        let timeout_seconds = Some(5);
        let system_prompt: Option<String> = None;
        let should_stream: Option<bool> = None;
        let top_p: Option<f32> = None;
        let top_k: Option<u32> = None;

        let backend = Google::new(
            api_key.expect("TODO: fix"),
            Some(model.into()),
            max_tokens,
            temperature,
            timeout_seconds,
            system_prompt,
            should_stream,
            top_p,
            top_k,
        );

        Self { backend }
    }
}

#[async_trait]
impl CommandAction for GoogleChatCommand {
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
