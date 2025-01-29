use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use llm::{
    backends::ollama::Ollama,
    chat::{ChatMessage, ChatProvider, ChatRole},
};

use super::voicekick_var;

pub struct OllamaChatCommand {
    backend: Ollama,
}

impl Default for OllamaChatCommand {
    fn default() -> Self {
        let base_url = voicekick_var("OLLAMA_URL").unwrap_or("http://localhost:11411".into());
        let api_key = voicekick_var("OLLAMA_API_KEY");

        // TODO: should be configurable
        let model = "llama3.1";
        let max_tokens = Some(512);
        let temperature = Some(0.7);
        let timeout_seconds = Some(5);
        let system_prompt: Option<String> = None;
        let should_stream: Option<bool> = None;
        let top_p: Option<f32> = None;
        let top_k: Option<u32> = None;

        let backend = Ollama::new(
            base_url,
            api_key,
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
impl CommandAction for OllamaChatCommand {
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
