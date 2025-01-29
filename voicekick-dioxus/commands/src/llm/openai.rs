use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use llm::{
    backends::openai::OpenAI,
    chat::{ChatMessage, ChatProvider, ChatRole, Tool},
};

use super::voicekick_var;

pub struct OpenaiChatCommand {
    backend: OpenAI,
}

impl Default for OpenaiChatCommand {
    fn default() -> Self {
        let api_key = voicekick_var("OPENAI_API_KEY");
        let model = "gpt-3.5-turbo";
        let max_tokens = Some(512);
        let temperature = Some(0.7);
        let timeout_seconds = Some(5);
        let system_prompt: Option<String> = None;
        let should_stream: Option<bool> = None;
        let top_p: Option<f32> = None;
        let top_k: Option<u32> = None;
        let tools: Option<Vec<Tool>> = None;
        let embedding_encoding_format: Option<String> = None;
        let embedding_dimensions: Option<u32> = None;

        let open_ai = OpenAI::new(
            api_key.unwrap_or("TODO: fix".into()),
            Some(model.into()),
            max_tokens,
            temperature,
            timeout_seconds,
            system_prompt,
            should_stream,
            top_p,
            top_k,
            embedding_encoding_format,
            embedding_dimensions,
            tools,
        );
        let backend = open_ai;

        Self { backend }
    }
}

#[async_trait]
impl CommandAction for OpenaiChatCommand {
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
