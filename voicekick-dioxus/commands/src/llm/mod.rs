use std::env::var;

fn voicekick_var(key: &str) -> Option<String> {
    let voicekick_var = format!("VOICEKICK_{}", key.to_uppercase());

    var(&voicekick_var).or(var(key)).ok()
}

mod anthropic;
pub use anthropic::AnthropicChatCommand;

mod openai;
pub use openai::OpenaiChatCommand;

mod ollama;
pub use ollama::OllamaChatCommand;

mod deepseek;
pub use deepseek::DeepseekChatCommand;

mod google;
pub use google::GoogleChatCommand;
