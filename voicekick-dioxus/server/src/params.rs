use serde::Deserialize;
use url::Url;
use validator::Validate;

use proto::{Command, Commandment, Namespace, ServerInfo};

#[derive(Deserialize, Validate)]
pub struct ServerCommandParams {
    #[serde(alias = "host")]
    pub url: Url,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    #[validate(length(max = 1000))]
    pub path: Option<String>,
    #[serde(default)]
    #[validate(length(max = 1000))]
    pub query: Option<String>,
}

impl From<ServerCommandParams> for ServerInfo {
    fn from(params: ServerCommandParams) -> Self {
        ServerInfo::new(params.url, params.port, params.path, params.query)
    }
}

#[derive(Deserialize, Validate)]
pub struct CommandParams {
    /// The namespace to execute the command in
    #[validate(length(min = 1, max = 100))]
    pub namespace: String,

    /// The command to execute
    #[validate(length(min = 1, max = 200))]
    pub command: String,

    /// The server to execute the command on
    pub server: ServerCommandParams,
}

impl From<CommandParams> for Commandment {
    fn from(params: CommandParams) -> Self {
        Commandment::new(
            Namespace::new(params.namespace),
            Command::new(params.command),
            params.server.into(),
        )
    }
}
