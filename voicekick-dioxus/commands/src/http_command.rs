use std::time::Duration;

use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};
use proto::Commandment;
use reqwest::Client;
use serde::Serialize;

fn build_reqwest_client() -> Client {
    let idle_timeout = Duration::from_secs(13);

    Client::builder()
        .referer(false)
        .redirect(reqwest::redirect::Policy::limited(1))
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .connect_timeout(Duration::from_millis(1300))
        .pool_max_idle_per_host(3)
        .pool_idle_timeout(idle_timeout)
        .tcp_nodelay(true)
        .tcp_keepalive(Some(idle_timeout + Duration::from_secs(13)))
        .danger_accept_invalid_certs(false)
        .use_rustls_tls()
        .hickory_dns(true)
        .build()
        .expect("valid reqwest client")
}

pub struct HttpCommand {
    url: String,
    client: Client,
    commandment: Commandment,
}

#[derive(Debug, Serialize)]
struct HttpRequest {
    namespace: String,
    command: String,
    args: Option<String>,
}

impl HttpCommand {
    pub fn new(commandment: Commandment) -> Self {
        Self {
            // TODO: fix unwrap
            url: commandment.server().url().unwrap().to_string(),
            client: build_reqwest_client(),
            commandment,
        }
    }
}

#[async_trait]
impl CommandAction for HttpCommand {
    async fn execute(&self, args: CommandArgs) -> CommandResult {
        let request = HttpRequest {
            namespace: self.commandment.namespace().as_ref().to_string(),
            command: self.commandment.command().as_ref().to_string(),
            args: match args {
                CommandArgs::Some(a) => Some(a),
                CommandArgs::None => None,
                _ => unimplemented!(),
            },
        };

        let response = self
            .client
            .clone()
            .post(&self.url)
            .json(&request)
            .send()
            .await?;

        let body = response.text().await?;
        if body.is_empty() {
            return Ok(CommandOutput::None);
        } else {
            Ok(CommandOutput::Ok(Some(body)))
        }
    }
}
