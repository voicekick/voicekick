use std::hash::{Hash, Hasher};

use url::{ParseError, Url};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ServerInfo {
    pub(crate) host: Url,
    pub(crate) port: Option<u16>,

    pub(crate) path: Option<String>,
    pub(crate) query: Option<String>,
}

impl ServerInfo {
    /// Initialize a new server info
    pub fn new(host: Url, port: Option<u16>, path: Option<String>, query: Option<String>) -> Self {
        Self {
            host,
            port,
            path,
            query,
        }
    }

    /// Get the full URL of the server
    pub fn url(&self) -> Result<Url, ParseError> {
        let mut url = self.host.clone();
        if let Some(port) = self.port {
            url.set_port(Some(port))
                .map_err(|_| ParseError::InvalidPort)?;
        }
        if let Some(path) = &self.path {
            url.set_path(path);
        }
        if let Some(query) = &self.query {
            url.set_query(Some(query));
        }

        Ok(url)
    }
}

/// Commandment houses all the commands from the client
#[derive(Clone, Eq, Debug)]
pub struct Commandment {
    pub(crate) namespace: Namespace,
    pub(crate) command: Command,

    pub(crate) server: ServerInfo,
}

impl Commandment {
    /// Initialize a new commandment
    pub fn new(namespace: Namespace, command: Command, server: ServerInfo) -> Self {
        Self {
            namespace,
            command,
            server,
        }
    }
}

impl PartialEq for Commandment {
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.command == other.command
    }
}

impl Commandment {
    /// Get the namespace of the command
    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Get the command to be executed
    pub fn command(&self) -> &Command {
        &self.command
    }

    pub fn server(&self) -> &ServerInfo {
        &self.server
    }
}

impl Hash for Commandment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.command.hash(state);
    }
}

/// A command to be accepted by the server
#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Namespace {
    /// The command to be executed
    pub namespace: String,
}

impl AsRef<str> for Namespace {
    fn as_ref(&self) -> &str {
        &self.namespace
    }
}

impl Namespace {
    /// Initialize a new namespace
    pub fn new(namespace: String) -> Self {
        Self { namespace }
    }
}

/// A command to be accepted by the server
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Command {
    /// The command to be executed
    pub command: String,
}

impl Command {
    /// Initialize a new command
    pub fn new(command: String) -> Self {
        Self { command }
    }
}

impl AsRef<str> for Command {
    fn as_ref(&self) -> &str {
        &self.command
    }
}
