use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
};

pub use async_trait::async_trait;
use strsim::levenshtein;

mod error;
pub use error::CommandParserError;

#[derive(Debug)]
#[non_exhaustive]
pub enum CommandOutput {
    Ok(Option<String>),
    None,
}

pub type CommandResult = Result<CommandOutput, Box<dyn std::error::Error>>;

#[derive(Debug)]
#[non_exhaustive]
pub enum CommandArgs {
    Some(String),
    None,
}

impl Default for CommandArgs {
    fn default() -> Self {
        CommandArgs::None
    }
}

#[async_trait]
pub trait CommandAction: Sync + Send {
    async fn execute(&self, args: CommandArgs)
        -> Result<CommandOutput, Box<dyn std::error::Error>>;
}

type CommandReturn<'s> = (Arc<dyn CommandAction>, CommandArgs);

/// A simple command parser that matches commands based on Levenshtein distance
#[derive(Clone, Default)]
pub struct CommandParser {
    inner: Arc<CommandParserInner>,
}

impl CommandParser {
    /// Creates a new `CommandParserBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new namespace with a specific threshold
    // TODO: add parsing strategy per command with several options (levenstein, regex) and allow custom
    pub fn register_namespace(
        &self,
        name: &str,
        threshold: Option<usize>,
    ) -> Result<&Self, CommandParserError> {
        let mut namespaces = self.inner.namespaces.write()?;

        if let Some(pos) = namespaces.iter().position(|ns| ns.name == name) {
            namespaces[pos].threshold = threshold.unwrap_or(1);
        } else {
            namespaces.push(Namespace {
                name: name.to_string(),
                threshold: threshold.unwrap_or(1),
                commands: Vec::new(),
            });
        }

        Ok(self)
    }

    /// Registers a new command under a specific namespace
    pub fn register_command(
        &self,
        namespace: impl AsRef<str>,
        name: impl AsRef<str>,
        command: Arc<dyn CommandAction>,
    ) -> Result<&Self, CommandParserError> {
        if let Some(ns) = self
            .inner
            .namespaces
            .write()?
            .iter_mut()
            .find(|ns| ns.name == namespace.as_ref())
        {
            let value = Command {
                command_parts_count: name.as_ref().split_whitespace().count(),
                name: name.as_ref().into(),
                command,
            };

            if let Some(pos) = ns.commands.iter().position(|cmd| cmd.name == name.as_ref()) {
                ns.commands[pos] = value
            } else {
                ns.commands.push(value);
            }

            Ok(self)
        } else {
            Err(CommandParserError::NamespaceNotFound(
                namespace.as_ref().to_string(),
            ))
        }
    }

    /// Removes a command from a namespace
    pub fn remove_command(
        &self,
        namespace: impl AsRef<str>,
        name: impl AsRef<str>,
    ) -> Result<bool, CommandParserError> {
        if let Some(ns) = self
            .inner
            .namespaces
            .write()?
            .iter_mut()
            .find(|ns| ns.name == namespace.as_ref())
        {
            if let Some(index) = ns.commands.iter().position(|cmd| cmd.name == name.as_ref()) {
                ns.commands.remove(index);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Removes a namespace and all its commands
    pub fn remove_namespace(&self, name: impl AsRef<str>) -> Result<bool, CommandParserError> {
        let mut namespaces = self.inner.namespaces.write()?;
        if let Some(index) = namespaces.iter().position(|ns| ns.name == name.as_ref()) {
            namespaces.remove(index);
            return Ok(true);
        }
        Ok(false)
    }

    /// Retrieve all availabe namescapes with commands
    pub fn commands(&self) -> Result<BTreeMap<String, Vec<String>>, CommandParserError> {
        let commands = self
            .inner
            .namespaces
            .read()?
            .iter()
            .map(|namespace| {
                (
                    namespace.name.clone(),
                    namespace
                        .commands
                        .iter()
                        .map(|command| command.name.clone())
                        .collect(),
                )
            })
            .collect();

        Ok(commands)
    }
}

#[derive(Default)]
struct CommandParserInner {
    namespaces: RwLock<Vec<Namespace>>,
}

struct Namespace {
    name: String,
    threshold: usize, // Maximum Levenshtein distance for matching
    commands: Vec<Command>,
}

struct Command {
    name: String,
    command_parts_count: usize,
    command: Arc<dyn CommandAction>,
}

struct Commander<'s> {
    command: &'s Command,
    user_command_text: String,
    distance: usize,
}

impl CommandParser {
    /// Parses a command input with a namespace prefix and executes the closest matching command
    pub fn parse(&self, input: &str) -> Result<CommandReturn, CommandParserError> {
        // Split input into namespace and rest of the command
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(CommandParserError::InvalidFormat);
        }
        let namespace_name = parts[0];

        // Find the namespace
        if let Some(ns) = self
            .inner
            .namespaces
            .read()?
            .iter()
            .find(|ns| ns.name == namespace_name)
        {
            // Find the closest matching command
            if let Some(command) = ns
                .commands
                .iter()
                .map(|command| {
                    // Defined commands will have multiple expected tokens to match hence
                    // check for the first n tokens of a given command
                    let user_command_text = parts
                        .get(1..command.command_parts_count + 1)
                        .map(|p| p.join(" "))
                        .unwrap_or(String::new());

                    Commander {
                        distance: levenshtein(&command.name, &user_command_text),
                        user_command_text,
                        command,
                    }
                })
                .min_by_key(|cmd| cmd.distance)
            {
                if command.distance <= ns.threshold {
                    let args = parts
                        .get(command.command.command_parts_count + 1..)
                        .map(|s| CommandArgs::Some(s.join(" ")))
                        .unwrap_or_default();

                    Ok((command.command.command.clone(), args))
                } else {
                    Err(CommandParserError::NoCloseMatches(
                        command.user_command_text.to_string(),
                    ))
                }
            } else {
                Err(CommandParserError::CommandNotFound(
                    parts.join(" ").to_string(),
                ))
            }
        } else {
            Err(CommandParserError::NamespaceNotFound(
                namespace_name.to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyCommand;

    #[async_trait]
    impl CommandAction for DummyCommand {
        async fn execute(&self, _args: CommandArgs) -> CommandResult {
            Ok(CommandOutput::Ok(None))
        }
    }

    fn dummy_action() -> Arc<DummyCommand> {
        Arc::new(DummyCommand {})
    }

    #[test]
    fn test_register_namespace() {
        let parser = CommandParser::new();

        parser.register_namespace("test", Some(3)).unwrap();

        let namespaces = parser.inner.namespaces.read().unwrap();

        assert_eq!(namespaces.len(), 1);
        assert_eq!(namespaces[0].name, "test");
        assert_eq!(namespaces[0].threshold, 3);
    }

    #[test]
    fn test_register_command() {
        let parser = CommandParser::new();

        parser
            .register_namespace("test", Some(3))
            .unwrap()
            .register_command("test", "dummy command", dummy_action())
            .unwrap();

        let namespaces = parser.inner.namespaces.read().unwrap();

        assert_eq!(namespaces[0].commands.len(), 1);
        assert_eq!(namespaces[0].commands[0].name, "dummy command");
    }

    #[test]
    fn test_register_command_invalid_namespace() {
        let parser = CommandParser::new();

        let result = parser.register_command("invalid", "dummy command", dummy_action());

        assert!(result.is_err());
        if let Err(CommandParserError::NamespaceNotFound(ns)) = result {
            assert_eq!(ns, "invalid");
        } else {
            panic!("Expected NamespaceNotFound error");
        }
    }

    #[test]
    fn test_parse_and_execute_valid_command() {
        let parser = CommandParser::new();

        parser
            .register_namespace("test", Some(3))
            .unwrap()
            .register_command("test", "dummy command", dummy_action())
            .unwrap();

        let result = parser.parse("test dummy          command");

        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_and_execute_invalid_format() {
        let parser = CommandParser::new();

        let result = parser.parse("invalid_format");

        assert!(result.is_err());
        assert!(matches!(result, Err(CommandParserError::InvalidFormat)));
    }

    #[test]
    fn test_parse_and_execute_unknown_namespace() {
        let parser = CommandParser::new();

        let result = parser.parse("unknown_namespace command");

        assert!(result.is_err());
        if let Err(CommandParserError::NamespaceNotFound(ns)) = result {
            assert_eq!(ns, "unknown_namespace");
        } else {
            panic!("Expected NamespaceNotFound error");
        }
    }

    #[test]
    fn test_parse_and_execute_unknown_command() {
        let parser = CommandParser::new();

        parser.register_namespace("test", Some(3)).unwrap();

        let result = parser.parse("test unknown_command");

        assert!(result.is_err());
        if let Err(CommandParserError::CommandNotFound(cmd)) = result {
            assert_eq!(cmd, "test unknown_command");
        } else {
            panic!("Expected CommandNotFound error");
        }
    }

    #[test]
    fn test_parse_and_execute_no_close_matches() {
        let parser = CommandParser::new();

        parser
            .register_namespace("test", None)
            .unwrap() // Low threshold for testing
            .register_command("test", "dummy command", dummy_action())
            .unwrap();

        let result = parser.parse("test dommm command"); // Intentional typo

        assert!(result.is_err());
        if let Err(CommandParserError::NoCloseMatches(cmd)) = result {
            assert_eq!(cmd, "dommm command");
        } else {
            panic!("Expected NoCloseMatches error");
        }
    }

    #[test]
    fn test_parse_and_execute_close_match() {
        let parser = CommandParser::new();

        parser
            .register_namespace("test", None)
            .unwrap() // Toleration for one character difference
            .register_namespace("move", Some(3))
            .unwrap() // Toleration for three character difference
            .register_namespace("move2", Some(5))
            .unwrap() // Toleration for five character difference
            .register_command("test", "dummy command", dummy_action())
            .unwrap()
            .register_command("move", "backwards command", dummy_action())
            .unwrap()
            .register_command("move2", "backwards command", dummy_action())
            .unwrap();

        let result = parser.parse("test dummm command"); // Intentional typo
        assert!(result.is_ok());

        let result = parser.parse("move backwards command");
        assert!(result.is_ok());

        let result = parser.parse("move backward command");
        assert!(result.is_ok());

        let result = parser.parse("move back command");
        assert!(result.is_err());

        // Toleration for higher threshold
        let result = parser.parse("move2 back command");
        assert!(result.is_ok());
    }
}
