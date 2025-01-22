use std::sync::Arc;

use strsim::levenshtein;

#[derive(Debug)]
#[non_exhaustive]
pub enum CommandResult {
    Ok(Option<String>),
    Error(Box<dyn std::error::Error>),
    None,
}

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

pub trait CommandAction: Send + Sync {
    fn execute(&self, args: CommandArgs) -> CommandResult;
}

type CommandReturn<'s> = (&'s dyn CommandAction, CommandArgs);

/// A simple command parser that matches commands based on Levenshtein distance
#[derive(Clone)]
pub struct CommandParser {
    namespaces: Arc<Vec<Namespace>>,
}

struct Namespace {
    name: String,
    threshold: usize, // Maximum Levenshtein distance for matching
    commands: Vec<Command>,
}

struct Command {
    name: String,
    command_parts_count: usize,
    command: Box<dyn CommandAction>,
}

#[derive(Debug)]
pub enum CommandParserError {
    InvalidFormat,
    NamespaceNotFound(String),
    CommandNotFound(String),
    NoCloseMatches(String),
}

#[derive(Default)]
pub struct CommandParserBuilder {
    namespaces: Vec<Namespace>,
}

impl CommandParserBuilder {
    /// Creates a new `CommandParserBuilder`
    pub fn new() -> Self {
        CommandParserBuilder::default()
    }

    /// Registers a new namespace with a specific threshold
    pub fn register_namespace(mut self, name: &str, threshold: Option<usize>) -> Self {
        self.namespaces.push(Namespace {
            name: name.to_string(),
            threshold: threshold.unwrap_or(1),
            commands: Vec::new(),
        });
        self
    }

    /// Registers a new command under a specific namespace
    pub fn register_command(
        mut self,
        namespace: impl AsRef<str>,
        name: impl AsRef<str>,
        command: Box<dyn CommandAction>,
    ) -> Result<Self, CommandParserError> {
        if let Some(ns) = self
            .namespaces
            .iter_mut()
            .find(|ns| ns.name == namespace.as_ref())
        {
            ns.commands.push(Command {
                command_parts_count: name.as_ref().split_whitespace().count(),
                name: name.as_ref().into(),
                command,
            });
            Ok(self)
        } else {
            Err(CommandParserError::NamespaceNotFound(
                namespace.as_ref().to_string(),
            ))
        }
    }

    /// Builds the `CommandParser`
    pub fn build(self) -> CommandParser {
        CommandParser {
            namespaces: self.namespaces.into(),
        }
    }
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
        if let Some(ns) = self.namespaces.iter().find(|ns| ns.name == namespace_name) {
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

                    Ok((&*command.command.command, args))
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

    impl CommandAction for DummyCommand {
        fn execute(&self, _args: CommandArgs) -> CommandResult {
            CommandResult::Ok(None)
        }
    }

    fn dummy_action() -> Box<DummyCommand> {
        Box::new(DummyCommand {})
    }

    #[test]
    fn test_register_namespace() {
        let parser = CommandParserBuilder::new()
            .register_namespace("test", Some(3))
            .build();

        assert_eq!(parser.namespaces.len(), 1);
        assert_eq!(parser.namespaces[0].name, "test");
        assert_eq!(parser.namespaces[0].threshold, 3);
    }

    #[test]
    fn test_register_command() {
        let parser = CommandParserBuilder::new()
            .register_namespace("test", Some(3))
            .register_command("test", "dummy command", dummy_action())
            .unwrap()
            .build();

        assert_eq!(parser.namespaces[0].commands.len(), 1);
        assert_eq!(parser.namespaces[0].commands[0].name, "dummy command");
    }

    #[test]
    fn test_register_command_invalid_namespace() {
        let result = CommandParserBuilder::new().register_command(
            "invalid",
            "dummy command",
            dummy_action(),
        );

        assert!(result.is_err());
        if let Err(CommandParserError::NamespaceNotFound(ns)) = result {
            assert_eq!(ns, "invalid");
        } else {
            panic!("Expected NamespaceNotFound error");
        }
    }

    #[test]
    fn test_parse_and_execute_valid_command() {
        let parser = CommandParserBuilder::new()
            .register_namespace("test", Some(3))
            .register_command("test", "dummy command", dummy_action())
            .unwrap()
            .build();

        let result = parser.parse("test dummy          command");

        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_and_execute_invalid_format() {
        let parser = CommandParserBuilder::new().build();

        let result = parser.parse("invalid_format");

        assert!(result.is_err());
        assert!(matches!(result, Err(CommandParserError::InvalidFormat)));
    }

    #[test]
    fn test_parse_and_execute_unknown_namespace() {
        let parser = CommandParserBuilder::new().build();

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
        let parser = CommandParserBuilder::new()
            .register_namespace("test", Some(3))
            .build();

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
        let parser = CommandParserBuilder::new()
            .register_namespace("test", None) // Low threshold for testing
            .register_command("test", "dummy command", dummy_action())
            .unwrap()
            .build();

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
        let parser = CommandParserBuilder::new()
            .register_namespace("test", None) // Toleration for one character difference
            .register_namespace("move", Some(3)) // Toleration for three character difference
            .register_namespace("move2", Some(5)) // Toleration for five character difference
            .register_command("test", "dummy command", dummy_action())
            .unwrap()
            .register_command("move", "backwards command", dummy_action())
            .unwrap()
            .register_command("move2", "backwards command", dummy_action())
            .unwrap()
            .build();

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
