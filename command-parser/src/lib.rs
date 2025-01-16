use strsim::levenshtein;

/// A simple command parser that matches commands based on Levenshtein distance
pub struct CommandParser {
    namespaces: Vec<Namespace>,
}

pub struct Namespace {
    pub name: String,
    pub threshold: usize, // Maximum Levenshtein distance for matching
    pub commands: Vec<Command>,
}

pub struct Command {
    pub name: String,
    pub command_parts_count: usize,
    pub action: fn(Option<&str>),
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
        namespace: &str,
        command_name: &str,
        action: fn(Option<&str>),
    ) -> Result<Self, CommandParserError> {
        if let Some(ns) = self.namespaces.iter_mut().find(|ns| ns.name == namespace) {
            ns.commands.push(Command {
                command_parts_count: command_name.split_whitespace().count(),
                name: command_name.to_string(),
                action,
            });
            Ok(self)
        } else {
            Err(CommandParserError::NamespaceNotFound(namespace.to_string()))
        }
    }

    /// Builds the `CommandParser`
    pub fn build(self) -> CommandParser {
        CommandParser {
            namespaces: self.namespaces,
        }
    }
}

type CommandAction<'s> = (&'s fn(Option<&str>), Option<String>);

impl CommandParser {
    /// Parses a command input with a namespace prefix and executes the closest matching command
    pub fn parse(&self, input: &str) -> Result<CommandAction, CommandParserError> {
        // Split input into namespace and rest of the command
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(CommandParserError::InvalidFormat);
        }
        let namespace_name = parts[0];

        // Find the namespace
        if let Some(ns) = self.namespaces.iter().find(|ns| ns.name == namespace_name) {
            // Find the closest matching command
            if let Some((command, distance, command_text)) = ns
                .commands
                .iter()
                .map(|command| {
                    // Defined commands will have multiple expected tokens to match hence
                    // check for the first n tokens of a given command
                    let command_text = parts
                        .get(1..command.command_parts_count + 1)
                        .map(|p| p.join(" "))
                        .unwrap_or(String::new());

                    (
                        command,
                        levenshtein(&command.name, &command_text),
                        command_text,
                    )
                })
                .min_by_key(|cmd| cmd.1)
            {
                if distance <= ns.threshold {
                    let args = parts
                        .get(command.command_parts_count..)
                        .map(|s| s.join(" "));

                    Ok((&command.action, args))
                } else {
                    Err(CommandParserError::NoCloseMatches(command_text.to_string()))
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

    fn dummy_action(_params: Option<&str>) {
        // Dummy action for testing
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
            .register_command("test", "dummy command", dummy_action)
            .unwrap()
            .build();

        assert_eq!(parser.namespaces[0].commands.len(), 1);
        assert_eq!(parser.namespaces[0].commands[0].name, "dummy command");
    }

    #[test]
    fn test_register_command_invalid_namespace() {
        let result =
            CommandParserBuilder::new().register_command("invalid", "dummy command", dummy_action);

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
            .register_command("test", "dummy command", dummy_action)
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
            .register_command("test", "dummy command", dummy_action)
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
            .register_command("test", "dummy command", dummy_action)
            .unwrap()
            .register_command("move", "backwards command", dummy_action)
            .unwrap()
            .register_command("move2", "backwards command", dummy_action)
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
