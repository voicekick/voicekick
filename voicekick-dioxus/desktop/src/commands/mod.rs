use std::io::prelude::*;
use std::{fs::OpenOptions, sync::Arc};

use command_parser::{CommandAction, CommandArgs, CommandParser, CommandResult};

struct DummyCommand;

impl CommandAction for DummyCommand {
    fn execute(&self, args: CommandArgs) -> CommandResult {
        println!("Command: {:?}", args);

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open("dummy_command_file.tmp")
            .unwrap();

        match args {
            CommandArgs::Some(a) => {
                if let Err(e) = writeln!(file, "{}", a) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
            CommandArgs::None => {
                if let Err(e) = writeln!(file, "N/A") {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
            _ => unimplemented!(),
        }

        CommandResult::Ok(None)
    }
}

fn dummy_action() -> Arc<DummyCommand> {
    Arc::new(DummyCommand {})
}

pub fn all() -> CommandParser {
    let parser = CommandParser::new();

    parser
        .register_namespace("test", Some(1))
        .unwrap()
        .register_namespace("new", Some(1))
        .unwrap()
        .register_command("test", "me", dummy_action())
        .unwrap()
        .register_command("new", "nothign", dummy_action())
        .unwrap()
        .register_command("new", "not a thing", dummy_action())
        .unwrap();

    parser
}
