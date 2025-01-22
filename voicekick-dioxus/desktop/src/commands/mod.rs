use std::fs::OpenOptions;
use std::io::prelude::*;

use command_parser::{
    CommandAction, CommandArgs, CommandParser, CommandParserBuilder, CommandResult,
};

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

fn dummy_action() -> Box<DummyCommand> {
    Box::new(DummyCommand {})
}

pub fn all() -> CommandParser {
    let parser = CommandParserBuilder::new()
        .register_namespace("test", Some(1)) // Toleration for one character difference
        .register_command("test", "me", dummy_action())
        .expect("TODO: fix")
        .build();

    parser
}
