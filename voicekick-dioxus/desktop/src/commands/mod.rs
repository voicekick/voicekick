use command_parser::{
    CommandAction, CommandArgs, CommandParser, CommandParserBuilder, CommandResult,
};

struct DummyCommand;

impl CommandAction for DummyCommand {
    fn execute(&self, args: CommandArgs) -> CommandResult {
        println!("Command: {:?}", args);
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
