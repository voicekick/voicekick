use command_parser::{async_trait, CommandAction, CommandArgs, CommandOutput, CommandResult};

/// Vocie log command
#[derive(Default)]
pub struct VoiceLogCommand;

#[async_trait]
impl CommandAction for VoiceLogCommand {
    async fn execute(&self, args: CommandArgs) -> CommandResult {
        use std::fs::OpenOptions;
        use std::io::prelude::*;

        let file = "voice_log.tmp";

        println!("Command: {:?} write to file {file}", args);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file)
            .expect("Couldn't open file {file}");

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

        Ok(CommandOutput::Ok(None))
    }
}
