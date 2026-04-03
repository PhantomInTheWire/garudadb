mod cli;
mod command;
mod parsing;

use clap::Parser;
use cli::{Cli, Command};
use command::run_command;
use garuda_engine::Database;

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let root = cli.root.clone();

    match cli.command {
        Command::Init => {
            Database::open(&root).map_err(|status| status.message)?;
            println!("{}", root.display());
            Ok(())
        }
        Command::Run(command) => {
            let db = Database::open(&root).map_err(|status| status.message)?;
            run_command(&db, command)
        }
    }
}
