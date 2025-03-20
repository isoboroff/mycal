use clap::{Arg, Command};
use kdam::tqdm;
use std::io::prelude::*;
use std::{error::Error, fs::File, io::BufWriter};

use mycal::DocsDb;

fn cli() -> Command {
    Command::new("find-ftr-splits")
        .about("Find even divisions in the feature file")
        .arg_required_else_help(true)
        .arg(
            Arg::new("coll")
                .help("The collection prefix")
                .required(true),
        )
        .arg(
            Arg::new("num_splits")
                .short('n')
                .long("num_splits")
                .value_parser(clap::value_parser!(usize))
                .default_value("10")
                .help("Number of split points to identify"),
        )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = cli().get_matches();
    let coll_prefix = args.get_one::<String>("coll").unwrap();
    let num_splits = args.get_one::<usize>("num_splits").unwrap();

    let docsdb_file = coll_prefix.to_string() + ".lib";
    let docsdb = DocsDb::open(&docsdb_file);
    let bincode_config = bincode::config::standard();

    let offsets: Vec<u64> = tqdm!(docsdb.db.iter())
        .map(|r| r.unwrap().1)
        .map(|v| bincode::decode_from_slice(&v, bincode_config).unwrap().0)
        .collect();
    let step = offsets.len() / num_splits;

    let out_file = coll_prefix.to_string() + ".cut";
    let mut out = BufWriter::new(File::create(out_file)?);
    let mut cur: usize = 0;
    for _ in 0..*num_splits {
        writeln!(out, "{:}", offsets[cur])?;
        cur += step;
    }
    writeln!(out, "{:}", offsets.last().unwrap())?;
    Ok(())
}
