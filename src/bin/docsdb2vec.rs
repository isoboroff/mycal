use std::{error::Error, fs::File, io::{BufWriter, Write}};

use clap::{Arg, Command};
use kdam::TqdmIterator;
use mycal::{DocsDb, DocInfo};

fn cli() -> Command {
    Command::new("docsdb2vec")
        .about("Create a serialized vector from the docsdb")
        .arg_required_else_help(true)
        .arg(
            Arg::new("coll")
                .help("The collection prefix")
                .required(true),
        )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = cli().get_matches();
    let coll_prefix = args.get_one::<String>("coll").unwrap();
    let docsdb_file = coll_prefix.to_string() + ".lib";
    let docvec_file = coll_prefix.to_string() + ".dvc";

    let docs = DocsDb::open(&docsdb_file);
    let mut divec = vec![];

    docs.db
        .iter().tqdm()
        .map(|res| res.unwrap())
        .for_each(|(_k, v)| {
            divec.push(bincode::deserialize::<DocInfo>(&v).unwrap());
        });


    let mut vecfile = BufWriter::new(File::create(docvec_file)?);
    bincode::serialize_into(&mut vecfile, &divec).expect("Error writing DI vector");
    vecfile.flush()?;

    Ok(())
}
