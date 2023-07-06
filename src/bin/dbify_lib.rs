use clap::Parser;
use kdam::{tqdm, BarExt};
use kv::*;
use mycal::{DocInfo, Docs};
use std::fs::File;
use std::io::{BufReader, Result};
use std::path::Path;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let root = Path::new(".");
    let docs_file = root.join(format!("{}.lib", args.coll_prefix));
    let db_file = root.join(format!("{}.lb2", args.coll_prefix));
    println!("Reading lib structure...");
    let docs_fp = BufReader::new(File::open(docs_file)?);
    let mut docs: Docs = bincode::deserialize_from(docs_fp).unwrap();

    println!("Converting to database...");
    let db = Config::new(db_file);
    let store = Store::new(db).unwrap();
    let bucket = store
        .bucket::<String, Bincode<DocInfo>>(Some("docinfo"))
        .unwrap();

    let mut progbar = tqdm!(total = docs.docs.len());

    for (docid, intid) in docs.m.drain() {
        let di = docs.docs.get(intid).unwrap();
        let dib = Bincode(DocInfo {
            intid: di.intid,
            docid: di.docid.clone(),
            offset: di.offset,
        });
        bucket.set(&docid, &dib).expect("Could not insert into db");
        progbar.update(1);
    }

    Ok(())
}
