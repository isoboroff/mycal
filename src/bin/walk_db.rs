use clap::Parser;
use mycal::{DocInfo, DocsDb};
use std::io::Result;
use std::path::Path;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let root = Path::new(".");
    let _dict_file = root.join(format!("{}.dct", args.coll_prefix));
    let docs_file = root.join(format!("{}.lib", args.coll_prefix));

    println!("Opening database...");
    let docs = DocsDb::open(docs_file.to_str().unwrap());
    // let db = Config::new(docs_file);
    // let store = Store::new(db).unwrap();
    // let bucket = store
    //    .bucket::<String, Bincode<DocInfo>>(Some("docinfo"))
    //    .unwrap();

    docs.db
        .iter()
        .map(|res| res.unwrap())
        .map(|(k, v)| {
            (
                String::from_utf8(k.to_vec()),
                bincode::deserialize::<DocInfo>(&v),
            )
        })
        .for_each(|(k, v)| println!("{:?} {:?}", k, v));

    Ok(())
}
