use bincode::config::Configuration;
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
    let bincode_config = bincode::config::standard();

    println!("Opening database...");
    let docs = DocsDb::open(&args.coll_prefix);

    docs.ext2int
        .iter()
        .map(|res| res.unwrap())
        .map(|(k, v)| {
            (
                String::from_utf8(k.to_vec()),
                bincode::decode_from_slice::<DocInfo, Configuration>(&v, bincode_config),
            )
        })
        .for_each(|(k, v)| println!("{:?} {:?}", k, v));

    docs.int2ext
        .iter()
        .map(|res| res.unwrap())
        .map(|(k, v)| {
            (
                String::from_utf8(k.to_vec()),
                bincode::decode_from_slice::<DocInfo, Configuration>(&v, bincode_config),
            )
        })
        .for_each(|(k, v)| println!("{:?} {:?}", k, v));

    Ok(())
}
