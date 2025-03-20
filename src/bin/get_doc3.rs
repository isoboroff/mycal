use clap::Parser;
use mycal::{DocsDb, FeatureVec};
use std::fs::File;
use std::io::{BufReader, Result, Seek, SeekFrom};
use std::path::Path;

#[derive(Parser)]
#[command(name = "get_doc2")]
#[command(about = "Fetch a feature vector given a docid, sled version.")]
struct Cli {
    coll_prefix: String,
    docid: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let root = Path::new(".");
    let _dict_file = root.join(format!("{}.dct", args.coll_prefix));
    let docs_file = root.join(format!("{}.lib", args.coll_prefix));
    let feat_file = root.join(format!("{}.ftr", args.coll_prefix));
    let bincode_config = bincode::config::standard();

    println!("Opening lb2 database...");
    let docs = DocsDb::open(docs_file.to_str().unwrap());
    // let db = Config::new(docs_file);
    // let store = Store::new(db).unwrap();
    // let bucket = store
    //    .bucket::<String, Bincode<DocInfo>>(Some("docinfo"))
    //    .unwrap();

    println!("Fetching docinfo...");
    let docinfo = match docs.get(&args.docid) {
        Some(di) => di,
        None => panic!("Document {} not found", args.docid),
    };

    println!("Looking up features...");
    let mut feat_fp = BufReader::new(File::open(feat_file)?);
    feat_fp.seek(SeekFrom::Start(docinfo.offset))?;

    let fv: FeatureVec = bincode::decode_from_std_read(&mut feat_fp, bincode_config).unwrap();
    println!("Doc {} ({}): {:?}", args.docid, docinfo.intid, fv);

    Ok(())
}
