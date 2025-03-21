use clap::Parser;
use mycal::{Docs, FeatureVec};
use std::fs::File;
use std::io::{BufReader, Result, Seek, SeekFrom};
use std::path::Path;

#[derive(Parser)]
#[command(name = "get_doc")]
#[command(about = "Fetch a feature vector given a docid, serialized hashmaps version.")]
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

    println!("Reading lib structure...");
    let mut docs_fp = BufReader::new(File::open(docs_file)?);
    let docs: Docs = bincode::decode_from_std_read(&mut docs_fp, bincode_config).unwrap();

    let intid = match docs.get_intid(&args.docid) {
        Some(i) => i,
        None => panic!("Docid {} not found", args.docid),
    };
    let docinfo = match docs.docs.get(*intid as usize) {
        Some(di) => di,
        None => panic!("Document {} not found", intid),
    };

    println!("Looking up features...");
    let mut feat_fp = BufReader::new(File::open(feat_file)?);
    feat_fp.seek(SeekFrom::Start(docinfo.offset))?;

    let fv: FeatureVec = bincode::decode_from_std_read(&mut feat_fp, bincode_config).unwrap();
    println!("Doc {} ({}): {:?}", args.docid, intid, fv);

    Ok(())
}
