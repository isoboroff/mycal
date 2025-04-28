use clap::Parser;
use mycal::{FeatureVec, Store};
use std::io::Result;

#[derive(Parser)]
#[command(name = "get_doc")]
#[command(about = "Fetch a feature vector given a docid, serialized hashmaps version.")]
struct Cli {
    coll_prefix: String,
    docid: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut coll = Store::open(&args.coll_prefix)?;

    let intid = coll
        .get_doc_intid(&args.docid)
        .expect(&format!("Docid {} not found", &args.docid));
    let mut fv: FeatureVec = coll
        .get_fv(intid)
        .expect(&format!("Feature vec not found for docid {}", &args.docid));
    println!("Doc {} ({})", args.docid, intid);
    fv.features.sort_by(|fva, fvb| fva.id.cmp(&fvb.id));
    fv.features
        .iter()
        .for_each(|fp| println!("  {} {}", fp.id, fp.value));

    Ok(())
}
