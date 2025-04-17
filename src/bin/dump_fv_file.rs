use clap::Parser;
use mycal::FeatureVec;
use std::{
    fs::File,
    io::{BufReader, Result},
};

#[derive(Parser)]
struct Cli {
    fv_file: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut fv_file = BufReader::new(File::open(&args.fv_file)?);
    loop {
        match FeatureVec::read_from(&mut fv_file) {
            Ok(fv) => {
                println!("fv: {}, {} features", fv.docid, fv.features.len());
            }
            Err(e) => {
                println!("error: {:?}", e);
                break;
            }
        }
    }
    Ok(())
}
