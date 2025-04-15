use bincode::{Decode, Encode};
use clap::Parser;
use log::{log_enabled, Level};
use std::{fs::File, io::BufReader};

// Can't figure out how to just use this from build_mapred.rs
#[derive(Clone, Encode, Decode, PartialEq, Eq, Ord, PartialOrd, Debug)]
pub struct PTuple {
    tok: usize,
    docid: String,
    count: u32,
}

fn dump_pt_file(filename: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut fp = BufReader::new(File::open(filename)?);
    let bincode_conf = bincode::config::standard();
    let mut pt: PTuple = bincode::decode_from_std_read(&mut fp, bincode_conf)?;
    loop {
        println!("pt {:?}", pt);
        pt = match bincode::decode_from_std_read(&mut fp, bincode_conf) {
            Ok(pt) => pt,
            Err(_e) => break,
        }
    }
    Ok(())
}

#[derive(Parser)]
struct Cli {
    filename: String,
}

fn main() {
    let args = Cli::parse();
    if log_enabled!(Level::Debug) {
        env_logger::init();
    }
    dump_pt_file(args.filename).expect("Problem dumping pt file");
}
