use clap::Parser;
use log::{log_enabled, Level};
use mycal::ptuple::PTuple;
use std::{fs::File, io::BufReader};

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
