use bincode::decode_from_std_read;
use clap::Parser;
use mycal::FeatureVec;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Seek, SeekFrom},
};

#[derive(Parser)]
struct Cli {
    offsets_file: String,
    fv_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    let mut offsets_file = BufReader::new(File::open(&args.offsets_file)?);
    let mut fv_file = BufReader::new(File::open(&args.fv_file)?);
    let offsets: HashMap<String, u64> =
        decode_from_std_read(&mut offsets_file, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    for (docid, offset) in offsets.iter() {
        fv_file.seek(SeekFrom::Start(*offset))?;
        let fv = FeatureVec::read_from(&mut fv_file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        println!("fv: {}, {} features", docid, fv.features.len());
    }

    Ok(())
}
