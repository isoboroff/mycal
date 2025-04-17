use clap::Parser;
use std::{fs::File, io::BufReader};

#[derive(Parser)]
struct Cli {
    odch_file: String,
    output_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    let mut infp = BufReader::new(File::open(&args.odch_file)?);
    let mut outfp = lz4_flex::frame::FrameEncoder::new(File::create(&args.output_file)?);
    std::io::copy(&mut infp, &mut outfp)?;

    Ok(())
}
