use bincode::encode_into_std_write;
use clap::Parser;
use mycal::odch::OnDiskCompressedHash;
use std::{fs::File, io::BufWriter};

#[derive(Parser)]
struct Cli {
    odch_file: String,
    output_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    let odch = OnDiskCompressedHash::open(&args.odch_file).unwrap();
    let output_file = File::create(&args.output_file)?;
    let mut writer = BufWriter::new(output_file);
    let _ = encode_into_std_write(odch.map(), &mut writer, bincode::config::standard()).unwrap();
    Ok(())
}
