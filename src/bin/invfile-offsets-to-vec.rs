use clap::Parser;
use kdam::tqdm;
use mycal::index::PostInfo;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Parser)]
struct Cli {
    offset_file: String,
    output_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();
    let config = bincode::config::standard();

    let mut offsets_file = BufReader::new(File::open(&args.offset_file)?);
    let offsets: HashMap<usize, PostInfo> =
        bincode::decode_from_std_read(&mut offsets_file, config).expect("Failed to decode offsets");
    let mut keys = offsets.keys().map(|x| *x).collect::<Vec<_>>();
    keys.sort_unstable();

    let mut new_offsets = vec![PostInfo::new(0, 0); keys.len()];
    for key in tqdm!(keys.into_iter()) {
        if key >= new_offsets.capacity() {
            new_offsets.resize(key + 1, PostInfo::new(0, 0));
        }
        new_offsets[key] = offsets[&key].clone();
    }

    let mut output_file = BufWriter::new(File::create(&args.output_file)?);
    bincode::encode_into_std_write(&new_offsets, &mut output_file, config)
        .expect("Failed to serialize offsets");

    Ok(())
}
