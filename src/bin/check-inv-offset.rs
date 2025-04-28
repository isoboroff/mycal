use clap::Parser;
use kdam::tqdm;
use mycal::index::PostInfo;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser)]
struct Cli {
    vec_file: String,
    hash_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();
    let config = bincode::config::standard();
    let mut vec_file = BufReader::new(File::open(&args.vec_file)?);
    let vec_offsets: Vec<PostInfo> =
        bincode::decode_from_std_read(&mut vec_file, config).expect("Failed to decode offsets vec");
    let mut hash_file = BufReader::new(File::open(&args.hash_file)?);
    let hash_offsets: HashMap<usize, PostInfo> =
        bincode::decode_from_std_read(&mut hash_file, config)
            .expect("Failed to decode offsets hash");

    for (k, v) in tqdm!(hash_offsets.iter()) {
        assert_eq!(vec_offsets[*k], *v, "k {}, v {:?}", k, v);
    }

    Ok(())
}
