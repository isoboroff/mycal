use clap::Parser;
use mycal::odch::OnDiskCompressedHash;

#[derive(Parser)]
struct Cli {
    odch_file: String,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    let odch = OnDiskCompressedHash::open(&args.odch_file).unwrap();
    for (k, v) in odch.map() {
        println!("{} {}", k, v);
    }
    Ok(())
}
