use clap::Parser;
use mycal::Classifier;
use std::io::Result;

#[derive(Parser)]
struct Cli {
    model: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let model = Classifier::load(&args.model).unwrap();

    println!("sparse model");
    for (i, f) in model.w.iter().enumerate() {
        if *f != 0.0 {
            println!("{}: {}", i, *f);
        }
    }

    println!("lambda: {}", model.lambda);
    println!("scale: {}", model.scale);
    println!("norm: {}", model.squared_norm);

    Ok(())
}
