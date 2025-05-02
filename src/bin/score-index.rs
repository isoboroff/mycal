use clap::Parser;
use mycal::{index, Classifier, Store};
use std::fs::File;
use std::io::{BufRead, BufReader, Result};

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
    model_file: String,
    #[arg(short, long, default_value_t = 100)]
    num_results: usize,
    #[arg(short, long)]
    exclude: Option<String>,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut coll = Store::open(&args.coll_prefix)?;

    let exclude_fn = args.exclude;

    let mut exclude = Vec::new();
    match exclude_fn {
        Some(efn) => {
            let exclude_fp = BufReader::new(File::open(efn)?);
            exclude_fp
                .lines()
                .map(|line| line.unwrap().split_whitespace().nth(2).unwrap().to_string())
                .for_each(|docid| {
                    exclude.push(docid);
                });
        }
        _ => (),
    }

    // Convert the model into a vector of FeaturePairs.
    // The weight vector is in tokid order.
    let model = Classifier::load(&args.model_file)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;

    let config = index::IndexSearchConfig::new()
        .with_num_results(args.num_results)
        .with_exclude_docs(exclude);
    let rvec = index::score_using_index(&mut coll, model, config).unwrap();

    for r in rvec.iter().take(args.num_results) {
        println!("{} {}", r.docid, r.score);
    }

    Ok(())
}
