use clap::Parser;
use kdam::{tqdm, BarExt};
use mycal::{Classifier, DocScore, FeaturePair, Store};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::io::Result;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
    model_file: String,
    #[arg(short, long, default_value_t = 100)]
    num_results: usize,
}

struct Score {
    docid: u32,
    score: f32,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut coll = Store::open(&args.coll_prefix)?;
    let bincode_config = bincode::config::standard();

    // Convert the model into a vector of FeaturePairs.
    // The weight vector is in tokid order.
    let model = Classifier::load(&args.model_file, bincode_config)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;
    let mut model_query = model
        .w
        .iter()
        .enumerate()
        .filter(|w| *w.1 != 0.0)
        .map(|(i, w)| FeaturePair {
            id: i + 1 as usize,
            value: *w,
        })
        .collect::<Vec<FeaturePair>>();

    // Run through the "query" in decreasing feature score order.
    // Later on we can try to stop early if we have to.
    model_query.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());

    let mut results: HashMap<u32, f32> = HashMap::new();

    // accumulate scores
    let mut bar = tqdm!(desc = "Scoring", total = model_query.len());
    for wt in model_query {
        bar.update(1)?;
        if let Ok(pl) = coll.get_posting_list(wt.id) {
            for p in pl.postings {
                let score = results.entry(p.doc_id).or_insert(0.0);
                *score += wt.value * p.tf as f32;
            }
        }
    }

    let mut rvec = results
        .into_iter()
        .map(|(k, v)| Score {
            docid: k,
            score: v * model.scale,
        })
        .map(|s| {
            let di = coll.get_docid(s.docid as usize).unwrap();
            DocScore {
                docid: di,
                score: OrderedFloat::from(s.score),
            }
        })
        .collect::<Vec<DocScore>>();
    rvec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for r in rvec.iter().take(args.num_results) {
        println!("{} {}", r.docid, r.score);
    }

    Ok(())
}
