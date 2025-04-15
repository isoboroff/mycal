use clap::Parser;
use mycal::index::InvertedFile;
use mycal::{Classifier, DocScore, DocsDb, FeaturePair};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::io::Result;
use std::path::Path;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
    model_file: String,
}

struct Score {
    docid: u32,
    score: f32,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let root = Path::new(".");
    let docsdb_filename = format!("{}/{}.lib", root.display(), args.coll_prefix);
    let docsdb = DocsDb::open(&docsdb_filename);
    let inv_filename = format!("{}/{}.inv", root.display(), args.coll_prefix);
    let bincode_config = bincode::config::standard();

    // Convert the model into a vector of FeaturePairs.
    // The weight vector is in tokid order.
    let model = Classifier::load(&args.model_file, bincode_config).unwrap();
    let mut model_query = model
        .w
        .iter()
        .enumerate()
        .map(|(i, w)| FeaturePair {
            id: i as usize,
            value: *w,
        })
        .collect::<Vec<FeaturePair>>();
    // Run through the "query" in decreasing feature score order.
    // Later on we can try to stop early if we have to.
    model_query.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

    let mut inv_file = InvertedFile::open(&inv_filename).unwrap();
    let mut results: HashMap<u32, f32> = HashMap::new();

    // accumulate scores
    for wt in model_query {
        let pl = inv_file.get_posting_list(wt.id).unwrap();
        for p in pl.postings {
            let score = results.entry(p.doc_id).or_insert(0.0);
            *score += wt.value * p.tf as f32;
        }
    }

    let mut rvec = results
        .into_iter()
        .map(|(k, v)| Score {
            docid: k,
            score: v * model.scale,
        })
        .map(|s| {
            let di = docsdb.get_intid(s.docid).unwrap();
            DocScore {
                docid: di.docid,
                score: OrderedFloat::from(s.score),
            }
        })
        .collect::<Vec<DocScore>>();
    rvec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for r in rvec {
        println!("{} {}", r.docid, r.score);
    }

    Ok(())
}
