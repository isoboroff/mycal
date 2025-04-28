use clap::{Arg, ArgMatches, Command};
use kdam::{tqdm, BarExt};
use log::debug;
use min_max_heap::MinMaxHeap;
use mycal::{Classifier, DocScore, Store};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::vec::Vec;

fn cli() -> Command {
    Command::new("mycal")
        .about("A continuous active learning tool")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            Arg::new("coll")
                .help("The collection prefix")
                .required(true),
        )
        .arg(Arg::new("model").help("The model file").required(true))
        .subcommand(
            Command::new("train")
                .about("Apply the given qrels file as training examples")
                .arg(Arg::new("qrels_file").help("The qrels file"))
                .arg(
                    Arg::new("negatives")
                        .short('n')
                        .long("negatives")
                        .value_parser(clap::value_parser!(usize))
                        .default_value("0")
                        .help("Add n randomly-sampled documents as nonrelevant."),
                )
                .arg(
                    Arg::new("level")
                        .short('l')
                        .long("level")
                        .value_parser(clap::value_parser!(i32))
                        .default_value("1")
                        .help("Minimum relevance level in the qrels to count as relevant."),
                ),
        )
        .subcommand(
            Command::new("score")
                .about("Score the collection and return the top n documents")
                .arg(
                    Arg::new("num_scores")
                        .short('n')
                        .long("num_scores")
                        .value_parser(clap::value_parser!(usize))
                        .default_value("100")
                        .help("Number of top-scoring documents to retrieve"),
                )
                .arg(
                    Arg::new("exclude")
                        .short('e')
                        .long("exclude")
                        .help("Qrels file of documents to exclude"),
                ),
        )
        .subcommand(
            Command::new("score_one")
                .about("Score one document, by docid")
                .arg(
                    Arg::new("docid")
                        .help("A document identifier")
                        .required(true),
                ),
        )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = cli().get_matches();
    let coll_prefix = args.get_one::<String>("coll").unwrap();
    let model_file = args.get_one::<String>("model").unwrap();

    env_logger::init();

    match args.subcommand() {
        Some(("train", qrels_args)) => {
            train_qrels(coll_prefix, model_file, qrels_args)?;
        }
        Some(("score", score_args)) => {
            score_collection(coll_prefix, model_file, score_args)?;
        }
        Some(("score_one", score_one_args)) => {
            score_one_doc(coll_prefix, model_file, score_one_args)?;
        }
        Some((&_, _)) => panic!("No subcommand specified"),
        None => panic!("No subcommand specified"),
    }
    Ok(())
}

fn train_qrels(
    coll_prefix: &str,
    model_file: &str,
    qrels_args: &ArgMatches,
) -> Result<Classifier, std::io::Error> {
    let mut coll = Store::open(coll_prefix)?;

    let model_path = Path::new(model_file);
    let mut model: Classifier;
    if model_path.exists() {
        debug!("Loading model from {}", model_file);
        model = Classifier::load(model_file).unwrap();
    } else {
        let num_toks = coll.num_features().unwrap();
        debug!("Creating new model of dim {}", num_toks);
        model = Classifier::new(num_toks, 200000);
    }

    let qrels_file = qrels_args.get_one::<String>("qrels_file").unwrap();
    let qrels = BufReader::new(File::open(qrels_file).expect("Could not open qrels file"));

    let mut pos = Vec::new();
    let mut neg = Vec::new();
    let mut using = HashSet::new();

    /*
    Read a qrels-formatted file specifying the training documents.
    Get each document's feature vector and add it to the appropriate list (pos or neg)
    */
    debug!("Getting examples from qrels file");
    qrels
        .lines()
        .filter(|result| !result.as_ref().unwrap().starts_with('#'))
        .map(|line| {
            let line = line.unwrap();
            line.split_whitespace().map(|x| x.to_string()).collect()
        })
        .for_each(|fields: Vec<String>| {
            if let Ok(intid) = coll.get_doc_intid(&fields[2]) {
                using.insert(fields[2].clone());
                if let Ok(mut fv) = coll.get_fv(intid) {
                    if fv.squared_norm == 0.0 {
                        fv.compute_norm();
                    }
                    let rel = i32::from_str(&fields[3]).unwrap();
                    let min = qrels_args.get_one::<i32>("level").unwrap();

                    if rel < *min {
                        neg.push(fv);
                        println!("qrels-neg {} {}", fields[2], fields[3]);
                    } else {
                        pos.push(fv);
                        println!("qrels-pos {} {}", fields[2], fields[3]);
                    };
                } else {
                    println!("Error reading fv for {}", fields[2]);
                }
            } else {
                println!("Document not found: {}", fields[2]);
            }
        });

    // If requested, add num_neg more negative examples to neg
    let num_neg = qrels_args.get_one::<usize>("negatives").unwrap();
    if *num_neg > 0 {
        let docvec = coll.docids.as_ref().unwrap().get_values();
        let mut rng = rand::rng();

        docvec
            .choose_multiple(&mut rng, *num_neg)
            .for_each(|intid| {
                let fv = coll.get_fv(*intid).unwrap();
                neg.push(fv);
            });
    }

    model.train(&pos, &neg);
    model.save(model_file)?;
    Ok(model)
}

fn score_collection(
    coll_prefix: &str,
    model_file: &str,
    score_args: &ArgMatches,
) -> Result<Vec<DocScore>, std::io::Error> {
    let model = Classifier::load(model_file).unwrap();
    println!("Score: classifier dim is {}", model.w.len());
    let n = score_args.get_one::<usize>("num_scores").unwrap();
    let exclude_fn = score_args.get_one::<String>("exclude");

    let mut exclude = HashSet::new();
    match exclude_fn {
        Some(efn) => {
            let exclude_fp = BufReader::new(File::open(efn)?);
            exclude_fp
                .lines()
                .map(|line| line.unwrap().split_whitespace().nth(2).unwrap().to_string())
                .for_each(|d| {
                    exclude.insert(d);
                });
        }
        _ => (),
    }

    let mut coll = Store::open(&coll_prefix)?;

    let mut top_scores: MinMaxHeap<DocScore> = MinMaxHeap::new();

    let num_docs = coll.num_docs().unwrap();
    let mut progress = tqdm!(total = num_docs);

    let feats = coll.get_fv_iter();
    for fv in feats {
        if exclude.contains(&fv.docid) {
            continue;
        }
        let score = model.inner_product(&fv);
        top_scores.push(DocScore {
            docid: fv.docid,
            score: OrderedFloat(score),
        });

        while top_scores.len() > *n {
            top_scores.pop_min();
        }
        progress.update(1)?;
    }

    let top = top_scores.into_vec_desc();
    top.iter()
        .for_each(|ds| println!("{} {}", ds.docid, ds.score));

    Ok(top)
}

fn score_one_doc(
    coll_prefix: &str,
    model_file: &str,
    score_one_args: &ArgMatches,
) -> Result<f32, std::io::Error> {
    let docid = score_one_args.get_one::<String>("docid").unwrap();

    let mut coll = Store::open(coll_prefix)?;
    let model = Classifier::load(model_file).unwrap();

    let intid = coll.get_doc_intid(docid).unwrap();
    let fv = coll.get_fv(intid).unwrap();

    let score = model.inner_product(&fv);
    println!("{:?}", score);
    Ok(score)
}
