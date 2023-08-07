use clap::{Arg, ArgMatches, Command};
use kdam::{tqdm, BarExt};
use min_max_heap::MinMaxHeap;
use mycal::{Classifier, Dict, DocInfo, DocsDb, FeatureVec};
use ordered_float::OrderedFloat;
use rand::distributions::Uniform;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
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
    let docsdb_file = coll_prefix.to_string() + ".lib";
    let dict_file = coll_prefix.to_string() + ".dct";
    let feat_file = coll_prefix.to_string() + ".ftr";

    let dict = Dict::load(&dict_file).unwrap();

    let model_path = Path::new(model_file);
    let mut model: Classifier;
    if model_path.exists() {
        model = Classifier::load(model_file).unwrap();
    } else {
        model = Classifier::new(dict.m.len(), 200000);
    }

    let docs = DocsDb::open(&docsdb_file);
    let mut feats = BufReader::new(File::open(feat_file).expect("Could not open feature file"));

    let qrels_file = qrels_args.get_one::<String>("qrels_file").unwrap();

    let qrels = BufReader::new(File::open(qrels_file).expect("Could not open qrels file"));
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    let mut using = HashSet::new();

    qrels
        .lines()
        .filter(|result| !result.as_ref().unwrap().starts_with('#'))
        .map(|line| {
            let line = line.unwrap();
            line.split_whitespace().map(|x| x.to_string()).collect()
        })
        .for_each(|fields: Vec<String>| {
            using.insert(fields[2].clone());
            let dib = docs.db.get(&fields[2]).unwrap().unwrap();
            let di: DocInfo = bincode::deserialize(&dib).unwrap();
            feats
                .seek(SeekFrom::Start(di.offset))
                .expect("Seek error in feats");
            let mut fv = FeatureVec::read_from(&mut feats).expect("Error reading feature vector");
            if fv.squared_norm == 0.0 {
                fv.compute_norm();
            }
            let rel = i32::from_str(&fields[3]).unwrap();
            if rel <= 0 {
                neg.push(fv);
                println!("qrels-neg {} {}", fields[2], fields[3]);
            } else {
                pos.push(fv);
                println!("qrels-pos {} {}", fields[2], fields[3]);
            };
        });

    let num_neg = qrels_args.get_one::<usize>("negatives").unwrap();
    if *num_neg > 0 {
        let docvec_file = coll_prefix.to_string() + ".dvc";
        let docvec_fp = BufReader::new(File::open(docvec_file)?);
        let docvec: Vec<DocInfo> = bincode::deserialize_from(docvec_fp).unwrap();
        let numdocs = docvec.len();
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0, numdocs as usize);

        (&mut rng)
            .sample_iter(uniform)
            .take(*num_neg)
            .map(|mut i| {
                let mut my_mut_rng = rand::thread_rng();
                while using.contains(&docvec[i].docid) {
                    i = (&mut my_mut_rng).sample(uniform);
                }
                using.insert(docvec[i].docid.clone());
                println!("samp-neg {} {}", docvec[i].docid, 0);
                docvec[i].offset
            })
            .for_each(|offset| {
                feats
                    .seek(SeekFrom::Start(offset))
                    .expect("Seek error in feats");
                let fv = FeatureVec::read_from(&mut feats).expect("Error reading feature vector");
                neg.push(fv);
            });
    }

    model.train(&pos, &neg);
    model.save(model_file)?;
    Ok(model)
}

#[derive(Eq, Debug, Clone)]
struct DocScore {
    docid: String,
    score: OrderedFloat<f32>,
}

impl Ord for DocScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score).reverse()
    }
}

impl PartialOrd for DocScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other).reverse())
    }
}

impl PartialEq for DocScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

fn score_collection(
    coll_prefix: &str,
    model_file: &str,
    score_args: &ArgMatches,
) -> Result<Vec<DocScore>, std::io::Error> {
    let model = Classifier::load(model_file).unwrap();
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

    let feat_file = coll_prefix.to_string() + ".ftr";

    let mut top_scores: MinMaxHeap<DocScore> = MinMaxHeap::new();

    let mut feats = BufReader::new(File::open(feat_file)?);
    let mut progress = tqdm!();

    while let Ok(fv) = FeatureVec::read_from(&mut feats) {
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
        progress.update(1);
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

    let docsdb_file = coll_prefix.to_string() + ".lib";
    let feat_file = coll_prefix.to_string() + ".ftr";

    let model = Classifier::load(model_file).unwrap();

    let docs = DocsDb::open(&docsdb_file);
    let mut feats = BufReader::new(File::open(feat_file).expect("Could not open feature file"));

    let dib = docs.db.get(docid).unwrap().unwrap();
    let di: DocInfo = bincode::deserialize(&dib).unwrap();
    feats.seek(SeekFrom::Start(di.offset))?;
    let fv = FeatureVec::read_from(&mut feats).expect("Error deserializing feature vec");

    let score = model.inner_product(&fv);
    println!("{:?}", score);
    Ok(score)
}
