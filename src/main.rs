use clap::{Arg, ArgMatches, Command};
use kdam::{tqdm, BarExt, RowManager};
use min_max_heap::MinMaxHeap;
use mycal::{Classifier, Dict, DocInfo, DocsDb, FeatureVec};
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
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
            Command::new("qrels")
                .about("Apply the given qrels file as training examples")
                .arg(Arg::new("qrels_file").help("The qrels file")),
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
        Some(("qrels", qrels_args)) => {
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
        model = Classifier::new(dict.m.len(), 100);
    }

    let docs = DocsDb::open(&docsdb_file);
    let mut feats = BufReader::new(File::open(feat_file).expect("Could not open feature file"));

    let qrels_file = qrels_args.get_one::<String>("qrels_file").unwrap();

    let qrels = BufReader::new(File::open(qrels_file).expect("Could not open qrels file"));
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    qrels
        .lines()
        .map(|line| {
            let line = line.unwrap();
            line.split_whitespace().map(|x| x.to_string()).collect()
        })
        .for_each(|fields: Vec<String>| {
            println!("{:?}", fields);
            let dib = docs.db.get(&fields[2]).unwrap().unwrap();
            let di: DocInfo = bincode::deserialize(&dib).unwrap();
            feats
                .seek(SeekFrom::Start(di.offset))
                .expect("Seek error in feats");
            let fv = FeatureVec::read_from(&mut feats).expect("Error reading feature vector");
            let rel = i32::from_str(&fields[3]).unwrap();
            if rel <= 0 {
                neg.push(fv);
            } else {
                pos.push(fv);
            };
        });

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

fn score_inner(
    feat_file: String,
    model: &Arc<Classifier>,
    cut: u64,
    next_cut: u64,
    num_scores: usize,
    //kdam_mgr: &mut MutexGuard<RowManager>,
) -> MinMaxHeap<DocScore> {
    let mut feats = BufReader::new(File::open(feat_file).expect("Can't open feature file"));
    let mut top_scores = MinMaxHeap::new();
    let total = (next_cut - cut) as usize;

    feats
        .seek(SeekFrom::Start(cut))
        .expect("Error seeking in feature file");
    while let Ok(fv) = FeatureVec::read_from(&mut feats) {
        if feats.get_ref().stream_position().unwrap() > next_cut {
            break;
        }
        let score = model.inner_product(&fv);
        top_scores.push(DocScore {
            docid: fv.docid,
            score: OrderedFloat(score),
        });

        while top_scores.len() > num_scores {
            top_scores.pop_min();
        }
        //kdam_mgr.get_mut(0).unwrap().update(1);
    }
    top_scores
}

fn score_collection_multithreaded(
    coll_prefix: &str,
    model_file: &str,
    score_args: &ArgMatches,
) -> Result<Vec<DocScore>, std::io::Error> {
    let model = Arc::new(Classifier::load(model_file).unwrap());
    let n = score_args.get_one::<usize>("num_scores").unwrap();

    let feat_file = coll_prefix.to_string() + ".ftr";
    let cuts_file = coll_prefix.to_string() + ".cut";

    let cuts = if Path::new(&cuts_file).exists() {
        let cuts_fp = BufReader::new(File::open(cuts_file)?);
        cuts_fp
            .lines()
            .map(|s| str::parse(&s.unwrap()).unwrap())
            .collect()
    } else {
        vec![0, 0]
    };
    let cuts_pairs = cuts.iter().zip(cuts.iter().skip(1));

    // let mut pbs = RowManager::new(cuts_pairs.len() as u16);
    // for (i, (cut, next_cut)) in cuts_pairs.clone().enumerate() {
    //     pbs.append(tqdm!(
    //         total = (next_cut - cut) as usize,
    //         leave = false,
    //         desc = format!("Thread {}", i),
    //         force_refresh = true
    //     ));
    // }
    // let pbs: Arc<Mutex<RowManager>> = Arc::new(Mutex::new(pbs));

    let mut handles: Vec<JoinHandle<MinMaxHeap<DocScore>>> = vec![];
    for (&cut, &next_cut) in cuts_pairs {
        // let pbs1 = pbs.clone();
        let model = model.clone();
        let n = *n;
        let feat_file = feat_file.clone();
        let handle = thread::spawn(move || {
            // let mut pbs = pbs1.lock().unwrap();
            println!("Forking {}", cut);
            score_inner(feat_file, &model, cut, next_cut, n) //, &mut pbs)
        });
        handles.push(handle);
    }

    let mut top_scores = vec![];
    for heap in handles {
        top_scores.extend_from_slice(&heap.join().unwrap().into_vec_desc());
    }
    top_scores.sort();
    top_scores.truncate(*n);
    println!("{:?}", top_scores);

    Ok(top_scores)
}

fn score_collection(
    coll_prefix: &str,
    model_file: &str,
    score_args: &ArgMatches,
) -> Result<Vec<DocScore>, std::io::Error> {
    let model = Classifier::load(model_file).unwrap();
    let n = score_args.get_one::<usize>("num_scores").unwrap();

    let feat_file = coll_prefix.to_string() + ".ftr";

    let mut top_scores: MinMaxHeap<DocScore> = MinMaxHeap::new();

    let mut feats = BufReader::new(File::open(feat_file)?);
    let mut progress = tqdm!();

    while let Ok(fv) = FeatureVec::read_from(&mut feats) {
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
    println!("{:?}", &top);

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
