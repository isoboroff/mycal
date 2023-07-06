use clap::{Arg, ArgMatches, Command};
use mycal::{Classifier, Dict, DocsDb, FeatureVec};
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
            Command::new("qrels")
                .about("Apply the given qrels file as training examples")
                .arg(Arg::new("qrels_file").help("The qrels file")),
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
        model = Classifier::new(dict.m.len(), 1.0, 100);
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
            let di = docs.db.get(&fields[2].to_string()).unwrap().unwrap();
            feats
                .seek(SeekFrom::Start(di.0.offset))
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
