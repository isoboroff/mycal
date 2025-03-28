use clap::Parser;
use kdam::{tqdm, BarExt};
use log::{log_enabled, Level};
use mycal::{
    index::InvertedFile,
    tok::{get_tokenizer, tokenize_and_map},
    utils, Dict, DocInfo, DocsDb, FeatureVec,
};
use serde_json::{from_str, Map, Value};
use std::{
    fs::File,
    io::{BufRead, BufWriter, Seek},
    path::Path,
};

// This is an in-memory inversion.  Inverting the Chinese
// NeuCLIR collection, with 3-grams as features
// required 45GB of RAM.
// Consider build_mapred for something more scalable.  Esp
// once it gets an external sort.

fn build_index(args: Cli) -> Result<(), std::io::Error> {
    let mut dict = Dict::new();
    let mut docsdb = DocsDb::create(&args.out_prefix);
    let invfile_fname = format!("{}.inv", &args.out_prefix);
    let mut invfile = InvertedFile::new(Path::new(&invfile_fname));
    let featfile_fname = format!("{}.ftr", &args.out_prefix);
    let mut featfile = BufWriter::new(File::create(&featfile_fname)?);
    let bincode_config = bincode::config::standard();

    println!("First pass: tokenizing, indexing, storing features");
    let num_docs: u32 = 0;
    for bundle in args.bundles {
        let tokenizer = get_tokenizer(args.tokenizer.as_ref().expect("Unknown tokenizer"));
        let mut reader = utils::reader(&bundle);

        let num_lines = reader.lines().count();
        let mut bar = tqdm!(total = num_lines);
        reader = utils::reader(&bundle);

        for line in reader.lines() {
            bar.update(1)?;
            let docmap = from_str::<Map<String, Value>>(&line.unwrap()).unwrap();
            let (docid, docmap) =
                tokenize_and_map(docmap, &tokenizer, &mut dict, &args.docid, &args.body);
            let intid = docsdb.add_doc(&docid).unwrap();
            assert_ne!(0, intid);
            let mut fv = FeatureVec::new(docid.clone());
            for (tok, count) in docmap {
                invfile.add_posting(tok, intid, count);
                let df = count as f32;
                fv.push(tok, df);
            }
            let offset = featfile.stream_position().unwrap();
            bincode::encode_into_std_write(fv, &mut featfile, bincode_config)
                .expect("Error writing feature vec");
            let di = DocInfo {
                intid,
                docid: docid.clone(),
                offset,
            };
            docsdb.insert_batch(&docid, &di, 100_000);
            docsdb.insert_batch(format!("{}", intid).as_str(), &di, 100_000);
        }
    }
    invfile.save();
    docsdb.process_remaining();
    let _ = docsdb.ext2int.flush();

    println!("Second pass: precompute IDF");
    let mut new_dict = Dict::new();
    dict.m.drain().for_each(|(_tok, tokid)| {
        if let Some(df) = dict.df.get(&tokid) {
            new_dict.df.insert(tokid, (num_docs as f32 / df).log10());
        }
    });
    new_dict
        .save(&format!("{}.dct", &args.out_prefix))
        .expect("Problem writing dict");
    Ok(())
}

#[derive(Parser)]
struct Cli {
    /// The prefix for on-disk structures
    out_prefix: String,
    /// The path to a file of documents, formatted as JSON lines
    bundles: Vec<String>,
    /// Document ID field
    #[arg(short, long, value_name = "FIELD", default_value = "doc_id")]
    docid: String,
    /// Document body text field
    #[arg(short, long, value_name = "FIELD", default_value = "text")]
    body: String,
    #[arg(short = 't', default_value = "englishstemlower")]
    tokenizer: Option<String>,
}
fn main() {
    let args = Cli::parse();
    if log_enabled!(Level::Debug) {
        env_logger::init();
    }
    build_index(args).expect("Problem building index");
}
