use clap::Parser;
use kdam::tqdm;
use log::{log_enabled, Level};
use mycal::{
    index::InvertedFile,
    tok::{get_tokenizer, tokenize_and_map},
    utils, Dict, DocsDb,
};
use serde_json::{from_str, Map, Value};
use std::{io::BufRead, path::Path};

fn build_index(args: Cli) {
    let mut dict = Dict::new();
    let docsdb_fname = format!("{}.db", &args.out_prefix);
    let mut docsdb = DocsDb::create(&docsdb_fname);
    let invfile_fname = format!("{}.inv", &args.out_prefix);
    let mut invfile = InvertedFile::new(Path::new(&invfile_fname));
    for bundle in args.bundles {
        let tokenizer = get_tokenizer(args.tokenizer.as_ref().expect("Unknown tokenizer"));
        let reader = utils::reader(&bundle);

        for line in tqdm!(reader.lines()) {
            let docmap = from_str::<Map<String, Value>>(&line.unwrap()).unwrap();
            let (docid, docmap) =
                tokenize_and_map(docmap, &tokenizer, &mut dict, &args.docid, &args.body);
            let intid = docsdb.add_doc(&docid).unwrap();
            assert_ne!(0, intid);
            for (tok, count) in docmap {
                invfile.add_posting(tok, intid, count);
            }
        }
    }
    invfile.save();
    let _ = dict.save(&format!("{}.dict", &args.out_prefix));
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
    build_index(args);
}
