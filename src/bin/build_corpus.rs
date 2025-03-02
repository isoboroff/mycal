use clap::Parser;
use flate2::read;
use kdam::{tqdm, Bar, BarExt};
use mycal::{tokenize, Dict, Docs, DocsDb, FeatureVec};
use serde_json::{from_str, Map, Value};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{remove_file, File};
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter, Result, Seek};
use std::path::Path;

#[derive(Parser)]
struct Cli {
    /// The prefix for on-disk structures
    out_prefix: String,
    /// The path to a file of documents, formatted as JSON lines
    bundles: Vec<String>,
    /// Document ID field
    #[arg(short = 'd', default_value = "doc_id")]
    docid: String,
    /// Document body text field
    #[arg(short = 'b', default_value = "text")]
    body: String,
}

/// Read normal or compressed files seamlessly
/// Uses the presence of a `.gz` extension to decide
/// from https://users.rust-lang.org/t/write-to-normal-or-gzip-file-transparently/35561/2
pub fn reader(filename: &str) -> Box<dyn BufRead> {
    let path = Path::new(filename);
    let file = match File::open(path) {
        Err(why) => panic!("couldn't open {}: {:?}", path.display(), why),
        Ok(file) => file,
    };

    if path.extension() == Some(OsStr::new("gz")) {
        Box::new(BufReader::with_capacity(
            128 * 1024,
            read::GzDecoder::new(file),
        ))
    } else {
        Box::new(BufReader::with_capacity(128 * 1024, file))
    }
}

fn tokenize_and_map(
    docmap: serde_json::Map<String, serde_json::Value>,
    dict: &mut Dict,
    docid_field: &String,
    text_field: &String,
) -> (String, HashMap<usize, i32>) {
    let mut m = HashMap::new();
    let docid = match docmap.contains_key(docid_field) {
        true => docmap[docid_field].as_str().unwrap(),
        false => panic!(
            "Document does not contain a {} field for the docid (use -d option?)",
            docid_field
        ),
    };
    let tokens = match docmap.contains_key(text_field) {
        true => tokenize(docmap[text_field].as_str().unwrap()),
        false => panic!(
            "Document does not contain a {} field for the text (use -t option?)",
            text_field
        ),
    };

    for x in tokens {
        let tokid = dict.add_tok(x.to_owned());
        if !m.contains_key(&tokid) {
            dict.incr_df(tokid);
        }
        *m.entry(tokid).or_insert(0) += 1;
    }

    (docid.to_owned(), m)
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // First pass: collect dictionary, df counts
    println!("First pass, collect dictionary and docfeqs");
    let mut dict: Dict = Dict::new();
    let mut library = Docs::new();

    let mut num_docs = 0;
    let mut binout = BufWriter::new(File::create(args.out_prefix.clone() + ".tmp")?);

    for bundle in args.bundles {
        let path = Path::new(&bundle);
        let desc = path.file_name().unwrap().to_str().unwrap();
        let mut progress = tqdm!();

        progress.set_description(desc);

        let reader = reader(&bundle);

        reader
            .lines()
            .map(|line| from_str::<Map<String, Value>>(&line.unwrap()).expect("Error parsing JSON"))
            .map(|docmap| tokenize_and_map(docmap, &mut dict, &args.docid, &args.body))
            .map(|(docid, docmap)| {
                let mut fv = FeatureVec::new(docid.clone());
                for (tok, count) in docmap {
                    fv.push(tok, count as f32);
                }
                library.add_doc(&docid);
                fv
            })
            .for_each(|fv| {
                let _ = progress.update(1);
                num_docs += 1;
                bincode::serialize_into(&mut binout, &fv).expect("Error writing to bin file");
            });

        binout.flush()?;
        let _ = progress.refresh();
    }

    // Compute IDF, drop singleton terms
    println!("Compute IDFs and prune dictionary");
    let mut new_dict = Dict::new();
    let mut old_to_new = HashMap::new();

    dict.m.drain().for_each(|(tok, tokid)| {
        if let Some(df) = dict.df.get(&tokid) {
            if *df > 1.0 {
                let new_tokid = new_dict.add_tok(tok);
                old_to_new.insert(tokid, new_tokid);
                new_dict
                    .df
                    .insert(new_tokid, (num_docs as f32 / df).log10());
            }
        }
    });

    println!(
        "Library len {} cap {}",
        library.docs.len(),
        library.docs.capacity()
    );

    // Reassign token IDs and precompute tfidf weights
    println!("Second pass: precompute weights and fix up tokenids");
    let mut progress = Bar::new(num_docs);
    let mut intid = 0;
    let mut binin = BufReader::new(File::open(args.out_prefix.clone() + ".tmp")?);
    binout = BufWriter::new(File::create(args.out_prefix.clone() + ".ftr")?);
    let libdb_fn = args.out_prefix.to_string() + ".lib";
    let mut lib = DocsDb::create(&libdb_fn);

    while let Ok(fv) = FeatureVec::read_from(&mut binin) {
        let mut new_fv = FeatureVec::new(fv.docid.clone());
        for f in &fv.features {
            if let Some(new_tokid) = old_to_new.get(&f.id) {
                let df = new_dict.df.get(new_tokid).unwrap();
                new_fv.push(*new_tokid, (1.0 + f.value.log10()) * df);
            }
        }
        new_fv.compute_norm();
        if intid >= library.docs.len() {
            println!("oh shit: {}", intid);
        }
        library.docs[intid].offset = binout.stream_position().unwrap();
        bincode::serialize_into(&mut binout, &new_fv).expect("Error writing to final bin file");
        binout.flush()?;

        lib.insert_batch(&library.docs[intid].docid, &library.docs[intid], 100_000);

        intid += 1;
        let _ = progress.update(1);
    }
    binout.flush()?;
    remove_file(args.out_prefix.to_string() + ".tmp")?;

    // let libdb_fn = args.out_prefix.to_string() + ".lib";
    // let mut lib = DocsDb::create(&libdb_fn);
    // progress = Bar::new(library.m.len());
    // for (docid, intid) in library.m.drain() {
    //     let di = library.docs.get(intid).unwrap();
    //     lib.insert_batch(&docid, &di, 100_000);
    //     progress.update(1);
    // }
    // lib.process_remaining();

    new_dict.save(&(args.out_prefix + ".dct"))?;

    Ok(())
}
