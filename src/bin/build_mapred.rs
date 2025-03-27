use bincode::{Decode, Encode};
use bytesize::{GB, MB};
use clap::Parser;
use kdam::{tqdm, BarExt};
use log::{log_enabled, Level};
use mycal::{
    external_sort,
    extsort::SerializeDeserialize,
    index::InvertedFile,
    tok::{get_tokenizer, tokenize_and_map},
    utils, Dict, DocInfo, DocsDb, FeatureVec,
};
use serde_json::{from_str, Map, Value};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek, Write},
    iter,
    path::Path,
};

#[derive(Clone, Encode, Decode, PartialEq, Eq, Ord, PartialOrd)]
struct PTuple {
    tok: usize,
    docid: u32,
    count: u32,
}

fn build_index(args: Cli) -> Result<(), std::io::Error> {
    let mut dict = Dict::new();
    let mut docsdb = DocsDb::create(&args.out_prefix);
    let invfile_fname = format!("{}.inv", &args.out_prefix);
    let mut invfile = InvertedFile::new(Path::new(&invfile_fname));
    let featfile_fname = format!("{}.ftr", &args.out_prefix);
    let mut featfile = BufWriter::new(File::create(&featfile_fname)?);
    let bincode_config = bincode::config::standard();

    println!("First pass: tokenizing, indexing, storing features");
    let mut num_docs: u32 = 0;
    let mut postings_out = BufWriter::new(File::create(format!("{}.tmp", args.out_prefix))?);
    let mut postcount: u32 = 0;
    let mut bar;

    for bundle in args.bundles {
        let tokenizer = get_tokenizer(args.tokenizer.as_ref().expect("Unknown tokenizer"));
        let mut reader = utils::reader(&bundle);

        let num_lines = reader.lines().count();
        bar = tqdm!(total = num_lines);
        reader = utils::reader(&bundle);

        println!("Tokenizing, saving feature vectors and docinfos");
        for line in reader.lines() {
            bar.update(1)?;
            num_docs += 1;
            let docmap = from_str::<Map<String, Value>>(&line.unwrap()).unwrap();
            let (docid, docmap) =
                tokenize_and_map(docmap, &tokenizer, &mut dict, &args.docid, &args.body);
            let mut fv = FeatureVec::new(docid.clone());
            for (tok, count) in docmap {
                let pt = PTuple {
                    tok: tok,
                    docid: num_docs,
                    count: count,
                };
                pt.serialize(&mut postings_out, bincode_config)
                    .expect("Error writing postings");
                fv.push(tok, count as f32);
                postcount += 1;
            }
            let offset = featfile.stream_position().unwrap();
            bincode::encode_into_std_write(fv, &mut featfile, bincode_config)
                .expect("Error writing feature vec");
            let di = DocInfo {
                intid: num_docs,
                docid: docid.clone(),
                offset,
            };
            docsdb.insert_batch(&docid, &di, 100_000);
        }
    }

    docsdb.process_remaining();
    let _ = docsdb.ext2int.flush();
    postings_out.flush().unwrap();
    std::mem::drop(postings_out);

    println!("Sorting postings");
    let post_in = format!("{}.tmp", args.out_prefix);
    let post_out = format!("{}.pst", args.out_prefix);
    external_sort::<PTuple>(&post_in, &post_out, 10_000_000, "./tmp", bincode_config)?;

    println!("Adding postings");
    bar = tqdm!(total = postcount as usize);
    let mut last_tok = 0;
    let mut postings = BufReader::new(File::open(post_out)?);
    for p in iter::from_fn(move || PTuple::deserialize(&mut postings, bincode_config).ok()) {
        bar.update(1)?;
        // Check if we should dump, but only after we have the complete posting list for a token.
        if p.tok != last_tok {
            last_tok = p.tok;
            if invfile.memusage() > 100 * MB as u32 {
                let _ = invfile.save()?;
            }
        }
        invfile.add_posting(p.tok, p.docid, p.count);
    }

    if invfile.memusage() > 0 {
        let _ = invfile.save()?;
    }

    std::fs::remove_file(post_in)?;

    println!("Precompute IDF");
    let mut new_dict = Dict::new();
    let mut last_tokid = 0;
    bar = tqdm!(total = dict.m.len());
    dict.m.drain().for_each(|(tok, tokid)| {
        let _ = bar.update(1);
        if let Some(df) = dict.df.get(&tokid) {
            new_dict.m.insert(tok, tokid);
            if tokid > last_tokid {
                last_tokid = tokid;
            }
            new_dict.df.insert(tokid, (num_docs as f32 / df).log10());
        }
    });
    new_dict.last_tokid = last_tokid;
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
