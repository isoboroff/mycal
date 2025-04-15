use bincode::{Decode, Encode};
use clap::Parser;
use kdam::{tqdm, BarExt};
use log::{log_enabled, Level};
use mycal::{
    extsort::external_sort_from, index::InvertedFile, odch::OnDiskCompressedHash,
    tok::get_tokenizer, utils, FeatureVec,
};
use serde_json::{from_str, Map, Value};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};
use uuid::Uuid;

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Debug, Encode, Decode)]
pub struct PTuple {
    tok: usize,
    docid: [u8; 16],
    count: u32,
}

fn build_index(args: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let bincode_conf = bincode::config::standard();
    let tokenizer = get_tokenizer(args.tokenizer.as_ref().expect("Unknown tokenizer"));
    let config = bincode::config::standard();
    let mut token_tokid_map = OnDiskCompressedHash::new();
    let mut token_id: usize = 0;
    let mut tuples_out = lz4_flex::frame::FrameEncoder::new(BufWriter::new(File::create(
        &format!("{}/tuples", args.out_prefix),
    )?));
    let mut num_tuples: u32 = 0;
    let mut num_docs: u32 = 0;

    println!("Tokenizing bundles");
    for bundle in args.bundles {
        let num_lines = utils::reader(&bundle).lines().count();
        let fp = utils::reader(&bundle);
        let mut bar = tqdm!(desc = &bundle, total = num_lines);

        for line in fp.lines() {
            let line = line?;
            bar.update(1)?;
            num_docs += 1;
            let doc: Map<String, Value> = from_str(&line)?;
            let docid = doc.get(&args.docid).unwrap().as_str().unwrap();
            let text = doc.get(&args.body).unwrap().as_str().unwrap();
            let mut token_map: HashMap<String, u32> = HashMap::new();

            for t in tokenizer.tokenize(text) {
                let v = token_map.entry(t).or_insert(0);
                *v = *v + 1;
            }
            for (t, count) in token_map {
                let tokid = token_tokid_map.get_or_insert(&t);
                token_id = token_id + 1;
                let pt = PTuple {
                    tok: tokid,
                    docid: Uuid::parse_str(docid)?.into_bytes(),
                    count: count,
                };
                bincode::encode_into_std_write(pt, &mut tuples_out, bincode_conf)?;
                num_tuples += 1;
            }
        }
    }

    println!("{} token tuples", num_tuples);
    println!("Saving vocab");
    tuples_out.finish()?;
    // std::mem::drop(tuples_out);
    token_tokid_map.save(&format!("{}/vocab", args.out_prefix))?;

    println!("Sorting tuples");
    let tuples_in = format!("{}/tuples", args.out_prefix);
    let mut tuples_in_fp = BufReader::new(File::open(&tuples_in)?);
    let tuples_out = format!("{}/tuples_sorted", args.out_prefix);
    let mut tuples_out_fp = BufWriter::new(File::create(&tuples_out)?);

    external_sort_from::<PTuple, _, _>(&mut tuples_in_fp, &mut tuples_out_fp, 10_000_000, "tmp")?;
    tuples_out_fp
        .flush()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;

    // build inverted file
    let mut sorted_tuples = BufReader::new(File::open(&tuples_out)?);
    let mut inverted_file = InvertedFile::new(&format!("{}/inverted_file", args.out_prefix));
    let mut docid_intid_map = OnDiskCompressedHash::new();
    let mut pt: PTuple = bincode::decode_from_std_read(&mut sorted_tuples, config)?;
    let mut bar = tqdm!(desc = "Building inverted file", total = num_tuples as usize);
    let mut current_uuid = pt.docid;
    let mut current_docid = Uuid::from_bytes(pt.docid).to_string();
    let mut current_intid = docid_intid_map.get_or_insert(&current_docid);
    loop {
        bar.update(1)?;
        if pt.docid != current_uuid {
            current_docid = Uuid::from_bytes(pt.docid).to_string();
            current_intid = docid_intid_map.get_or_insert(&current_docid);
            current_uuid = pt.docid;
        }
        inverted_file.add_posting(pt.tok, current_intid as u32, pt.count);
        pt = match bincode::decode_from_std_read(&mut sorted_tuples, config) {
            Ok(pt) => pt,
            Err(_e) => break,
        }
    }

    println!("Saving invfile");
    inverted_file.save()?;
    println!("Saving doc-int map");
    docid_intid_map.save(&format!("{}/docid_map", args.out_prefix))?;

    let mut tuples = BufReader::new(File::open(&tuples_in)?);
    let mut featfile = BufWriter::new(File::create(&format!(
        "{}/feature_vectors",
        args.out_prefix
    ))?);
    let mut pt: PTuple = bincode::decode_from_std_read(&mut tuples, config)?;
    let mut current_uuid = pt.docid;
    let mut current_docid = Uuid::from_bytes(current_uuid).to_string();
    let mut fv = FeatureVec::new(current_docid.clone());
    let mut df: HashMap<u32, f32> = HashMap::new();
    bar = tqdm!(desc = "Storing feature vectors", total = num_docs as usize);
    loop {
        if pt.docid != current_uuid {
            fv.compute_norm();
            fv.write_to(&mut featfile)?;
            let idf = (num_docs as f32 / fv.num_features() as f32).log10();
            let intid = docid_intid_map
                .get(&current_docid)
                .expect("Missing docid in idf computation");
            df.insert(intid as u32, idf);
            bar.update(1)?;
            current_uuid = pt.docid;
            current_docid = Uuid::from_bytes(current_uuid).to_string();
            fv = FeatureVec::new(current_docid.clone());
        }
        fv.push(pt.tok, pt.count as f32);
        pt = match bincode::decode_from_std_read(&mut tuples, config) {
            Ok(pt) => pt,
            Err(_) => break,
        }
    }
    featfile.flush()?;

    let mut df_file = BufWriter::new(File::create(format!("{}/idf", args.out_prefix))?);
    bincode::encode_into_std_write(df, &mut df_file, bincode_conf)?;
    df_file.flush()?;

    std::fs::remove_file(format!("{}/tuples", args.out_prefix))?;
    std::fs::remove_file(format!("{}/tuples_sorted", args.out_prefix))?;

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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    if log_enabled!(Level::Debug) {
        env_logger::init();
    }
    build_index(args)
}
