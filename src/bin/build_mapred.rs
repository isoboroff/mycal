use bytesize::GB;
use clap::Parser;
use kdam::{tqdm, BarExt};
use log::{log_enabled, Level};
use mycal::{
    extsort::external_sort_from, index::InvertedFile, odch::OnDiskCompressedHash, ptuple::PTuple,
    store::Config, tok::get_tokenizer, utils, FeatureVec,
};
use serde_json::{from_str, Map, Value};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek, Write},
};

fn build_index(args: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let bincode_conf = bincode::config::standard();
    let tokenizer = get_tokenizer(args.tokenizer.as_ref().expect("Unknown tokenizer"));
    let config = bincode::config::standard();
    let mut token_tokid_map = OnDiskCompressedHash::new();
    let mut docid_intid_map = OnDiskCompressedHash::new();
    let mut tuples_out = BufWriter::new(File::create(&format!("{}/tuples", args.out_prefix))?);
    let mut num_tuples: u32 = 0;
    let mut num_docs: u32 = 0;

    // Step 1:
    // Parse documents into a sequence of (token, docid, count) tuples.
    // tokens are internal token ids.
    // docids are u128 versions of UUIDs.

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
            let intid = docid_intid_map.get_or_insert(&docid);
            let text = doc.get(&args.body).unwrap().as_str().unwrap();
            let mut token_map: HashMap<String, u32> = HashMap::new();

            for t in tokenizer.tokenize(text) {
                let v = token_map.entry(t).or_insert(0);
                *v = *v + 1;
            }
            for (t, count) in token_map {
                let tokid = token_tokid_map.get_or_insert(&t);
                let pt = PTuple::new(tokid, intid, count);
                bincode::encode_into_std_write(pt, &mut tuples_out, bincode_conf)?;
                num_tuples += 1;
            }
        }
    }

    println!("{} token tuples", num_tuples);
    println!("Saving vocab");
    tuples_out.flush()?;
    // std::mem::drop(tuples_out);
    token_tokid_map.save(&format!("{}/vocab", args.out_prefix))?;
    docid_intid_map.save(&format!("{}/docid_map", args.out_prefix))?;

    // Step 2
    // Sort the tuples, so now they are in token-id order

    println!("Sorting tuples");
    let tuples_in = format!("{}/tuples", args.out_prefix);
    let mut tuples_in_fp = BufReader::new(File::open(&tuples_in)?);
    let tuples_out = format!("{}/tuples_sorted", args.out_prefix);
    let mut tuples_out_fp = BufWriter::new(File::create(&tuples_out)?);

    external_sort_from::<PTuple, _, _>(&mut tuples_in_fp, &mut tuples_out_fp, 10_000_000, "tmp")?;
    tuples_out_fp.flush()?;

    // Step 3
    // build inverted file

    let mut sorted_tuples = BufReader::new(File::open(&tuples_out)?);
    let mut inverted_file = InvertedFile::new(&format!("{}/inverted_file", args.out_prefix));
    let mut pt: PTuple = bincode::decode_from_std_read(&mut sorted_tuples, config)?;
    let mut bar = tqdm!(desc = "Building inverted file", total = num_tuples as usize);
    let mut current_intid = pt.docid;
    let mut tuple_count: u32 = 1;
    loop {
        bar.update(1)?;
        if pt.docid != current_intid {
            current_intid = pt.docid;
            // To do: check cache size and save out invfile if needed
            if inverted_file.memusage() > (100 * GB) as u32 {
                bar.set_description("checkpointing invfile");
                inverted_file.save()?;
                bar.set_description("Building inverted file");
            }
        }
        inverted_file.add_posting(pt.tok, current_intid as u32, pt.count);
        pt = match bincode::decode_from_std_read(&mut sorted_tuples, config) {
            Ok(pt) => pt,
            Err(_e) => break,
        };
        tuple_count += 1;
    }

    assert_eq!(
        tuple_count, num_tuples,
        "Tuple count mismatch, {} parsed, {} inverted",
        num_tuples, tuple_count
    );
    println!("Saving invfile");
    inverted_file.save()?;

    // Step 4
    // build feature vector file

    let mut tuples = BufReader::new(File::open(&tuples_in)?);
    let mut featfile = BufWriter::new(File::create(&format!(
        "{}/feature_vectors",
        args.out_prefix
    ))?);
    let mut offsets: HashMap<usize, u64> = HashMap::new();
    let mut bar = tqdm!(desc = "Building feature vectors", total = num_docs as usize);
    let mut pt: PTuple = bincode::decode_from_std_read(&mut tuples, config)?;
    let mut current_intid = pt.docid;
    let mut current_docid = docid_intid_map.get_key_for(current_intid).unwrap();
    let mut fv = FeatureVec::new(current_docid.clone());
    let mut df: HashMap<u32, f32> = HashMap::new();
    loop {
        if pt.docid != current_intid {
            fv.compute_norm();
            offsets.insert(current_intid, featfile.stream_position()?);
            fv.write_to(&mut featfile)?;
            let idf = (num_docs as f32 / fv.num_features() as f32).log10();
            df.insert(current_intid as u32, idf);
            bar.update(1)?;
            current_intid = pt.docid;
            current_docid = match docid_intid_map.get_key_for(current_intid) {
                Some(docid) => docid,
                None => panic!("Missing docid for intid {}", current_intid),
            };
            bar.update(1)?;
            fv = FeatureVec::new(current_docid.clone());
        }
        fv.push(pt.tok, pt.count as f32);
        pt = match bincode::decode_from_std_read(&mut tuples, config) {
            Ok(pt) => pt,
            Err(_) => break,
        }
    }
    fv.compute_norm();
    offsets.insert(current_intid, featfile.stream_position()?);
    fv.write_to(&mut featfile)?;
    let idf = (num_docs as f32 / fv.num_features() as f32).log10();
    df.insert(current_intid as u32, idf);
    bar.update(1)?;
    featfile.flush()?;

    let mut df_file = BufWriter::new(File::create(format!("{}/idf", args.out_prefix))?);
    bincode::encode_into_std_write(df, &mut df_file, bincode_conf)?;
    df_file.flush()?;

    let mut off_file = BufWriter::new(File::create(format!("{}/fv_offsets", args.out_prefix))?);
    bincode::encode_into_std_write(offsets, &mut off_file, bincode_conf)?;
    off_file.flush()?;

    std::fs::remove_file(format!("{}/tuples", args.out_prefix))?;
    std::fs::remove_file(format!("{}/tuples_sorted", args.out_prefix))?;

    let config = Config {
        num_docs: docid_intid_map.len(),
        num_features: token_tokid_map.len(),
    };
    let mut conf_file = File::create(format!("{}/config.toml", args.out_prefix))?;
    conf_file.write_all(toml::to_string(&config).unwrap().as_ref())?;

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
