use bincode::{Decode, Encode};
use kdam::{tqdm, BarExt};
use log::debug;
use ordered_float::OrderedFloat;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};

use crate::{
    compress::{magic_bytes_required, vbyte_bytes_required, MagicEncodedBuffer},
    lrucache::LruCache,
    Classifier, DocScore, FeaturePair, Store,
};

// This lets us print arrays of bytes in binary format
struct Bytes<'a>(&'a [u8]);

impl<'a> std::fmt::Binary for Bytes<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for byte in self.0 {
            std::fmt::Binary::fmt(byte, f)?;
            write!(f, " ")?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Ord, Eq)]
pub struct Posting {
    pub doc_id: u32,
    pub tf: u32,
}

impl Display for Posting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.doc_id, self.tf)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PostingList {
    pub postings: Vec<Posting>,
}

impl PostingList {
    pub fn new() -> PostingList {
        PostingList { postings: vec![] }
    }
    pub fn from_vec(&self, postings: Vec<Posting>) -> PostingList {
        let my_posts = postings.clone();
        PostingList { postings: my_posts }
    }
    pub fn add_posting(&mut self, doc_id: u32, tf: u32) {
        assert!(doc_id > 0, "doc_id {} tf {}", doc_id, tf);
        assert!(tf > 0, "doc_id {} tf {}", doc_id, tf);
        let posting = Posting { doc_id, tf };
        self.postings.push(posting);
    }
    pub fn bytes_to_serialize(&mut self) -> usize {
        let mut bytes: usize = 0;
        let mut last_docid: u32 = 0;
        self.postings.sort();

        bytes += vbyte_bytes_required(self.postings.len() as u32);
        for p in &self.postings {
            let mut docgap = p.doc_id;
            if last_docid > 0 {
                docgap = p.doc_id - last_docid;
            }
            assert!(docgap != 0, "p {:?} last_docid {}", p, last_docid);
            bytes += magic_bytes_required(docgap, p.tf);
            last_docid = p.doc_id;
        }
        bytes
    }
    pub fn serialize_into(&mut self, buf: &mut MagicEncodedBuffer) {
        let num_postings = self.postings.len() as u32;
        self.postings.sort();
        buf.reset();
        buf.vbyte_write(num_postings);
        debug!(
            "num_postings: {} {:08b}",
            num_postings,
            Bytes(buf.byte_slice(0, buf.tell()))
        );
        for posting in &self.postings {
            let pt: usize = buf.tell();
            buf.write(posting.doc_id, posting.tf);
            debug!(
                "posting: {} {:08b}",
                posting,
                Bytes(buf.byte_slice(pt, buf.tell()))
            );
        }
        buf.buffer.buffer.shrink_to_fit();
    }

    pub fn deserialize(buf: &mut MagicEncodedBuffer) -> PostingList {
        buf.reset();
        let num_postings = buf.vbyte_read().unwrap();
        debug!(
            "num_postings: {} {:08b}",
            num_postings,
            Bytes(buf.byte_slice(0, buf.tell()))
        );
        let mut postings = Vec::with_capacity(num_postings as usize);
        for _ in 0..num_postings {
            let pt: usize = buf.tell();
            let (docid, tf) = buf.read();
            debug!(
                "posting: {} {} {:08b}",
                docid,
                tf,
                Bytes(buf.byte_slice(pt, buf.tell()))
            );
            postings.push(Posting {
                doc_id: docid,
                tf: tf,
            });
        }
        PostingList { postings: postings }
    }
}

impl AsRef<PostingList> for PostingList {
    fn as_ref(&self) -> &PostingList {
        self
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_posting_list() {
        env_logger::init();
        let mut pl = PostingList::new();
        let mut rng = rand::rng();
        let docs = rand::seq::index::sample(&mut rng, 1000, 100);
        for d in docs {
            let tf = rng.random_range(1..100);
            pl.add_posting(d as u32, tf);
        }
        let bytes_needed = pl.bytes_to_serialize();
        let mut buf: MagicEncodedBuffer = MagicEncodedBuffer::new_with_capacity(bytes_needed);
        pl.serialize_into(&mut buf);
        let pl2 = PostingList::deserialize(&mut buf);
        assert_eq!(pl, pl2);
    }
}

pub enum Token {
    Token(String),
    Id(usize),
    None,
}

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PostInfo {
    offset: u64,
    len: usize,
}

impl PostInfo {
    pub fn new(offset: u64, len: usize) -> Self {
        PostInfo { offset, len }
    }
}

pub struct InvertedFile {
    inv_filename: String,
    offsets: Vec<PostInfo>,
    cache: LruCache<usize, PostingList>,
    bincode_config: bincode::config::Configuration,
    cache_count: u32,
    pl_gets: u32,
    cache_hits: u32,
}

impl InvertedFile {
    pub fn open(path: &str, cache_capacity: usize) -> Result<InvertedFile, std::io::Error> {
        let offfile = File::open(Path::new(path).with_extension("off"))?;
        let mut offreader = BufReader::new(offfile);
        let config = bincode::config::standard();
        let offsets: Vec<PostInfo> = bincode::decode_from_std_read(&mut offreader, config).unwrap();
        Ok(InvertedFile {
            inv_filename: path.to_string(),
            offsets: offsets,
            cache: LruCache::new(cache_capacity),
            bincode_config: config,
            cache_count: 0,
            pl_gets: 0,
            cache_hits: 0,
        })
    }
    pub fn new(path: &str) -> InvertedFile {
        InvertedFile {
            inv_filename: path.to_string(),
            offsets: Vec::new(),
            cache: LruCache::new(0),
            bincode_config: bincode::config::standard(),
            cache_count: 0,
            pl_gets: 0,
            cache_hits: 0,
        }
    }
    pub fn add_posting(&mut self, tokid: usize, docid: u32, tf: u32) {
        assert_ne!(0, tokid);
        assert_ne!(0, docid);
        assert_ne!(0, tf);
        self.pl_gets += 1;
        let pl: &mut PostingList = match self.cache.get_mut(&tokid) {
            Some(pl) => {
                self.cache_hits += 1;
                pl
            }
            None => {
                let pl: PostingList = PostingList::new();
                self.cache.insert(tokid, pl);
                self.cache.get_mut(&tokid).unwrap()
            }
        };
        pl.add_posting(docid, tf);
        self.cache_count += 1;
    }

    pub fn get_posting_list(&mut self, tokid: usize) -> Result<&PostingList, std::io::Error> {
        assert_ne!(tokid, 0);
        self.pl_gets += 1;
        if !self.cache.contains_key(tokid) {
            let file = File::open(&self.inv_filename)?;
            let mut invfile = BufReader::new(file);
            let pi = &self.offsets[tokid];
            invfile.seek(std::io::SeekFrom::Start(pi.offset)).unwrap();
            let mut bytes = vec![0; pi.len].into_boxed_slice();
            invfile.read_exact(&mut bytes).unwrap();
            let pl: PostingList =
                PostingList::deserialize(&mut MagicEncodedBuffer::from_vec((*bytes).to_vec()));
            self.cache.insert(tokid, pl.as_ref().clone());
        } else {
            self.cache_hits += 1;
        }
        Ok(self.cache.get(&tokid).unwrap())
    }

    pub fn memusage(&self) -> u32 {
        self.cache_count * std::mem::size_of::<Posting>() as u32
    }

    pub fn print_cache_stats(&self) {
        println!(
            "Gets: {} Cache hits: {} Hit rate {:.2}%",
            self.pl_gets,
            self.cache_hits,
            (self.cache_hits as f32 * 100.0 as f32) / self.pl_gets as f32
        );
    }

    // Append cache to the inverted file and dump the offset table
    // Clears the cache
    pub fn save(&mut self) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.inv_filename)?;
        let mut invfile: BufWriter<File> = BufWriter::new(file);

        for (tokid, pl) in tqdm!(self.cache.iter(), desc = "posting lists", position = 1) {
            let offset = invfile.stream_position()?;
            let mut pl = pl.clone();
            let bytes = pl.bytes_to_serialize();
            self.offsets.insert(*tokid, PostInfo { offset, len: bytes });
            let mut buf = MagicEncodedBuffer::new_with_capacity(bytes);
            pl.serialize_into(&mut buf);
            invfile.write_all(buf.as_slice())?;
        }
        invfile.flush()?;
        self.cache.clear();

        let offfpath = Path::new(&self.inv_filename).with_extension("off");
        println!("Saving offsets to {:?}", offfpath);
        let offfile = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(offfpath)?;
        let mut offwriter = BufWriter::new(offfile);
        match bincode::encode_into_std_write(&self.offsets, &mut offwriter, self.bincode_config) {
            Err(e) => Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
            Ok(_bytes) => offwriter.flush(),
        }
    }
}

pub struct IndexSearchConfig {
    num_results: usize,
    excludes: Option<Vec<String>>,
}

impl IndexSearchConfig {
    pub fn new() -> Self {
        IndexSearchConfig {
            num_results: 0,
            excludes: None,
        }
    }
    pub fn with_num_results(self, num_results: usize) -> Self {
        let mut me = self;
        me.num_results = num_results;
        me
    }
    pub fn with_exclude_docs(self, excludes: Vec<String>) -> Self {
        let mut me = self;
        me.excludes = Some(excludes);
        me
    }
    pub fn num_results(self) -> usize {
        self.num_results
    }
    pub fn excludes(self) -> Option<Vec<String>> {
        self.excludes
    }
}

pub fn score_using_index(
    coll: &mut Store,
    model: Classifier,
    config: IndexSearchConfig,
) -> Result<Vec<DocScore>, Box<dyn std::error::Error>> {
    let exclude = match config.excludes {
        Some(excl_set) => {
            let mut excludes = HashSet::new();
            for docid in excl_set {
                excludes.insert(coll.get_doc_intid(&docid)?);
            }
            excludes
        }
        _ => HashSet::new(),
    };

    // Convert the model into a vector of FeaturePairs.
    // The weight vector is in tokid order.
    let mut model_query = model
        .w
        .iter()
        .enumerate()
        .filter(|w| *w.1 != 0.0)
        .map(|(i, w)| FeaturePair {
            id: i as usize,
            value: *w,
        })
        .collect::<Vec<FeaturePair>>();

    // Run through the "query" in decreasing feature score order.
    // Later on we can try to stop early if we have to.
    model_query.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());

    let mut results: HashMap<u32, f32> = HashMap::new();

    // accumulate scores
    let mut bar = tqdm!(desc = "Scoring", total = model_query.len());
    for fpair in model_query {
        bar.update(1)?;
        let num_docs = coll.num_docs().unwrap() as f32;
        let pl = coll.get_posting_list(fpair.id).unwrap();
        let _idf = num_docs / (pl.postings.len() as f32);
        for p in pl.postings.iter() {
            if exclude.contains(&(p.doc_id as usize)) {
                continue;
            }
            let score = results.entry(p.doc_id).or_insert(0.0);
            *score += fpair.value * (p.tf as f32); // * idf
        }
    }

    let mut rvec = results
        .into_iter()
        .map(|(k, v)| {
            let di = coll.get_docid(k as usize).unwrap();
            DocScore {
                docid: di,
                score: OrderedFloat::from(v) * model.scale,
            }
        })
        .collect::<Vec<DocScore>>();
    rvec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    //for r in rvec.iter().take(config.num_results) {
    //    println!("{} {}", r.docid, r.score);
    //}
    coll.print_cache_stats();
    Ok(rvec)
}
