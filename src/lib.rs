use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use ordered_float::OrderedFloat;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub mod classifier;
pub use classifier::Classifier;
pub mod tok;
pub use tok::Tokenizer;
pub mod compress;
pub mod extsort;
pub mod index;
pub mod utils;
pub use extsort::external_sort;
pub mod store;
pub use store::Store;
pub mod lrucache;
pub mod odch;
pub mod ptuple;

// DocInfos help us find the document features in the feature vec file
#[derive(Debug, Encode, Decode, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct DocInfo {
    pub intid: u32,    // internal id (1 ..)
    pub docid: String, // external id (from original doc)
    pub offset: u64,   // offset into the feature vec file
}

// A hashmap of docid -> intid, and a Vector of DocInfos
// This is used in-memory to store docinfo information
// from the first pass of build_corpus().
// Since the addition of the DocsDb, we don't persist
// this on disk anymore.
#[derive(Debug, Encode, Decode)]
pub struct Docs {
    pub m: HashMap<String, u32>, // map of docid to internal id
    pub docs: Vec<DocInfo>,      // vec of DocInfo (docid, intid, offset) tuples
}

impl Docs {
    pub fn new() -> Docs {
        Docs {
            m: HashMap::new(),
            docs: Vec::new(),
        }
    }
    pub fn get_intid(&self, docid: &str) -> Option<&u32> {
        self.m.get(docid)
    }
    pub fn add_doc(&mut self, docid: &str) -> u32 {
        if self.m.contains_key(docid) {
            self.m.get(docid).unwrap().to_owned()
        } else {
            let intid = self.docs.len() as u32 + 1;
            self.m.insert(docid.to_string(), intid);
            self.docs.push(DocInfo {
                docid: docid.to_string(),
                intid,
                offset: 0,
            });
            intid
        }
    }
}

// The Dict is the lexicon, mapping tokens to internal token ids,
// and keeping document frequencies for each token.
// This is kept on disk and brought into memory completely when
// we use it.
// TODO why don't we just keep the offsets in here? Why DocInfo?
#[derive(Debug, Encode, Decode)]
pub struct Dict {
    pub m: HashMap<String, usize>, // map token to internal tokid
    pub df: HashMap<usize, f32>,   // df for each tokid
    pub last_tokid: usize,
}

impl Dict {
    pub fn new() -> Dict {
        Dict {
            m: HashMap::new(),
            df: HashMap::new(),
            last_tokid: 1,
        }
    }
    pub fn load(filename: &str) -> Dict {
        let mut infp = BufReader::new(File::open(filename).expect("Error opening Dict file"));
        bincode::decode_from_std_read(&mut infp, bincode::config::standard())
            .expect("Error reading Dict")
    }
    pub fn has_tok(&self, tok: String) -> bool {
        self.m.contains_key(&tok)
    }
    pub fn get_tokid(&self, tok: String) -> Option<&usize> {
        self.m.get(&tok)
    }
    pub fn add_tok(&mut self, tok: String) -> usize {
        if self.m.contains_key(&tok) {
            self.m.get(&tok).unwrap().to_owned()
        } else {
            self.last_tokid += 1;
            self.m.insert(tok, self.last_tokid);
            self.last_tokid
        }
    }
    pub fn incr_df(&mut self, tokid: usize) {
        *self.df.entry(tokid).or_insert(0.0) += 1.0;
    }
    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut outfp = BufWriter::new(File::create(filename)?);
        bincode::encode_into_std_write(self, &mut outfp, bincode::config::standard())
            .expect("Error writing Dict");
        outfp.flush()?;
        Ok(())
    }
}

// A feature, an internal token ID and a float value.
// This is where we learned that core Rust refuses to
// sort floats because maybe NaN.  There is a create
// called OrderedFloat that we use.  #ffs
#[derive(Debug, Encode, Decode)]
pub struct FeaturePair {
    pub id: usize,
    pub value: f32,
}

// Feature vectors are the parsed representation of documents.
// We use these while training up the classifier, and in
// the score_collection function in main, we compute scores
// by iterating the FeatureVec file.  That's pretty fast
// up until a million docs or so.
#[derive(Debug, Encode, Decode)]
pub struct FeatureVec {
    pub docid: String,
    pub features: Vec<FeaturePair>,
    pub squared_norm: f32,
}

impl FeatureVec {
    pub fn new(docid: String) -> FeatureVec {
        FeatureVec {
            docid,
            features: Vec::new(),
            squared_norm: 0.0,
        }
    }
    pub fn read_from<R: BufRead + Sized>(fp: &mut R) -> Result<FeatureVec, DecodeError> {
        bincode::decode_from_std_read(fp, bincode::config::standard())
    }
    pub fn write_to<W: Write>(&self, fp: &mut W) -> Result<usize, EncodeError> {
        bincode::encode_into_std_write(self, fp, bincode::config::standard())
    }
    pub fn num_features(&self) -> usize {
        self.features.len()
    }
    pub fn feature_at(&self, i: usize) -> usize {
        self.features[i].id
    }
    pub fn value_at(&self, i: usize) -> f32 {
        self.features[i].value
    }
    pub fn push(&mut self, id: usize, val: f32) {
        self.features.push(FeaturePair { id, value: val });
    }
    pub fn compute_norm(&mut self) {
        let norm = self
            .features
            .iter()
            .map(|fp| fp.value * fp.value)
            .reduce(|acc, e| acc + e)
            .unwrap_or(0.0)
            .sqrt();
        self.squared_norm = norm;
    }
    // pub fn normalize(&mut self) -> &FeatureVec {
    //     let new_features = self.features.iter().map(|fp| FeaturePair { id: fp.id, value: fp.value / self.squared_norm }).collect();
    //     self.features = new_features;
    //     self
    // }
}

// Scores for documents, used during score_collection.
// Note we need OrderedFloat because core Rust won't
// sort floats because NaN.
#[derive(Eq, Debug, Clone)]
pub struct DocScore {
    pub docid: String,
    pub score: OrderedFloat<f32>,
}

// We need Serialize to be able to ship DocScores out from webcal
impl Serialize for DocScore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DocScore", 2)?;
        state.serialize_field("docid", &self.docid)?;
        state.serialize_field("score", &f32::from(self.score))?;
        state.end()
    }
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
