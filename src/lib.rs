use bincode::config::Configuration;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

pub mod classifier;
pub use classifier::Classifier;
pub mod tok;
pub use tok::Tokenizer;
pub mod compress;
pub mod extsort;
pub mod index;
pub mod utils;
pub use extsort::{external_sort, SerializeDeserialize};

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

// The DocsDb is a sled::Db mapping of external docid to DocInfo.
// This implementation happens because sled was my intro to
// swerialization.
pub struct DocsDb {
    // Sled-based docid -> DocInfo table
    pub prefix: String,
    pub ext2int: sled::Db,
    pub int2ext: sled::Db,
    pub next_intid: u32,

    e2i_batch: sled::Batch,
    i2e_batch: sled::Batch,
    batch_len: usize,
    bincode_config: bincode::config::Configuration,
}

impl DocsDb {
    // for reading
    pub fn open(prefix: &str) -> DocsDb {
        let e2i_path = format!("{}.e2i", prefix);
        let i2e_path = format!("{}.i2e", prefix);
        if let Ok(ext2int) = sled::Config::default()
            .cache_capacity(10_000_000)
            .use_compression(false)
            .path(e2i_path)
            .mode(sled::Mode::LowSpace)
            .open()
        {
            if let Ok(int2ext) = sled::Config::default()
                .cache_capacity(10_000_000)
                .use_compression(false)
                .path(i2e_path)
                .mode(sled::Mode::LowSpace)
                .open()
            {
                return DocsDb {
                    prefix: prefix.to_string(),
                    ext2int: ext2int.clone(),
                    int2ext: int2ext.clone(),
                    next_intid: 1,
                    e2i_batch: sled::Batch::default(),
                    i2e_batch: sled::Batch::default(),
                    batch_len: 0,
                    bincode_config: bincode::config::standard(),
                };
            } else {
                panic!("Could not open i2e database");
            }
        } else {
            panic!("Could not open e2i database");
        }
    }

    pub fn create(prefix: &str) -> DocsDb {
        let e2i_conf = sled::Config::default()
            .cache_capacity(10_000_000)
            .use_compression(false)
            .path(format!("{}.e2i", prefix))
            .mode(sled::Mode::HighThroughput)
            .open();
        let i2e_conf = sled::Config::default()
            .cache_capacity(10_000_000)
            .use_compression(false)
            .path(format!("{}.i2e", prefix))
            .mode(sled::Mode::HighThroughput)
            .open();

        DocsDb {
            prefix: prefix.to_string(),
            ext2int: e2i_conf.expect("Couldn't open e2i"),
            int2ext: i2e_conf.expect("Couldn't open i2e"),
            next_intid: 1,
            e2i_batch: sled::Batch::default(),
            i2e_batch: sled::Batch::default(),
            batch_len: 0,
            bincode_config: bincode::config::standard(),
        }
    }

    pub fn get_docid(&self, docid: &str) -> Option<DocInfo> {
        self.ext2int.get(docid).unwrap().map(|ivec| {
            bincode::decode_from_slice::<DocInfo, Configuration>(&ivec, self.bincode_config)
                .unwrap()
                .0
        })
    }

    pub fn get_intid(&self, intid: u32) -> Option<DocInfo> {
        self.int2ext
            .get(u32::to_be_bytes(intid))
            .unwrap()
            .map(|ivec| {
                bincode::decode_from_slice::<DocInfo, Configuration>(&ivec, self.bincode_config)
                    .unwrap()
                    .0
            })
    }

    pub fn insert(&self, docid: &str, di: &DocInfo) -> Option<sled::IVec> {
        let dib = bincode::encode_to_vec::<DocInfo, Configuration>(di.clone(), self.bincode_config)
            .unwrap();
        // not happy about the error reporting here...
        self.int2ext
            .insert(u32::to_be_bytes(di.intid), dib.to_vec())
            .ok()
            .unwrap();
        self.ext2int.insert(docid, dib.to_vec()).ok().unwrap()
    }

    pub fn insert_batch(&mut self, docid: &str, di: &DocInfo, batch_size: usize) {
        let dib = bincode::encode_to_vec::<DocInfo, Configuration>(di.clone(), self.bincode_config)
            .unwrap();
        if self.batch_len > batch_size {
            // We need to use swap because apply_batch consumes the batch
            let mut local_batch = sled::Batch::default();
            std::mem::swap(&mut local_batch, &mut self.e2i_batch);
            self.ext2int
                .apply_batch(local_batch)
                .expect("Batch apply fail");
            local_batch = sled::Batch::default();
            std::mem::swap(&mut local_batch, &mut self.i2e_batch);
            self.int2ext
                .apply_batch(local_batch)
                .expect("Batch apply fail");
            self.batch_len = 0;
        }
        self.e2i_batch.insert(docid, dib.to_vec());
        self.i2e_batch
            .insert(&u32::to_be_bytes(di.intid), dib.to_vec());
        self.batch_len += 1;
    }

    pub fn process_remaining(&mut self) {
        if self.batch_len > 0 {
            let mut local_batch = sled::Batch::default();
            std::mem::swap(&mut local_batch, &mut self.e2i_batch);
            self.ext2int
                .apply_batch(local_batch)
                .expect("Batch apply fail");
            local_batch = sled::Batch::default();
            std::mem::swap(&mut local_batch, &mut self.i2e_batch);
            self.int2ext
                .apply_batch(local_batch)
                .expect("Batch apply fail");
            self.batch_len = 0;
        }
    }

    // pub fn insert_iter(
    //     &mut self,
    //     library: &Docs,
    //     stuff: impl Iterator<Item = (String, u32)>,
    // ) -> Result<()> {
    //     stuff.for_each(|(docid, intid)| {
    //         let di = library.docs.get(intid).unwrap();
    //         self.insert_batch(&docid, &di, 100_000);
    //     });
    //     Ok(())
    // }

    pub fn add_doc(&mut self, docid: &str) -> Option<u32> {
        let tmp_docid = docid.to_string();
        let tmp_di = self.ext2int.get(&tmp_docid).unwrap();
        match tmp_di {
            Some(di) => {
                let ddi: DocInfo = bincode::decode_from_slice(&di, self.bincode_config)
                    .unwrap()
                    .0;
                Some(ddi.intid)
            }
            None => {
                let intid = self.next_intid;
                assert_ne!(0, intid);
                self.next_intid = intid + 1;
                let new_di = DocInfo {
                    intid,
                    docid: tmp_docid.clone(),
                    offset: 0,
                };
                let sdi = bincode::encode_to_vec(new_di, self.bincode_config).unwrap();
                self.ext2int
                    .insert(&tmp_docid, sdi.to_vec())
                    .expect("Could not insert DocInfo into db");
                Some(intid)
            }
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
    pub fn read_from(fp: &mut BufReader<File>) -> Result<FeatureVec, DecodeError> {
        bincode::decode_from_std_read(fp, bincode::config::standard())
    }
    pub fn write_to(&self, mut fp: BufWriter<File>) -> Result<(), EncodeError> {
        let _ = bincode::encode_into_std_write(self, &mut fp, bincode::config::standard());
        Ok(())
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
