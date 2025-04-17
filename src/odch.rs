//! The OnDiskCompressedHash is a HashMap of String to usize
//! backed by a Vec, compressed on disk.  So the order of the
//! Vec are the values of the hashed Strings.
//!
//! Basically this a Vec with a reverse lookup table.
//!
//! The application is things like mapping from external
//! strings to an internal auto-incremented index.  
//! Once you save() the hash, it will not let you
//! save it again.

use bincode::{decode_from_std_read, encode_to_vec, Decode, Encode};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use std::io::Write;
use std::{collections::HashMap, fs::File};

#[derive(Encode, Decode)]
pub struct OnDiskCompressedHash {
    map: HashMap<String, usize>,
    idx: Vec<String>,
    finalized: bool,
}

impl OnDiskCompressedHash {
    pub fn new() -> OnDiskCompressedHash {
        let map = HashMap::new();
        let idx = Vec::new();
        OnDiskCompressedHash {
            map,
            idx,
            finalized: false,
        }
    }
    pub fn open(filename: &str) -> Result<OnDiskCompressedHash, bincode::error::DecodeError> {
        let mut map: HashMap<String, usize> = HashMap::new();
        let fp = File::open(filename).unwrap();
        let mut decompressed = FrameDecoder::new(fp);
        let idx: Vec<String> =
            decode_from_std_read(&mut decompressed, bincode::config::standard())?;
        for (i, el) in idx.iter().enumerate() {
            map.insert(el.clone(), i);
        }
        Ok(OnDiskCompressedHash {
            map,
            idx,
            finalized: false,
        })
    }

    pub fn from_hash(map: HashMap<String, usize>) -> OnDiskCompressedHash {
        let idx = map.keys().cloned().collect();
        OnDiskCompressedHash {
            map,
            idx,
            finalized: false,
        }
    }

    pub fn save(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.finalized {
            panic!("Cannot save a finalized OnDiskCompressedHash");
        }
        // Do not do this: self.idx.sort();
        let encoded = encode_to_vec(&self.idx, bincode::config::standard())?;
        let fp = File::create(filename).unwrap();
        let mut compressed = FrameEncoder::new(fp);
        compressed.write(&encoded)?;
        compressed.flush()?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<usize> {
        self.map.get(key).copied()
    }

    pub fn get_or_insert(&mut self, key: &str) -> usize {
        if self.finalized {
            panic!("Cannot add to a finalized OnDiskCompressedHash");
        }
        if self.map.contains_key(key) {
            return self.map.get(key).copied().unwrap();
        }
        let id = self.idx.len() + 1;
        self.map.insert(key.to_string(), id);
        self.idx.push(key.to_string());
        id
    }

    pub fn get_idx_for(&self, key: &str) -> Option<usize> {
        self.map.get(key).cloned()
    }

    pub fn get_key_for(&self, intid: usize) -> Option<&String> {
        self.idx.get(intid - 1)
    }

    pub fn get_keys(&self) -> Vec<String> {
        self.idx.clone()
    }

    pub fn get_values(&self) -> Vec<usize> {
        self.map.values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.idx.len()
    }

    pub fn map(&self) -> &HashMap<String, usize> {
        &self.map
    }
}
