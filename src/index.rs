use bincode::{Decode, Encode};
use log::debug;
use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use crate::compress::{magic_bytes_required, vbyte_bytes_required, MagicEncodedBuffer};

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
        let posting = Posting { doc_id, tf };
        self.postings.push(posting);
    }
    pub fn bytes_to_serialize(&mut self) -> usize {
        let mut bytes: usize = 0;
        let mut last_docid: u32 = 0;
        self.postings.sort();

        bytes += vbyte_bytes_required(self.postings.len() as u32);
        for p in &self.postings {
            bytes += magic_bytes_required(p.doc_id - last_docid, p.tf);
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

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_posting_list() {
        env_logger::init();
        let mut pl = PostingList::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let docid = rng.random_range(1..1000);
            let tf = rng.random_range(1..100);
            pl.add_posting(docid, tf);
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

#[derive(Encode, Decode)]
pub struct PostInfo {
    offset: u64,
    len: usize,
}

pub struct InvertedFile {
    inv_filename: PathBuf,
    offsets: HashMap<usize, PostInfo>,
    cache: HashMap<usize, PostingList>,
    bincode_config: bincode::config::Configuration,
}

impl InvertedFile {
    pub fn open(path: &str) -> Result<InvertedFile, std::io::Error> {
        let offfile = File::open(Path::new(path).with_extension("off"))?;
        let mut offreader = BufReader::new(offfile);
        let config = bincode::config::standard();
        let offsets: HashMap<usize, PostInfo> =
            bincode::decode_from_std_read(&mut offreader, config).unwrap();
        Ok(InvertedFile {
            inv_filename: Path::new(path).with_extension("inv"),
            offsets: offsets,
            cache: HashMap::new(),
            bincode_config: config,
        })
    }
    pub fn new(path: &Path) -> InvertedFile {
        InvertedFile {
            inv_filename: Path::new(path).with_extension("inv"),
            offsets: HashMap::new(),
            cache: HashMap::new(),
            bincode_config: bincode::config::standard(),
        }
    }
    pub fn add_posting(&mut self, tokid: usize, docid: u32, tf: u32) {
        assert_ne!(0, docid);
        let pl = self.cache.entry(tokid).or_insert(PostingList::new());
        pl.add_posting(docid, tf);
    }
    pub fn get_posting_list(&mut self, tokid: usize) -> Result<PostingList, std::io::Error> {
        if self.cache.contains_key(&tokid) {
            Ok(self.cache.get(&tokid).unwrap().clone())
        } else if self.offsets.contains_key(&tokid) {
            let file = File::open(&self.inv_filename)?;
            let mut invfile = BufReader::new(file);
            let pi = self.offsets.get(&tokid).unwrap();
            invfile.seek(std::io::SeekFrom::Start(pi.offset)).unwrap();
            let mut bytes = vec![0; pi.len].into_boxed_slice();
            invfile.read_exact(&mut bytes).unwrap();
            let pl = PostingList::deserialize(&mut MagicEncodedBuffer::from_vec((*bytes).to_vec()));
            Ok(pl)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Can't get posting list",
            ))
        }
    }
    pub fn save(&mut self) {
        let file = File::create(&self.inv_filename).unwrap();
        let mut invfile = BufWriter::new(file);
        for (tokid, pl) in &mut self.cache {
            let offset = invfile.stream_position().unwrap();
            let bytes = pl.bytes_to_serialize();
            self.offsets.insert(*tokid, PostInfo { offset, len: bytes });
            let mut buf = MagicEncodedBuffer::new_with_capacity(bytes);
            pl.serialize_into(&mut buf);
            invfile.write_all(buf.as_slice()).unwrap();
        }
        let offfile = File::create(self.inv_filename.with_extension("off")).unwrap();
        let mut offwriter = BufWriter::new(offfile);
        bincode::encode_into_std_write(&self.offsets, &mut offwriter, self.bincode_config).unwrap();
    }
}
