use bincode::decode_from_std_read;
use serde::{Deserialize, Serialize};

use crate::index::{InvertedFile, PostingList};
use crate::odch::OnDiskCompressedHash;
use crate::FeatureVec;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub num_docs: usize,
    pub num_features: usize,
    pub pl_cache_size: usize,
}

pub struct FeatureVecReader {
    fp: BufReader<File>,
}

impl FeatureVecReader {
    pub fn new(fp: BufReader<File>) -> Self {
        FeatureVecReader { fp }
    }
}

impl Iterator for FeatureVecReader {
    type Item = FeatureVec;
    fn next(&mut self) -> Option<FeatureVec> {
        FeatureVec::read_from(&mut self.fp).ok()
    }
}

pub struct Store {
    pub path: String,
    pub fvfile: Option<BufReader<File>>,
    pub invfile: Option<InvertedFile>,
    pub vocab: Option<OnDiskCompressedHash>,
    pub docids: Option<OnDiskCompressedHash>,
    pub fv_offsets: Option<HashMap<usize, u64>>,
    pub idf_values: Option<Vec<f32>>,
    pub config: Config,
}

impl Store {
    pub fn open(path: &str) -> Result<Store, std::io::Error> {
        if !std::fs::exists(path)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find path",
            ));
        }
        let config: Config = match File::open(format!("{}/config.toml", path)) {
            Ok(fp) => {
                let mut buf = String::new();
                BufReader::new(fp).read_to_string(&mut buf)?;
                toml::from_str(&buf).unwrap()
            }
            Err(_) => Config {
                num_docs: 0,
                num_features: 0,
                pl_cache_size: 100_000,
            },
        };

        Ok(Store {
            path: path.to_string(),
            fvfile: None,
            invfile: None,
            vocab: None,
            docids: None,
            fv_offsets: None,
            idf_values: None,
            config: config,
        })
    }

    pub fn get_tokid(&mut self, token: &str) -> Result<usize, Box<dyn std::error::Error>> {
        if self.vocab.is_none() {
            self.load_vocab()?;
        }
        let tokid = self
            .vocab
            .as_ref()
            .unwrap()
            .get_idx_for(token)
            .ok_or("Could not find token")?;
        Ok(tokid)
    }

    pub fn num_features(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        if self.config.num_features == 0 {
            if self.vocab.is_none() {
                match self.load_vocab() {
                    Ok(_) => self.config.num_features = self.vocab.as_ref().unwrap().len(),
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(self.config.num_features)
    }

    pub fn num_docs(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        if self.config.num_docs == 0 {
            if self.docids.is_none() {
                match self.load_docinfo() {
                    Ok(_) => self.config.num_docs = self.docids.as_ref().unwrap().len(),
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(self.config.num_docs)
    }

    pub fn get_doc_intid(&mut self, docid: &str) -> Result<usize, Box<dyn std::error::Error>> {
        if self.docids.is_none() {
            self.load_docinfo()?;
        }
        let intid = self
            .docids
            .as_ref()
            .unwrap()
            .get_idx_for(docid)
            .ok_or("Could not find docid")?;
        Ok(intid)
    }

    pub fn get_docid(&mut self, intid: usize) -> Result<String, Box<dyn std::error::Error>> {
        if self.docids.is_none() {
            self.load_docinfo()?;
        }
        let docid = self
            .docids
            .as_ref()
            .unwrap()
            .get_key_for(intid)
            .ok_or("Could not find intid")?;
        Ok(docid.clone())
    }

    pub fn get_intids(&mut self) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        if self.docids.is_none() {
            self.load_docinfo()?;
        }
        Ok(self.docids.as_ref().unwrap().get_values())
    }

    pub fn get_posting_list(
        &mut self,
        tokid: usize,
    ) -> Result<&PostingList, Box<dyn std::error::Error>> {
        if self.invfile.is_none() {
            self.invfile = Some(InvertedFile::open(
                &format!("{}/inverted_file", self.path),
                self.config.pl_cache_size,
            )?);
        }
        Ok(self
            .invfile
            .as_mut()
            .unwrap()
            .get_posting_list(tokid)
            .expect("posting list error"))
    }

    pub fn print_cache_stats(&self) {
        if self.invfile.is_some() {
            self.invfile.as_ref().unwrap().print_cache_stats();
        }
    }

    pub fn get_fv(&mut self, intid: usize) -> Result<FeatureVec, Box<dyn std::error::Error>> {
        if self.fv_offsets.is_none() {
            self.load_fv_offsets()?;
        }
        if self.fvfile.is_none() {
            self.fvfile = Some(BufReader::new(File::open(format!(
                "{}/feature_vectors",
                self.path
            ))?));
        }
        let offset = self.fv_offsets.as_ref().unwrap().get(&intid).unwrap();
        self.fvfile
            .as_mut()
            .unwrap()
            .seek(std::io::SeekFrom::Start(*offset))?;
        let fv = FeatureVec::read_from(self.fvfile.as_mut().unwrap())?;
        Ok(fv)
    }

    pub fn get_fv_iter(&mut self) -> FeatureVecReader {
        if self.fvfile.is_none() {
            self.fvfile = Some(BufReader::new(
                File::open(format!("{}/feature_vectors", self.path)).unwrap(),
            ));
        }
        FeatureVecReader::new(self.fvfile.take().unwrap())
    }

    pub fn get_idf(&mut self, tokid: usize) -> Result<f32, Box<dyn std::error::Error>> {
        if self.idf_values.is_none() {
            self.load_idf_values()?;
        }
        Ok(self.idf_values.as_ref().unwrap()[tokid])
    }

    fn load_vocab(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.vocab.is_some() {
            return Ok(());
        }
        let vocab_path = format!("{}/vocab", self.path);
        let vocab = OnDiskCompressedHash::open(&vocab_path)?;
        self.vocab = Some(vocab);
        Ok(())
    }

    fn load_docinfo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.docids.is_some() {
            return Ok(());
        }
        let docinfob_path = format!("{}/docid_map", self.path);
        let docinfo = OnDiskCompressedHash::open(&docinfob_path)?;
        self.docids = Some(docinfo);
        Ok(())
    }

    fn load_idf_values(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.idf_values.is_some() {
            return Ok(());
        }
        let idf_path = format!("{}/idf", self.path);
        let mut idf_file = BufReader::new(File::open(idf_path)?);
        self.idf_values = Some(decode_from_std_read(
            &mut idf_file,
            bincode::config::standard(),
        )?);
        Ok(())
    }

    fn load_fv_offsets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.fv_offsets.is_some() {
            return Ok(());
        }
        let fv_offsets_path = format!("{}/fv_offsets", self.path);
        let mut fv_offsets_file = File::open(fv_offsets_path)?;
        self.fv_offsets = Some(decode_from_std_read(
            &mut fv_offsets_file,
            bincode::config::standard(),
        )?);
        Ok(())
    }
}
