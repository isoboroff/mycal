use bincode::decode_from_std_read;
use serde::{Deserialize, Serialize};

use crate::index::InvertedFile;
use crate::odch::OnDiskCompressedHash;
use crate::FeatureVec;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub num_docs: usize,
    pub num_features: usize,
}

pub struct Store {
    pub path: String,
    pub fvfile: Option<File>,
    pub invfile: Option<InvertedFile>,
    pub vocab: Option<OnDiskCompressedHash>,
    pub docids: Option<OnDiskCompressedHash>,
    pub fv_offsets: Option<HashMap<usize, u64>>,
    pub idf_values: Option<Vec<f32>>,
    pub config: Config,
}

impl Store {
    pub fn new(path: &str) -> Result<Store, std::io::Error> {
        std::fs::create_dir_all(path)?;
        Ok(Store {
            path: path.to_string(),
            fvfile: Some(File::create(format!("{}/feature_vectors", path))?),
            invfile: Some(InvertedFile::new(&format!("{}/inverted_file", path))),
            vocab: Some(OnDiskCompressedHash::new()),
            docids: Some(OnDiskCompressedHash::new()),
            fv_offsets: Some(HashMap::new()),
            idf_values: Some(Vec::new()),
            config: Config {
                num_docs: 0,
                num_features: 0,
            },
        })
    }

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

    pub fn get_fv(&mut self, intid: usize) -> Result<FeatureVec, Box<dyn std::error::Error>> {
        if self.fv_offsets.is_none() {
            self.load_fv_offsets()?;
        }
        if self.fvfile.is_none() {
            self.fvfile = Some(File::open(format!("{}/feature_vectors", self.path))?);
        }
        let offset = self.fv_offsets.as_ref().unwrap().get(&intid).unwrap();
        let mut fv_reader = BufReader::new(self.fvfile.as_mut().unwrap());
        fv_reader.seek(std::io::SeekFrom::Start(*offset))?;
        let fv = FeatureVec::read_from(&mut fv_reader)?;
        Ok(fv)
    }

    fn load_vocab(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.vocab.is_some() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let vocab_path = format!("{}/vocab", self.path);
        let vocab = OnDiskCompressedHash::open(&vocab_path)?;
        println!("Loaded vocab in {:?}", now.elapsed());
        self.vocab = Some(vocab);
        Ok(())
    }

    fn load_docinfo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.docids.is_some() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let docinfob_path = format!("{}/docid_map", self.path);
        let docinfo = OnDiskCompressedHash::open(&docinfob_path)?;
        println!("Loaded docinfo in {:?}", now.elapsed());
        self.docids = Some(docinfo);
        Ok(())
    }

    fn load_fv_offsets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.fv_offsets.is_some() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let fv_offsets_path = format!("{}/fv_offsets", self.path);
        let mut fv_offsets_file = File::open(fv_offsets_path)?;
        self.fv_offsets = Some(decode_from_std_read(
            &mut fv_offsets_file,
            bincode::config::standard(),
        )?);
        println!("Loaded fv_offsets in {:?}", now.elapsed());
        Ok(())
    }
}
