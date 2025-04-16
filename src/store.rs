use bincode::{decode_from_std_read, encode_into_std_write};

use crate::index::InvertedFile;
use crate::odch::OnDiskCompressedHash;
use crate::FeatureVec;
use std::fs::File;

pub struct Store {
    pub path: String,
    pub fvfile: Option<File>,
    pub invfile: Option<InvertedFile>,
    pub vocab: Option<OnDiskCompressedHash>,
    pub docids: Option<OnDiskCompressedHash>,
    pub fv_offsets: Option<Vec<u64>>,
    pub idf_values: Option<Vec<f32>>,
}

impl Store {
    pub fn new(path: &str) -> Result<Store, std::io::Error> {
        std::fs::create_dir_all(path)?;
        Ok(Store {
            path: path.to_string(),
            fvfile: Some(File::create(format!("{}/feature_vectors", path))?),
            invfile: Some(InvertedFile::new(&format!("{}/invfile", path))),
            vocab: Some(OnDiskCompressedHash::new()),
            docids: Some(OnDiskCompressedHash::new()),
            fv_offsets: Some(Vec::new()),
            idf_values: Some(Vec::new()),
        })
    }

    pub fn open(path: &str) -> Result<Store, std::io::Error> {
        if !std::fs::exists(path)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find path",
            ));
        }
        Ok(Store {
            path: path.to_string(),
            fvfile: None,
            invfile: None,
            vocab: None,
            docids: None,
            fv_offsets: None,
            idf_values: None,
        })
    }

    pub fn add_token(&mut self, token: &str) -> Result<usize, Box<dyn std::error::Error>> {
        if self.vocab.is_none() {
            self.vocab = Some(OnDiskCompressedHash::new());
        }
        let tokid = self.vocab.as_mut().unwrap().get_or_insert(token);
        Ok(tokid)
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

    pub fn add_docid(&mut self, docid: &str) -> Result<usize, Box<dyn std::error::Error>> {
        if self.docids.is_none() {
            self.docids = Some(OnDiskCompressedHash::new());
        }
        let intid = self.docids.as_mut().unwrap().get_or_insert(docid);
        Ok(intid)
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

    pub fn add_fv_offset(&mut self, offset: u64) -> Result<(), Box<dyn std::error::Error>> {
        if self.fv_offsets.is_none() {
            self.fv_offsets = Some(Vec::new());
        }
        self.fv_offsets.as_mut().unwrap().push(offset);
        Ok(())
    }

    pub fn save_fv(&mut self, fv: FeatureVec) -> Result<(), Box<dyn std::error::Error>> {
        encode_into_std_write(
            fv,
            self.fvfile.as_mut().unwrap(),
            bincode::config::standard(),
        )?;
        Ok(())
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
        let docinfob_path = format!("{}/docinfo", self.path);
        let docinfo = OnDiskCompressedHash::open(&docinfob_path)?;
        self.docids = Some(docinfo);
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

    pub fn save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.vocab
            .as_mut()
            .unwrap()
            .save(&format!("{}/vocab", self.path))?;
        self.docids
            .as_mut()
            .unwrap()
            .save(&format!("{}/docinfo", self.path))?;
        self.invfile.as_mut().unwrap().save()?;
        let fv_offsets_path = format!("{}/fv_offsets", self.path);
        let mut fv_offsets_file = File::create(fv_offsets_path)?;
        encode_into_std_write(
            self.fv_offsets.as_ref().unwrap(),
            &mut fv_offsets_file,
            bincode::config::standard(),
        )?;
        Ok(())
    }
}
