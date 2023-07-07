use bincode::Result;
use porter_stemmer::stem;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct DocInfo {
    pub intid: usize,
    pub docid: String,
    pub offset: u64,
}

pub struct DocsDb {
    pub filename: String,
    pub db: sled::Db,
    pub next_intid: usize,

    batch: sled::Batch,
    batch_len: usize,
}

impl DocsDb {
    pub fn open(filename: &str) -> DocsDb {
        let conf = sled::Config::default()
            .path(filename.to_owned())
            .cache_capacity(10_000_000)
            .use_compression(false)
            .mode(sled::Mode::LowSpace);
        let db = conf.open().unwrap();

        DocsDb {
            filename: filename.to_string(),
            db,
            next_intid: 0,
            batch: sled::Batch::default(),
            batch_len: 0,
        }
    }

    pub fn create(filename: &str) -> DocsDb {
        let conf = sled::Config::default()
            .path(filename.to_owned())
            .cache_capacity(10_000_000)
            .use_compression(false)
            .mode(sled::Mode::HighThroughput);
        let db = conf.open().unwrap();

        DocsDb {
            filename: filename.to_string(),
            db,
            next_intid: 0,
            batch: sled::Batch::default(),
            batch_len: 0,
        }
    }

    pub fn get(&self, docid: &str) -> Option<DocInfo> {
        self.db
            .get(docid)
            .unwrap()
            .map(|ivec| bincode::deserialize(&ivec).unwrap())
    }

    pub fn insert(&self, docid: &str, di: &DocInfo) -> Option<sled::IVec> {
        let dib = bincode::serialize(di).unwrap();
        // not happy about the error reporting here...
        self.db.insert(docid, dib).ok().unwrap()
    }

    pub fn insert_batch(&mut self, docid: &str, di: &DocInfo, batch_size: usize) {
        let dib = bincode::serialize(di).unwrap();
        if self.batch_len > batch_size {
            // We need to use swap because apply_batch consumes the batch
            let mut local_batch = sled::Batch::default();
            std::mem::swap(&mut local_batch, &mut self.batch);
            self.db.apply_batch(local_batch).expect("Batch apply fail");
            self.batch_len = 0;
        }
        self.batch.insert(docid, dib);
        self.batch_len += 1;
    }

    pub fn process_remaining(&mut self) {
        if self.batch_len > 0 {
            let mut batch_to_send = sled::Batch::default();
            std::mem::swap(&mut batch_to_send, &mut self.batch);
            self.db
                .apply_batch(batch_to_send)
                .expect("Batch apply fail");
            self.batch_len = 0;
        }
    }

    pub fn insert_iter(
        &mut self,
        library: &Docs,
        stuff: impl Iterator<Item = (String, usize)>,
    ) -> Result<()> {
        stuff.for_each(|(docid, intid)| {
            let di = library.docs.get(intid).unwrap();
            self.insert_batch(&docid, &di, 100_000);
        });
        Ok(())
    }

    pub fn get_intid(&self, docid: &str) -> Option<usize> {
        let tmp_docid = docid.to_string();
        let docinfo = self.db.get(tmp_docid).unwrap();
        match docinfo {
            Some(bytes) => {
                let di: DocInfo = bincode::deserialize(&bytes).unwrap();
                Some(di.intid)
            }
            None => None,
        }
    }

    pub fn add_doc(&mut self, docid: &str) -> Option<usize> {
        let tmp_docid = docid.to_string();
        match self.db.get(&tmp_docid) {
            Ok(di) => match di {
                Some(di) => {
                    let ddi: DocInfo = bincode::deserialize(&di).unwrap();
                    Some(ddi.intid)
                }
                None => None,
            },
            _ => {
                let intid = self.next_intid;
                self.next_intid = intid + 1;
                let new_di = DocInfo {
                    intid,
                    docid: tmp_docid.clone(),
                    offset: 0,
                };
                let sdi = bincode::serialize(&new_di).unwrap();
                self.db
                    .insert(&tmp_docid, sdi)
                    .expect("Could not insert DocInfo into db");
                Some(intid)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Docs {
    pub m: HashMap<String, usize>,
    pub docs: Vec<DocInfo>,
}

impl Docs {
    pub fn new() -> Docs {
        Docs {
            m: HashMap::new(),
            docs: Vec::new(),
        }
    }
    pub fn load(filename: &str) -> Result<Docs> {
        let mut infp = BufReader::new(File::open(filename)?);
        bincode::deserialize_from::<&mut BufReader<File>, Docs>(&mut infp)
    }
    pub fn get_intid(&self, docid: &str) -> Option<&usize> {
        self.m.get(docid)
    }
    pub fn add_doc(&mut self, docid: &str) -> usize {
        if self.m.contains_key(docid) {
            self.m.get(docid).unwrap().to_owned()
        } else {
            let intid = self.docs.len();
            self.m.insert(docid.to_string(), intid);
            self.docs.push(DocInfo {
                docid: docid.to_string(),
                intid,
                offset: 0,
            });
            intid
        }
    }
    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut outfp = BufWriter::new(File::create(filename)?);
        bincode::serialize_into(&mut outfp, self).expect("Error writing dictionary");
        outfp.flush()?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dict {
    pub m: HashMap<String, usize>,
    pub df: HashMap<usize, f32>,
    pub last_tokid: usize,
}

impl Dict {
    pub fn new() -> Dict {
        Dict {
            m: HashMap::new(),
            df: HashMap::new(),
            last_tokid: 0,
        }
    }
    pub fn load(filename: &str) -> Result<Dict> {
        let mut infp = BufReader::new(File::open(filename)?);
        bincode::deserialize_from::<&mut BufReader<File>, Dict>(&mut infp)
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
        bincode::serialize_into(&mut outfp, self).expect("Error writing dictionary");
        outfp.flush()?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeaturePair {
    pub id: usize,
    pub value: f32,
}

#[derive(Debug, Serialize, Deserialize)]
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
    pub fn read_from(fp: &mut BufReader<File>) -> Result<FeatureVec> {
        bincode::deserialize_from::<&mut BufReader<File>, FeatureVec>(fp)
    }
    pub fn write_to(&self, fp: BufWriter<File>) -> Result<()> {
        bincode::serialize_into(fp, self).expect("Error writing FeatureVec");
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
    pub fn normalize(&mut self) {
        let norm = self
            .features
            .iter()
            .map(|fp| fp.value * fp.value)
            .reduce(|acc, e| acc + e)
            .unwrap_or(0.0)
            .sqrt();
        self.squared_norm = norm;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Classifier {
    pub lambda: f32,
    pub num_iters: u32,

    w: Vec<f32>,
    scale: f32,
    squared_norm: f32,
}

impl Classifier {
    pub fn new(dimensionality: usize, lambda: f32, num_iters: u32) -> Classifier {
        Classifier {
            w: vec![0.0; dimensionality + 1],
            lambda,
            num_iters,
            scale: 1.0,
            squared_norm: 0.0,
        }
    }

    pub fn load(filename: &str) -> Result<Classifier> {
        let mut infp = BufReader::new(File::open(filename)?);
        bincode::deserialize_from::<&mut BufReader<File>, Classifier>(&mut infp)
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut outfp = BufWriter::new(File::create(filename)?);
        bincode::serialize_into(&mut outfp, self).expect("Error writing model");
        outfp.flush()?;
        Ok(())
    }

    pub fn train(&mut self, positives: &Vec<FeatureVec>, negatives: &Vec<FeatureVec>) {
        let mut rng = thread_rng();

        for i in 0..self.num_iters {
            let eta = 1.0 / (self.lambda * i as f32);
            let a = positives.choose(&mut rng).unwrap();
            let b = negatives.choose(&mut rng).unwrap();

            let y = 1.0;
            let mut loss = self.inner_product_on_difference(a, b);
            loss *= 1.0 * y;
            loss = loss.exp();
            loss = y / loss;

            // Regularize
            let scaling_factor = 1.0 - (eta * self.lambda);
            self.scale_by(&scaling_factor);

            self.add_vector(&a, &(eta * loss));
            self.add_vector(&b, &(-1.0 * eta * loss));

            // Pegasos projection
            let projection_val = 1.0 / (self.lambda * self.squared_norm).sqrt();
            if projection_val < 1.0 {
                self.scale_by(&projection_val);
            }
        }

        self.scale_to_one();
    }

    pub fn inner_product(&self, x: &FeatureVec) -> f32 {
        let mut prod = 0.0;
        for (i, _feat) in x.features.iter().enumerate() {
            prod += self.w[x.feature_at(i)] * x.value_at(i);
        }
        prod * self.scale
    }

    pub fn inner_product_on_difference(&self, a: &FeatureVec, b: &FeatureVec) -> f32 {
        self.inner_product(a) - self.inner_product(b)
    }

    fn scale_to_one(&mut self) {
        for wt in self.w.iter_mut() {
            *wt *= self.scale;
        }
        self.scale = 1.0;
    }

    const MIN_SCALE: f32 = 0.00000000001;

    fn scale_by(&mut self, scaling_factor: &f32) {
        if self.scale < Self::MIN_SCALE {
            self.scale_to_one();
        }
        self.squared_norm *= *scaling_factor * *scaling_factor;

        if scaling_factor > &0.0 {
            self.scale *= *scaling_factor;
        }
    }

    fn add_vector(&mut self, x: &FeatureVec, x_scale: &f32) {
        let mut inner_product = 0.0;

        for (i, _feat) in x.features.iter().enumerate() {
            let this_x_value = x.value_at(i) * x_scale;
            let this_x_feature = x.feature_at(i);
            inner_product += self.w[this_x_feature] * this_x_value;
            self.w[this_x_feature] += this_x_value / self.scale;
        }

        self.squared_norm +=
            x.squared_norm * x_scale * x_scale + (2.0 * self.scale * inner_product);
    }
}

fn is_alpha(s: &str) -> bool {
    s.chars().all(|c| c.is_alphabetic())
}

// Tokens are stemmed, lowercased sequences of alphanumeric characters
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_lowercase())
        .map(|s| if is_alpha(&s) { stem(&s) } else { s })
        .collect()
}
