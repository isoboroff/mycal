use crate::FeatureVec;
use bincode;
use rand::seq::IndexedRandom;
use serde_derive::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize)]
pub struct Classifier {
    pub lambda: f32,
    pub num_iters: u32,

    pub w: Vec<f32>,
    pub scale: f32,
    pub squared_norm: f32,
}

impl Classifier {
    pub fn new(dimensionality: usize, num_iters: u32) -> Classifier {
        Classifier {
            w: vec![0.0; dimensionality + 1],
            lambda: 0.0001,
            num_iters,
            scale: 1.0,
            squared_norm: 0.0,
        }
    }

    pub fn load(filename: &str) -> Result<Classifier, Box<bincode::ErrorKind>> {
        let mut infp = BufReader::new(File::open(filename)?);
        bincode::deserialize_from::<&mut BufReader<File>, Classifier>(&mut infp)
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut outfp = BufWriter::new(File::create(filename)?);
        bincode::serialize_into(&mut outfp, self).expect("Error writing model");
        outfp.flush()?;
        Ok(())
    }

    const MIN_SCALE: f32 = 0.00000000001;

    pub fn train(&mut self, positives: &Vec<FeatureVec>, negatives: &Vec<FeatureVec>) {
        assert!(!positives.is_empty(), "No positive examples");
        assert!(!negatives.is_empty(), "No negative examples");
        let mut rng = rand::rng();

        for i in 0..self.num_iters {
            let eta = 1.0 / (self.lambda * (i + 1) as f32);
            let a = positives.choose(&mut rng).unwrap();
            let b = negatives.choose(&mut rng).unwrap();

            // let mut loss = self.inner_product_on_difference(a, b);
            // loss *= y;
            // loss = loss.exp();
            // loss = y / (1.0 + loss);
            let y = 1.0;
            let ip = self.inner_product_on_difference(a, b);
            let loss = y / (1.0 + f32::exp(y * ip));
            // println!("ip {:.5} loss {:.5}", ip, loss);

            // Regularize
            let scaling_factor = 1.0 - (eta * self.lambda);
            if scaling_factor > Self::MIN_SCALE {
                self.scale_by(scaling_factor);
            } else {
                self.scale_by(Self::MIN_SCALE);
            }

            if loss != 0.0 {
                self.add_vector(a, eta * loss);
                self.add_vector(b, -1.0 * eta * loss);
            }

            // Pegasos projection
            let projection_val = 1.0 / (self.lambda * self.squared_norm).sqrt();
            if projection_val < 1.0 {
                self.scale_by(projection_val);
            }
        }

        self.scale_to_one();

        let (mut tpos, mut fpos, mut _tneg, mut fneg) = (0, 0, 0, 0);
        for pos in positives.iter() {
            let p = self.inner_product(pos);
            if p > 0.0 {
                tpos += 1
            } else if p <= 0.0 {
                fneg += 1
            }
        }
        for neg in negatives.iter() {
            let p = self.inner_product(neg);
            if p >= 0.0 {
                fpos += 1
            } else if p < 0.0 {
                _tneg += 1
            }
        }
        println!(
            "training precision {:.5}, recall {:.5}",
            tpos as f32 / (tpos + fpos) as f32,
            tpos as f32 / (tpos + fneg) as f32
        );
    }

    pub fn inner_product(&self, x: &FeatureVec) -> f32 {
        let mut prod = 0.0;
        for feat in x.features.iter() {
            prod += self.w[feat.id] * feat.value;
        }
        prod * self.scale
    }

    pub fn inner_product_on_difference(&self, a: &FeatureVec, b: &FeatureVec) -> f32 {
        let mut prod = 0.0;
        prod += self.inner_product(a);
        prod += self.inner_product(b) * -1.0;
        prod
    }

    fn scale_to_one(&mut self) {
        for wt in self.w.iter_mut() {
            *wt *= self.scale;
        }
        self.scale = 1.0;
    }

    fn scale_by(&mut self, scaling_factor: f32) {
        if self.scale < Self::MIN_SCALE {
            self.scale_to_one();
        }
        self.squared_norm *= scaling_factor * scaling_factor;

        if scaling_factor > 0.0 {
            self.scale *= scaling_factor;
        }
    }

    fn add_vector(&mut self, x: &FeatureVec, x_scale: f32) {
        let mut inner_product = 0.0;

        for feat in x.features.iter() {
            let this_x_value = feat.value * x_scale;
            let this_x_feature = feat.id;
            inner_product += self.w[this_x_feature] * this_x_value;
            self.w[this_x_feature] += this_x_value / self.scale;
        }

        self.squared_norm +=
            x.squared_norm * x_scale * x_scale + (2.0 * self.scale * inner_product);
    }
}
