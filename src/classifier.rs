use crate::FeatureVec;
use bincode;
use bincode::config::Configuration;
use bincode::error::DecodeError;
use rand::seq::IndexedRandom;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

pub struct Classifier {
    pub lambda: f32,
    pub num_iters: u32,

    pub w: Vec<f32>,
    pub scale: f32,
    pub squared_norm: f32,

    config: bincode::config::Configuration,
}

// The impls for Encode and Decode so we can serialize the Classifier
impl bincode::Encode for Classifier {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.lambda, encoder)?;
        bincode::Encode::encode(&self.num_iters, encoder)?;
        bincode::Encode::encode(&self.w, encoder)?;
        bincode::Encode::encode(&self.scale, encoder)?;
        bincode::Encode::encode(&self.squared_norm, encoder)?;
        Ok(())
    }
}

impl<Context> bincode::Decode<Context> for Classifier {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            lambda: bincode::Decode::decode(decoder)?,
            num_iters: bincode::Decode::decode(decoder)?,
            w: bincode::Decode::decode(decoder)?,
            scale: bincode::Decode::decode(decoder)?,
            squared_norm: bincode::Decode::decode(decoder)?,
            config: bincode::config::standard(),
        })
    }
}
impl<'de, Context> bincode::BorrowDecode<'de, Context> for Classifier {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            lambda: bincode::Decode::decode(decoder)?,
            num_iters: bincode::Decode::decode(decoder)?,
            w: bincode::Decode::decode(decoder)?,
            scale: bincode::Decode::decode(decoder)?,
            squared_norm: bincode::Decode::decode(decoder)?,
            config: bincode::config::standard(),
        })
    }
}

impl Classifier {
    pub fn new(dimensionality: usize, num_iters: u32) -> Classifier {
        Classifier {
            w: vec![0.0; dimensionality + 1],
            lambda: 0.0001,
            num_iters,
            scale: 1.0,
            squared_norm: 0.0,
            config: bincode::config::standard(),
        }
    }

    pub fn load(filename: &str, config: Configuration) -> Result<Classifier, DecodeError> {
        let mut infp = BufReader::new(File::open(filename).expect("Can't open classifier file"));
        bincode::decode_from_std_read(&mut infp, config)
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut outfp = BufWriter::new(File::create(filename)?);
        bincode::encode_into_std_write(self, &mut outfp, self.config).expect("Error writing model");
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
