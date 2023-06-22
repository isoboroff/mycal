use porter_stemmer::stem;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

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
    pub fn encode(&self) -> Vec<u8> {
        bincode::serialize(&self.features).unwrap()
    }
    pub fn decode(encoded: &[u8]) -> FeatureVec {
        bincode::deserialize(encoded).unwrap()
    }
}

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
            w: vec![0.0; dimensionality],
            lambda,
            num_iters,
            scale: 1.0,
            squared_norm: 0.0,
        }
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

    fn inner_product(&self, x: &FeatureVec) -> f32 {
        let mut prod = 0.0;
        for (i, _feat) in x.features.iter().enumerate() {
            prod += self.w[x.feature_at(i)] * x.value_at(i);
        }
        prod * self.scale
    }

    fn inner_product_on_difference(&self, a: &FeatureVec, b: &FeatureVec) -> f32 {
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
