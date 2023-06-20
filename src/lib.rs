use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct FeaturePair {
    pub id: usize,
    pub value: f32,
}

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
}

pub struct Classifier {
    pub lambda: f32,
    pub num_iters: u32,
}

fn inner_product(x: &FeatureVec, w: &Vec<f32>, scale: &f32) -> f32 {
    let mut prod = 0.0;
    for (i, _feat) in x.features.iter().enumerate() {
        prod += w[x.feature_at(i)] * x.value_at(i);
    }
    prod * scale
}

fn inner_product_on_difference(a: &FeatureVec, b: &FeatureVec, w: &Vec<f32>, scale: &f32) -> f32 {
    inner_product(a, w, scale) - inner_product(b, w, scale)
}

fn scale_to_one(weights: &mut Vec<f32>, scale: &mut f32) {
    for w in weights {
        *w *= *scale;
    }
    *scale = 1.0;
}

const MIN_SCALE: f32 = 0.00000000001;

fn scale_by(w: &mut Vec<f32>, scaling_factor: &f32, scale: &mut f32, squared_norm: &mut f32) {
    if *scale < MIN_SCALE {
        scale_to_one(w, scale);
    }
    *squared_norm *= *scaling_factor * *scaling_factor;

    if scaling_factor > &0.0 {
        *scale *= *scaling_factor;
    }
}

fn add_vector(
    x: &FeatureVec,
    x_scale: &f32,
    w: &mut Vec<f32>,
    scale: &f32,
    squared_norm: &mut f32,
) {
    let mut inner_product = 0.0;

    for (i, _feat) in x.features.iter().enumerate() {
        let this_x_value = x.value_at(i) * x_scale;
        let this_x_feature = x.feature_at(i);
        inner_product += w[this_x_feature] * this_x_value;
        w[this_x_feature] += this_x_value / scale;
    }

    *squared_norm += x.squared_norm * x_scale * x_scale + (2.0 * scale * inner_product);
}

impl Classifier {
    pub fn train(
        &self,
        positives: &Vec<FeatureVec>,
        negatives: &Vec<FeatureVec>,
        dimensionality: usize,
    ) -> Vec<f32> {
        let mut w = vec![0.0; dimensionality];
        let mut scale = 1.0;
        let mut squared_norm = 0.0;
        let mut rng = thread_rng();

        for i in 0..self.num_iters {
            let eta = 1.0 / (self.lambda * i as f32);
            let a = positives.choose(&mut rng).unwrap();
            let b = negatives.choose(&mut rng).unwrap();

            let y = 1.0;
            let mut loss = inner_product_on_difference(&a, &b, &w, &scale);
            loss = 1.0 * y * loss;
            loss = loss.exp();
            loss = y / loss;

            // Regularize
            let scaling_factor = 1.0 - (eta * self.lambda);
            scale_by(&mut w, &scaling_factor, &mut scale, &mut squared_norm);

            add_vector(&a, &(eta * loss), &mut w, &scale, &mut squared_norm);
            add_vector(&b, &(-1.0 * eta * loss), &mut w, &scale, &mut squared_norm);

            // Pegasos projection
            let projection_val = 1.0 / (self.lambda * squared_norm).sqrt();
            if projection_val < 1.0 {
                scale_by(&mut w, &projection_val, &mut scale, &mut squared_norm);
            }
        }

        scale_to_one(&mut w, &mut scale);
        w
    }
}
