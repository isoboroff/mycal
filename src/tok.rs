use porter_stemmer::stem;
use rust_tokenizers::tokenizer::Tokenizer as RustTokenizer;
use rust_tokenizers::tokenizer::XLMRobertaTokenizer;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use crate::Dict;

pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn name(&self) -> String;
}

fn is_alpha(s: &str) -> bool {
    s.chars().all(|c| c.is_alphabetic())
}

// A tokenizer that assumes English.  It splits on whitespace and stems using the Porter stemmer.
#[derive(Copy, Clone)]
pub struct EnglishStemLowercase;

impl Tokenizer for EnglishStemLowercase {
    fn name(&self) -> String {
        "englishstemlower".to_string()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2)
            .map(|s| {
                if is_alpha(&s) {
                    stem(&s)
                } else {
                    s.to_string()
                }
            })
            .collect()
    }
}

#[derive(Copy, Clone)]
pub struct NGrams {
    n: usize,
}

impl Tokenizer for NGrams {
    fn name(&self) -> String {
        format!("ngram.{}", self.n)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = text.to_lowercase().nfkc().collect::<String>();

        normalized
            .graphemes(true)
            .collect::<Vec<_>>()
            .windows(self.n)
            .map(|w| w.join(""))
            .collect()
    }
}

#[derive(Copy, Clone)]
pub struct NGramsHashed {
    n: usize,
    hmod: u32,
}

impl Tokenizer for NGramsHashed {
    fn name(&self) -> String {
        format!("ngramhashed.{}.{}", self.n, self.hmod)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = text.to_lowercase().nfkc().collect::<String>();

        normalized
            .graphemes(true)
            .collect::<Vec<_>>()
            .windows(self.n)
            .map(|w| w.join(""))
            .map(|t| self.hash(t.as_bytes() as &[u8]))
            .map(|h| h.to_string())
            .collect()
    }
}

impl NGramsHashed {
    pub fn new(n: usize, hmod: u32) -> Self {
        let mut prime_hmod = hmod;
        if prime_hmod % 2 == 0 {
            prime_hmod += 1;
        }
        while !NGramsHashed::is_prime(prime_hmod as u64) {
            prime_hmod += 2;
        }
        NGramsHashed {
            n,
            hmod: prime_hmod,
        }
    }

    pub fn is_prime(n: u64) -> bool {
        if n < 4 {
            n > 1
        } else if n % 2 == 0 || n % 3 == 0 {
            false
        } else {
            let max_p = (n as f64).sqrt().ceil() as u64;
            match (5..=max_p)
                .step_by(6)
                .find(|p| n % p == 0 || n % (p + 2) == 0)
            {
                Some(_) => false,
                None => true,
            }
        }
    }

    // A djb variant via http://www.cse.yorku.ca/~oz/hash.html
    pub fn hash(&self, str: &[u8]) -> u64 {
        let mut hash: u64 = 5381;

        for &c in str {
            hash = (hash << 5) ^ (hash >> 2) ^ (c as u64);
        }

        hash % self.hmod as u64
    }
}

pub struct XLMR {
    intok: XLMRobertaTokenizer,
}

impl XLMR {
    fn new(vocab_path: &str) -> Self {
        let intok = XLMRobertaTokenizer::from_file(vocab_path, true).unwrap();
        Self { intok }
    }
    // "/Users/soboroff/mycal-project/mycal/sentencepiece.bpe.model",
}

impl Tokenizer for XLMR {
    fn name(&self) -> String {
        "XLMR".to_string()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.intok.tokenize(text)
    }
}

pub fn get_tokenizer(which: &str) -> Box<dyn Tokenizer> {
    match which {
        "englishstemlower" => Box::new(EnglishStemLowercase),
        s if s.starts_with("ngram.") => {
            let n = s.split('.').nth(1).unwrap().parse::<usize>().unwrap();
            Box::new(NGrams { n })
        }
        s if s.starts_with("nghash.") => {
            let n = s.split('.').nth(1).unwrap().parse::<usize>().unwrap();
            let hmod = s.split('.').nth(2).unwrap().parse::<u32>().unwrap();
            Box::new(NGramsHashed::new(n, hmod))
        }
        "xlmr" => Box::new(XLMR::new(
            "/Users/soboroff/mycal-project/mycal/sentencepiece.bpe.model",
        )),

        _ => Box::new(EnglishStemLowercase),
    }
}

// Tokens are stemmed, lowercased sequences of alphanumeric characters
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_lowercase())
        .map(|s| if is_alpha(&s) { stem(&s) } else { s })
        .collect()
}

pub fn tokenize_and_map(
    docmap: serde_json::Map<String, serde_json::Value>,
    tokenizer: &Box<dyn Tokenizer>,
    dict: &mut Dict,
    docid_field: &String,
    text_field: &String,
) -> (String, HashMap<usize, i32>) {
    let mut m = HashMap::new();
    let docid = match docmap.contains_key(docid_field) {
        true => docmap[docid_field].as_str().unwrap(),
        false => panic!(
            "Document does not contain a {} field for the docid (use -d option?)",
            docid_field
        ),
    };
    let tokens = match docmap.contains_key(text_field) {
        true => tokenizer.tokenize(docmap[text_field].as_str().unwrap()),
        false => panic!(
            "Document does not contain a {} field for the text (use -t option?)",
            text_field
        ),
    };

    for x in tokens {
        let tokid = dict.add_tok(x.to_owned());
        if !m.contains_key(&tokid) {
            dict.incr_df(tokid);
        }
        *m.entry(tokid).or_insert(0) += 1;
    }

    (docid.to_owned(), m)
}
