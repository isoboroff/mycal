use bincode::{Decode, Encode};

// Right now this is written specifically for NeuCLIR/RAGTIME
// where docids are UUIDs.

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Debug, Encode, Decode)]
pub struct PTuple {
    pub tok: usize,
    pub docid: usize,
    pub count: u32,
}

impl PTuple {
    pub fn new(tok: usize, docid: usize, count: u32) -> PTuple {
        PTuple { tok, docid, count }
    }
}
