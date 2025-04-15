use bitvec::prelude::*;
use core::sync;
use std::{collections::HashMap, sync::LazyLock};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Uuid {
    pub id: String,
}

static CODER: LazyLock<HuffmanCoder> = LazyLock::new(|| HuffmanCoder::new());

pub type BV = BitVec<u8, LocalBits>;

impl Uuid {
    pub fn new(s: &str) -> Self {
        Uuid { id: s.to_string() }
    }

    pub fn serialize(&mut self) -> Result<BV, String> {
        CODER.encode_uuid(&self.id)
    }

    pub fn deserialize(buf: &mut BV) -> Uuid {
        let uuid_str = CODER.decode(&buf);
        Uuid { id: uuid_str }
    }

    pub fn to_string(&self) -> String {
        self.id.clone()
    }
}

struct HuffmanCoder {
    code_map: HashMap<char, BV>,
    tree: Vec<HuffmanNode>,
}

#[derive(Debug, Clone)]
enum HuffmanNode {
    Internal { zero: usize, one: usize },
    Leaf { symbol: char },
}

impl HuffmanCoder {
    fn new() -> Self {
        let huffman_codes: [(char, BV); 17] = [
            ('0', bitvec![u8, LocalBits; 0, 0, 0, 0]),
            ('1', bitvec![u8, LocalBits; 0, 0, 0, 1]),
            ('2', bitvec![u8, LocalBits; 0, 0, 1, 0]),
            ('3', bitvec![u8, LocalBits; 0, 0, 1, 1]),
            ('4', bitvec![u8, LocalBits; 0, 1, 0]),
            ('5', bitvec![u8, LocalBits; 0, 1, 1, 0]),
            ('6', bitvec![u8, LocalBits; 0, 1, 1, 1]),
            ('7', bitvec![u8, LocalBits; 1, 0, 0, 0]),
            ('8', bitvec![u8, LocalBits; 1, 0, 0, 1]),
            ('9', bitvec![u8, LocalBits; 1, 0, 1, 0, 0]),
            ('a', bitvec![u8, LocalBits; 1, 0, 1, 0, 1]),
            ('b', bitvec![u8, LocalBits; 1, 0, 1, 1, 0]),
            ('c', bitvec![u8, LocalBits; 1, 0, 1, 1, 1]),
            ('d', bitvec![u8, LocalBits; 1, 1, 0, 0, 0]),
            ('e', bitvec![u8, LocalBits; 1, 1, 0, 0, 1]),
            ('f', bitvec![u8, LocalBits; 1, 1, 0, 1, 0]),
            ('-', bitvec![u8, LocalBits; 1, 1, 1]),
        ];

        let tree: Vec<HuffmanNode> = vec![
            HuffmanNode::Internal { zero: 1, one: 2 },   // root
            HuffmanNode::Internal { zero: 3, one: 4 },   // 0
            HuffmanNode::Internal { zero: 5, one: 6 },   // 1
            HuffmanNode::Internal { zero: 7, one: 8 },   // 00
            HuffmanNode::Internal { zero: 9, one: 10 },  // 01
            HuffmanNode::Internal { zero: 11, one: 12 }, // 10
            HuffmanNode::Internal { zero: 13, one: 14 }, // 11
            HuffmanNode::Internal { zero: 15, one: 16 }, // 000
            HuffmanNode::Internal { zero: 17, one: 18 }, // 001
            HuffmanNode::Leaf { symbol: '4' },           // 010
            HuffmanNode::Internal { zero: 19, one: 20 }, // 011
            HuffmanNode::Internal { zero: 21, one: 22 }, // 100
            HuffmanNode::Internal { zero: 23, one: 24 }, // 101
            HuffmanNode::Internal { zero: 25, one: 26 }, // 110
            HuffmanNode::Leaf { symbol: '-' },           // 111
            HuffmanNode::Leaf { symbol: '0' },           // 0000
            HuffmanNode::Leaf { symbol: '1' },           // 0001
            HuffmanNode::Leaf { symbol: '2' },           // 0010
            HuffmanNode::Leaf { symbol: '3' },           // 0011
            HuffmanNode::Leaf { symbol: '5' },           // 0110
            HuffmanNode::Leaf { symbol: '6' },           // 0111
            HuffmanNode::Leaf { symbol: '7' },           // 1000
            HuffmanNode::Leaf { symbol: '8' },           // 1001
            HuffmanNode::Internal { zero: 27, one: 28 }, // 1010
            HuffmanNode::Internal { zero: 29, one: 30 }, // 1011
            HuffmanNode::Internal { zero: 31, one: 32 }, // 1100
            HuffmanNode::Internal { zero: 33, one: 34 }, // 1101
            HuffmanNode::Leaf { symbol: '9' },           // 10100
            HuffmanNode::Leaf { symbol: 'a' },           // 10101
            HuffmanNode::Leaf { symbol: 'b' },           // 10110
            HuffmanNode::Leaf { symbol: 'c' },           // 10111
            HuffmanNode::Leaf { symbol: 'd' },           // 11000
            HuffmanNode::Leaf { symbol: 'e' },           // 11001
            HuffmanNode::Leaf { symbol: 'f' },           // 11010
        ];

        let mut code_map: HashMap<char, BV> = HashMap::new();
        for (c, code) in huffman_codes {
            code_map.insert(c, code.clone());
        }

        HuffmanCoder {
            code_map: code_map,
            tree: tree,
        }
    }

    fn encode_uuid(&self, uuid: &str) -> Result<BV, String> {
        let mut encoded_bits = BitVec::new();
        //println!("Encoding {:?}", uuid);
        for ch in uuid.chars() {
            if let Some(code) = self.code_map.get(&ch) {
                //println!("{} {:?}", ch, code);
                // Convert the code string to a bit vector and append to encoded_bits
                encoded_bits.extend_from_bitslice(&code);
            } else {
                return Err(format!("Character '{}' not in Huffman code map", ch));
            }
        }
        //println!("Encoded: {:?}", encoded_bits);
        Ok(encoded_bits)
    }

    // This decoder uses the Huffman tree, and can decode a UUID in 20ms.
    fn decode(&self, bits: &BV) -> String {
        let mut decoded_string = String::new();
        let mut index: usize = 0;
        let mut codestr = String::new();

        let mut bitter = bits.iter();
        let mut bit = bitter.next().unwrap();
        loop {
            match self.tree[index] {
                HuffmanNode::Internal { zero, one } => {
                    if *bit {
                        index = one;
                        codestr.push('1');
                    } else {
                        index = zero;
                        codestr.push('0');
                    }
                    //println!("codestr {}", codestr);
                    bit = match bitter.next() {
                        Some(b) => b,
                        None => {
                            //println!("no next bit");
                            if let HuffmanNode::Leaf { symbol } = self.tree[index] {
                                decoded_string.push(symbol);
                            }
                            break;
                        }
                    }
                }
                HuffmanNode::Leaf { symbol } => {
                    codestr.clear();
                    //println!("symbol {}", symbol);
                    decoded_string.push(symbol);
                    if decoded_string.len() == 36 {
                        //println!("len is {}", decoded_string.len());
                        break;
                    }
                    index = 0;
                }
            }
        }
        decoded_string
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::{order::Msb0, vec::BitVec};
    use std::{
        error,
        fs::File,
        io::{self, Write},
        time::Duration,
    };

    #[test]
    fn test_encode() -> Result<(), Box<dyn error::Error>> {
        // Example UUID
        let uuid = "123e4567-e89b-12d3-a456-426614174000";
        let enc = HuffmanCoder::new();

        // Encode the UUID
        let encoded_bits = enc.encode_uuid(uuid).expect("Failed to encode UUID");

        // Write encoded bits to a file
        let mut file = File::create("encoded_uuid.bin")?;
        file.write_all(encoded_bits.as_raw_slice())?;

        // Read encoded bits from a file
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .open("encoded_uuid.bin")?;
        let mut encoded_bits_read = BV::new();
        let _bytes = io::copy(&mut file, &mut encoded_bits_read)?;

        // Decode the UUID
        let decode = enc.decode(&encoded_bits_read);
        println!("DECODE IS {}", decode);
        assert!(decode == uuid);

        Ok(())
    }

    #[test]
    fn test_encode_lots() -> Result<(), Box<dyn error::Error>> {
        use std::time::Instant;
        use uuid::Uuid as TheUuid;

        let mut bytes_used = 0.0;
        let mut enc_time_used = Duration::new(0, 0);
        let mut dec_time_used = Duration::new(0, 0);
        let enc = HuffmanCoder::new();

        for _ in 0..10000 {
            let uuid = TheUuid::new_v4();
            let t0 = Instant::now();
            let encoded_bits = enc
                .encode_uuid(&uuid.to_string())
                .expect("Failed to encode UUID");
            enc_time_used += t0.elapsed();

            bytes_used += encoded_bits.len() as f32 / 8.0;

            let t0 = Instant::now();
            let s = enc.decode(&encoded_bits);
            dec_time_used += t0.elapsed();

            assert_eq!(s, uuid.to_string());
        }

        println!("Bytes used per UUID: {}", bytes_used / 10000.0);
        println!(
            "Encoding time per UUID: {:?}",
            enc_time_used.div_f32(10000.0)
        );
        println!(
            "Decoding time per UUID: {:?}",
            dec_time_used.div_f32(10000.0)
        );

        Ok(())
    }
}
