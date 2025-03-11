// ported from Joel McKenzie's immediate_access project
// https://github.com/JMMackenzie/immediate-access/blob/main/compress.hpp

use std::collections::VecDeque;

const MAGIC_F: u32 = 4;

// Estimates how many bytes we need to encode a value
pub fn bytes_required(value: u32) -> usize {
    if value < (1 << 7) {
        return 1;
    } else if value < (1 << 14) {
        return 2;
    } else if value < (1 << 21) {
        return 3;
    } else if value < (1 << 28) {
        return 4;
    }
    return 5;
}

// Vbyte encodes value into buffer and returns the number of bytes consumed
pub fn vbyte_encode(value: u32, buffer: &mut VecDeque<u8>) -> usize {
    let mut written = 0;
    let mut write_byte = (value & 0x7f) as u8;
    let mut value = value >> 7;
    while value > 0 {
        write_byte = write_byte | 0x80;
        buffer.push_back(write_byte);
        written += 1;
        write_byte = (value & 0x7f) as u8;
        value = value >> 7;
    }
    buffer.push_back(write_byte);
    written += 1;
    return written;
}

pub fn vbyte_decode(buffer: &mut VecDeque<u8>) -> (u32, usize) {
    let mut value: u32 = 0u32;
    let mut shift: usize = 0;
    let mut bytes_read: usize = 0;

    loop {
        match buffer.pop_front() {
            None => break,
            Some(byte) => {
                bytes_read += 1;
                value |= ((byte & 0x7f) as u32) << shift;
                println!("byte {}, value {}", byte, value);
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
        }
    }
    (value, bytes_read)
}

// This is the "Double-VByte" encoder
// See Algorithm 2 in the paper
pub fn encode_magic(docgap: u32, freq: u32, buffer: &mut VecDeque<u8>) -> usize {
    let mut magic_value;
    let mut bytes = 0;
    if freq < MAGIC_F {
        magic_value = (docgap - 1) * MAGIC_F + freq;
        bytes += vbyte_encode(magic_value, buffer);
    } else {
        magic_value = docgap * MAGIC_F;
        bytes += vbyte_encode(magic_value, buffer);
        magic_value = freq - MAGIC_F + 1;
        bytes += vbyte_encode(magic_value, buffer);
    }
    return bytes;
}

// This is the "Double-VByte" decoder
// See Algorithm 2
pub fn decode_magic(buffer: &mut VecDeque<u8>) -> (u32, u32) {
    let (decoded, mut _bytes_read) = vbyte_decode(buffer);
    let docgap: u32;
    let mut freq: u32;
    if decoded % MAGIC_F > 0 {
        docgap = 1 + decoded / MAGIC_F;
        freq = decoded % MAGIC_F;
    } else {
        docgap = decoded / MAGIC_F;
        (freq, _bytes_read) = vbyte_decode(buffer);
        freq = MAGIC_F + freq - 1;
    }
    (docgap, freq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vbyte() {
        let victim = 12345678;
        let mut buffer: VecDeque<u8> = VecDeque::new();
        vbyte_encode(victim, &mut buffer);
        println!("Value {} encoded {:?}", victim, &buffer);
        let (decoded, _) = vbyte_decode(&mut buffer);
        println!("Value {} decoded {:?}", victim, decoded);
        assert!(victim == decoded);
    }

    #[test]
    fn test_magic() {
        let doc_id = 1234;
        let freq = 5;
        let mut buffer: VecDeque<u8> = VecDeque::new();
        let bytes = encode_magic(doc_id, freq, &mut buffer);
        println!(
            "Docid {} Freq {} encoded {:?} bytes {}",
            doc_id, freq, &buffer, bytes
        );
        let (decoded_doc_id, decoded_freq) = decode_magic(&mut buffer);
        println!("Docid {} Freq{}", decoded_doc_id, decoded_freq);
        assert!(doc_id == decoded_doc_id);
        assert!(freq == decoded_freq);
    }
}
