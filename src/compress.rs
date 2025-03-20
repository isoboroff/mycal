// ported from Joel McKenzie's immediate_access project
// https://github.com/JMMackenzie/immediate-access/blob/main/compress.hpp

use std::collections::VecDeque;

const MAGIC_F: u32 = 4;

// Estimates how many bytes we need to encode a value
pub fn vbyte_bytes_required(value: u32) -> usize {
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
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
        }
    }
    (value, bytes_read)
}

pub struct VbyteEncodedBuffer {
    pub buffer: Vec<u8>,
    pub index: usize,
}

impl VbyteEncodedBuffer {
    pub fn new(buffer: Vec<u8>) -> VbyteEncodedBuffer {
        VbyteEncodedBuffer { buffer, index: 0 }
    }
    pub fn new_with_capacity(capacity: usize) -> VbyteEncodedBuffer {
        VbyteEncodedBuffer {
            buffer: Vec::with_capacity(capacity),
            index: 0,
        }
    }
    pub fn tell(&self) -> usize {
        self.index
    }
    pub fn seek(&mut self, offset: usize) {
        self.index = offset;
    }
    pub fn read(&mut self) -> Result<u32, &'static str> {
        let mut value: u32 = 0u32;
        let mut shift: usize = 0;
        loop {
            if self.index >= self.buffer.len() {
                return Result::Err("Attempt to read off end of buffer");
            }
            let byte = self.buffer[self.index];
            self.index += 1;
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Result::Ok(value)
    }
    pub fn write(&mut self, value: u32) -> usize {
        let mut write_byte = (value & 0x7f) as u8;
        let mut value = value >> 7;
        let mut bytes_written: usize = 0;
        if self.buffer.len() < self.buffer.capacity() {
            self.buffer.resize(self.buffer.capacity(), 0);
        }
        while value > 0 {
            write_byte = write_byte | 0x80;
            if self.index >= self.buffer.len() {
                self.buffer.resize(self.index * 2, 0);
            }
            self.buffer[self.index] = write_byte;
            self.index += 1;
            bytes_written += 1;
            write_byte = (value & 0x7f) as u8;
            value = value >> 7;
        }
        if self.index >= self.buffer.len() {
            self.buffer.resize(self.index * 2, 0);
        }
        self.buffer[self.index] = write_byte;
        self.index += 1;
        bytes_written + 1
    }
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

pub fn magic_bytes_required(docgap: u32, freq: u32) -> usize {
    let mut bytes: usize = 0;
    if freq < MAGIC_F {
        let magic_value = (docgap - 1) * MAGIC_F + freq;
        bytes += vbyte_bytes_required(magic_value);
    } else {
        let mut magic_value = docgap * MAGIC_F;
        bytes += vbyte_bytes_required(magic_value);
        magic_value = freq - MAGIC_F + 1;
        bytes += vbyte_bytes_required(magic_value);
    }
    bytes
}

pub struct MagicEncodedBuffer {
    pub buffer: VbyteEncodedBuffer,
    pub last_docid: u32,
}

impl MagicEncodedBuffer {
    pub fn new(buffer: VbyteEncodedBuffer) -> MagicEncodedBuffer {
        MagicEncodedBuffer {
            buffer,
            last_docid: 0,
        }
    }
    pub fn new_with_capacity(capacity: usize) -> MagicEncodedBuffer {
        MagicEncodedBuffer {
            buffer: VbyteEncodedBuffer::new_with_capacity(capacity),
            last_docid: 0,
        }
    }
    pub fn from_vec(buffer: Vec<u8>) -> MagicEncodedBuffer {
        MagicEncodedBuffer {
            buffer: VbyteEncodedBuffer::new(buffer),
            last_docid: 0,
        }
    }
    pub fn tell(&self) -> usize {
        self.buffer.index
    }
    pub fn seek(&mut self, index: usize) {
        // BUG need to reset the last_docid
        self.buffer.index = index;
    }
    pub fn reset(&mut self) {
        self.seek(0);
        self.last_docid = 0;
    }
    pub fn byte_slice(&mut self, from: usize, to: usize) -> &[u8] {
        &self.buffer.buffer.as_slice()[from..to]
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer.buffer.as_slice()
    }
    pub fn vbyte_read(&mut self) -> Result<u32, &str> {
        self.buffer.read()
    }
    pub fn read(&mut self) -> (u32, u32) {
        let decoded = match self.buffer.read() {
            Ok(value) => value,
            Err(e) => panic!("Error reading from buffer: {}", e),
        };
        let gap: u32;
        let mut freq: u32;
        if decoded % MAGIC_F > 0 {
            gap = 1 + decoded / MAGIC_F;
            freq = decoded % MAGIC_F;
        } else {
            gap = decoded / MAGIC_F;
            freq = match self.buffer.read() {
                Ok(value) => value,
                Err(e) => panic!("Error reading from buffer: {}", e),
            };
            freq = MAGIC_F + freq - 1;
        }
        self.last_docid += gap;
        (self.last_docid, freq)
    }
    pub fn vbyte_write(&mut self, value: u32) -> usize {
        self.buffer.write(value)
    }
    pub fn write(&mut self, docid: u32, freq: u32) -> usize {
        let mut magic_value;
        let docgap = docid - self.last_docid;
        self.last_docid = docid;
        let mut bytes_written = 0;
        if freq < MAGIC_F {
            magic_value = (docgap - 1) * MAGIC_F + freq;
            bytes_written += self.buffer.write(magic_value);
        } else {
            magic_value = docgap * MAGIC_F;
            bytes_written += self.buffer.write(magic_value);
            magic_value = freq - MAGIC_F + 1;
            bytes_written += self.buffer.write(magic_value);
        }
        bytes_written
    }
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
