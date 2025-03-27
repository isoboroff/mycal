use bincode::config::Configuration;
use bincode::{decode_from_std_read, encode_into_std_write, Decode, Encode};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// Define a trait for serializing and deserializing data
pub trait SerializeDeserialize: Sized {
    fn serialize(
        &self,
        writer: &mut impl Write,
        config: Configuration,
    ) -> Result<usize, std::io::Error>;
    fn deserialize(reader: &mut impl Read, config: Configuration) -> Result<Self, std::io::Error>;
}

// Implement SerializeDeserialize for types that implement Encode and Decode
impl<T: Encode + Decode<()>> SerializeDeserialize for T {
    fn serialize(
        &self,
        writer: &mut impl Write,
        config: Configuration,
    ) -> Result<usize, std::io::Error> {
        encode_into_std_write(self, writer, config)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
    }

    fn deserialize(reader: &mut impl Read, config: Configuration) -> Result<Self, std::io::Error> {
        decode_from_std_read(reader, config)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
    }
}

// Define a struct to hold the temporary file names
struct RunFiles {
    files: Vec<String>,
}

// Define the external sorting function
pub fn external_sort<T: SerializeDeserialize + Ord + Clone>(
    input_file: &str,
    output_file: &str,
    buffer_size: usize,
    temp_dir: &str,
    config: bincode::config::Configuration,
) -> std::io::Result<()> {
    // Create a directory for temporary files if it doesn't exist
    std::fs::create_dir_all(temp_dir)?;

    // Divide the data into runs
    let run_files = divide_into_runs::<T>(input_file, buffer_size, temp_dir, config)?;

    // Merge the sorted runs
    merge_runs::<T>(output_file, &run_files, buffer_size, config)?;

    // Remove the temporary run files
    for file in run_files.files {
        std::fs::remove_file(file)?;
    }
    std::fs::remove_dir_all(temp_dir)?;

    Ok(())
}

// Divide the data into runs
fn divide_into_runs<T: SerializeDeserialize + Ord + Clone>(
    input_file: &str,
    buffer_size: usize,
    temp_dir: &str,
    config: bincode::config::Configuration,
) -> std::io::Result<RunFiles> {
    let mut run_files = RunFiles { files: Vec::new() };
    let mut buffer: Vec<T> = Vec::with_capacity(buffer_size);

    let file = File::open(input_file)?;
    let mut reader = BufReader::new(file);

    loop {
        let obj = match T::deserialize(&mut reader, config) {
            Ok(obj) => obj,
            Err(_) => break,
        };

        buffer.push(obj);

        if buffer.len() == buffer_size {
            // Sort the buffer
            buffer.sort();

            // Write the sorted buffer to a temporary file
            let file_name = format!("{}/run_{}.bin", temp_dir, run_files.files.len());
            let mut writer = BufWriter::new(File::create(&file_name)?);
            for obj in &buffer {
                obj.serialize(&mut writer, config)?;
            }
            run_files.files.push(file_name.clone());

            // Clear the buffer
            buffer.clear();
        }
    }

    // Write the remaining buffer to a temporary file
    if !buffer.is_empty() {
        buffer.sort();

        let file_name = format!("{}/run_{}.bin", temp_dir, run_files.files.len());
        let mut writer = BufWriter::new(File::create(&file_name)?);
        for obj in &buffer {
            obj.serialize(&mut writer, config)?;
        }
        run_files.files.push(file_name);
    }

    Ok(run_files)
}

// Merge the sorted runs
fn merge_runs<T: SerializeDeserialize + Ord + Clone>(
    output_file: &str,
    run_files: &RunFiles,
    buffer_size: usize,
    config: bincode::config::Configuration,
) -> std::io::Result<()> {
    let mut files: Vec<BufReader<File>> = Vec::new();
    for f in &run_files.files {
        files.push(BufReader::new(File::open(f)?));
    }

    let mut heap: Vec<(T, usize)> = Vec::with_capacity(buffer_size);
    // to fix error, iterate over indexes? Like below
    for (i, file) in files.iter_mut().enumerate() {
        let obj = match T::deserialize(file, config) {
            Ok(obj) => obj,
            Err(_) => continue,
        };
        heap.push((obj, i));
    }

    let mut writer = BufWriter::new(File::create(output_file)?);

    while !heap.is_empty() {
        // Sort the heap
        heap.sort();

        // Write the smallest object to the output file
        let (obj, file_index) = heap.pop().unwrap();
        obj.serialize(&mut writer, config)?;

        // Read the next object from the file
        let obj = match T::deserialize(&mut files[file_index], config) {
            Ok(obj) => obj,
            Err(_) => continue,
        };
        heap.push((obj, file_index));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::{Decode, Encode};

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Encode, Decode)]
    struct Integer {
        value: i32,
    }

    #[test]
    fn test_external_sort() -> std::io::Result<()> {
        let config = bincode::config::standard();
        external_sort::<Integer>("input.bin", "output.bin", 1000, "temp", config)?;
        Ok(())
    }
}
