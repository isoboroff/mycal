//! An external sort crate based on bincode serialization and
//! deserialization.  As long as the underlying object being
//! sorted implements Clone, Encode, Decode, PartialEq, Eq, Ord,
//! PartialOrd, it can be sorted using this crate.
//!
//! Individual chunks are sorted as vectors, then all the chunks
//! are merged into a single sorted file.
//!
//! Bugs
//! - the final merge could be done faster using a real data
//! structure and not just sorting the head items each time.

use bincode::{Decode, Encode};
use kdam::{tqdm, BarExt};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

// Define a struct to hold the temporary file names
struct RunFiles {
    files: Vec<String>,
    items: u32,
}

/// The main entry point.
/// input_file and output_file are pathnames.  The input_file is the file to be
/// sorted, and output_file is the final sorted file.
/// buffer_size is the number of objects to be read for each chunk.
/// temp_dir is where sorted sub-files are stored.
/// config is a bincode::config::Configuration object
pub fn external_sort<T, R, W>(
    input_file: &str,
    output_file: &str,
    buffer_size: usize,
    temp_dir: &str,
) -> std::io::Result<()>
where
    T: Encode + Decode<()> + Ord + Clone + Send,
    R: BufRead + Sized,
    W: Write + Sized,
{
    let mut input_fp: BufReader<File> = BufReader::new(File::open(input_file)?);
    let mut output_fp: BufWriter<File> = BufWriter::new(File::create(output_file)?);
    external_sort_from::<T, _, _>(&mut input_fp, &mut output_fp, buffer_size, temp_dir)?;
    Ok(())
}

pub fn external_sort_from<T, R, W>(
    input_reader: &mut R,
    output_writer: &mut W,
    buffer_size: usize,
    temp_dir: &str,
) -> std::io::Result<()>
where
    T: Encode + Decode<()> + Ord + Clone + Send,
    R: BufRead + Sized,
    W: Write + Sized,
{
    // Create a directory for temporary files if it doesn't exist
    std::fs::create_dir_all(temp_dir)?;

    // Divide the data into runs
    println!("dividing into sorted runs");
    let run_files = divide_into_runs::<T, R>(input_reader, buffer_size, temp_dir)?;

    // Merge the sorted runs
    println!("merging runs");
    merge_runs::<T, W>(output_writer, &run_files)?;

    // Remove the temporary run files
    for file in run_files.files {
        std::fs::remove_file(file)?;
    }
    std::fs::remove_dir_all(temp_dir)?;
    output_writer.flush()?;
    Ok(())
}

// Divide the data into runs
fn divide_into_runs<T, R>(
    input_reader: &mut R,
    buffer_size: usize,
    temp_dir: &str,
) -> std::io::Result<RunFiles>
where
    T: Encode + Decode<()> + Ord + Clone + Send,
    R: Read + Sized,
{
    let mut run_files = RunFiles {
        files: Vec::new(),
        items: 0,
    };
    let mut buffer: Vec<T> = Vec::with_capacity(buffer_size);
    let config = bincode::config::standard();

    loop {
        let obj = match bincode::decode_from_std_read(input_reader, config) {
            Ok(obj) => obj,
            Err(_) => break,
        };
        run_files.items += 1;

        buffer.push(obj);

        if buffer.len() == buffer_size {
            // Sort the buffer
            buffer.par_sort();

            // Write the sorted buffer to a temporary file
            let file_name = format!("{}/run_{}.bin", temp_dir, run_files.files.len());
            let mut writer = BufWriter::new(File::create(&file_name)?);
            for obj in &buffer {
                bincode::encode_into_std_write(obj, &mut writer, config)
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
            }

            run_files.files.push(file_name.clone());
            writer.flush()?;

            // Clear the buffer
            buffer.clear();
        }
    }
    // Write the remaining buffer to a temporary file
    if !buffer.is_empty() {
        buffer.par_sort();

        let file_name = format!("{}/run_{}.bin", temp_dir, run_files.files.len());
        let mut writer = BufWriter::new(File::create(&file_name)?);
        for obj in &buffer {
            bincode::encode_into_std_write(obj, &mut writer, config)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
        }
        run_files.files.push(file_name);
        writer.flush()?;

        println!(
            "{} chunks of max {} items each, {} items total",
            run_files.files.len(),
            buffer_size,
            run_files.items,
        );
    }

    Ok(run_files)
}

// This struct holds an item (like a PTuple), and a file number we read
// it from.  By keeping a heap of these sorted on the item, we can
// merge the files using a priority queue.
struct ObjAndFileIndex<T>
where
    T: Ord,
{
    obj: T,
    i: usize,
}
impl<T> Ord for ObjAndFileIndex<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.obj.cmp(&self.obj)
    }
}
impl<T> PartialOrd for ObjAndFileIndex<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.obj.cmp(&self.obj))
    }
}
impl<T> PartialEq for ObjAndFileIndex<T>
where
    T: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj
    }
}
impl<T> Eq for ObjAndFileIndex<T> where T: Ord {}

// Merge the sorted runs
fn merge_runs<T, W>(output_writer: &mut W, run_files: &RunFiles) -> std::io::Result<()>
where
    T: Encode + Decode<()> + Ord + Clone + Send,
    W: Write + Sized,
{
    let mut files: Vec<BufReader<File>> = Vec::new();
    for f in &run_files.files {
        files.push(BufReader::new(File::open(f)?));
    }
    let config = bincode::config::standard();

    println!("merge setup");
    let mut heap: BinaryHeap<ObjAndFileIndex<T>> = BinaryHeap::with_capacity(files.len());
    for (i, file) in files.iter_mut().enumerate() {
        let obj = bincode::decode_from_std_read(file, config)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
        heap.push(ObjAndFileIndex { obj, i });
    }

    let mut bar = tqdm!(desc = "Merging", total = run_files.items as usize);
    while !heap.is_empty() {
        bar.update(1)?;
        // Write the smallest object to the output file
        let oi = heap.pop().unwrap();
        bincode::encode_into_std_write(oi.obj, output_writer, config)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
        output_writer.flush()?;

        // Read the next object from the file
        let obj = match bincode::decode_from_std_read(&mut files[oi.i], config) {
            Ok(obj) => obj,
            Err(_) => continue,
        };
        heap.push(ObjAndFileIndex { obj, i: oi.i });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, BufWriter, Write},
    };

    use super::external_sort_from;
    use bincode::{Decode, Encode};
    use rand::distr::{Alphanumeric, SampleString};

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Encode, Decode)]
    struct Integer {
        value: u32,
        another: i32,
        label: String,
    }

    #[test]
    fn test_external_sort() -> std::io::Result<()> {
        let ints = Vec::from_iter((0..10000).map(|_i| Integer {
            value: rand::random::<u32>() + 1,
            another: rand::random::<i32>() + 1,
            label: Alphanumeric.sample_string(&mut rand::rng(), rand::random_range(10..15)),
        }));
        let config = bincode::config::standard();
        let mut writer = BufWriter::new(File::create("input.bin")?);
        for int in &ints {
            bincode::encode_into_std_write(int, &mut writer, config)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
        }
        writer.flush()?;

        println!("Unsorted ints");
        let mut foo_reader = BufReader::new(File::open("input.bin")?);
        while let Ok(foo) = bincode::decode_from_std_read::<Integer, _, _>(&mut foo_reader, config)
        {
            println!("{:?}", foo);
        }

        let mut reader = BufReader::new(File::open("input.bin")?);
        let mut writer = BufWriter::new(File::create("output.bin")?);
        external_sort_from::<Integer, _, _>(&mut reader, &mut writer, 1000, "temp")?;
        writer.flush()?;

        let mut reader = BufReader::new(File::open("output.bin")?);
        let mut last_int = 0;
        let mut i: u32 = 0;
        println!("Decoding sorted ints");
        while let Ok(int) = bincode::decode_from_std_read::<Integer, _, _>(&mut reader, config) {
            println!("{:?}", int);
            assert!(
                int.value >= last_int,
                "{} was not >= {}",
                int.value,
                last_int
            );
            last_int = int.value;
            i += 1;
        }
        assert_eq!(i, 10000);
        Ok(())
    }
}
