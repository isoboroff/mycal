use flate2::read::GzDecoder;
use std::ffi::OsStr;
use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

/// Read normal or compressed files seamlessly
/// Uses the presence of a `.gz` extension to decide
/// from https://users.rust-lang.org/t/write-to-normal-or-gzip-file-transparently/35561/2
pub fn reader(filename: &str) -> Box<dyn BufRead> {
    let path = Path::new(filename);
    let file = match fs::File::open(path) {
        Err(why) => panic!("couldn't open {}: {:?}", path.display(), why),
        Ok(file) => file,
    };

    if path.extension() == Some(OsStr::new("gz")) {
        Box::new(BufReader::with_capacity(128 * 1024, GzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(128 * 1024, file))
    }
}
