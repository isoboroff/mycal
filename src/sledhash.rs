extern crate bincode;
extern crate sled;

use std::marker::PhantomData;

use bincode::{config::Configuration, error::DecodeError, Decode, Encode};
use sled::{Batch, Db};

pub struct SledHash<K, V> {
    db: Db,
    binconf: bincode::config::Configuration,
    key_type: PhantomData<K>,
    value_type: PhantomData<V>,
}

impl<K: Encode + Decode<()>, V: Encode + Decode<()>> SledHash<K, V> {
    pub fn new(filename: &str) -> Result<SledHash<K, V>, sled::Error> {
        let conf = sled::Config::default()
            .path(filename)
            .cache_capacity(10_000_000_000)
            .mode(sled::Mode::HighThroughput)
            .flush_every_ms(Some(1000));

        Ok(SledHash::<K, V> {
            db: conf.open()?,
            binconf: bincode::config::standard(),
            key_type: PhantomData::<K>,
            value_type: PhantomData::<V>,
        })
    }

    pub fn open(filename: &str) -> Result<SledHash<K, V>, sled::Error> {
        let conf = sled::Config::default()
            .path(filename)
            .cache_capacity(10_000_000_000)
            .mode(sled::Mode::LowSpace)
            .flush_every_ms(Some(1000));

        Ok(SledHash::<K, V> {
            db: conf.open()?,
            binconf: bincode::config::standard(),
            key_type: PhantomData::<K>,
            value_type: PhantomData::<V>,
        })
    }

    pub fn insert(&self, key: &K, value: &V) -> Result<(), sled::Error> {
        let key_bytes = bincode::encode_to_vec(key, self.binconf)
            .map_err(|err| sled::Error::Unsupported(err.to_string()))?;
        let value_bytes = bincode::encode_to_vec(value, self.binconf)
            .map_err(|err| sled::Error::Unsupported(err.to_string()))?;
        self.db.insert(key_bytes, value_bytes)?;
        Ok(())
    }

    pub fn start_batch(&self) -> Result<sled::Batch, sled::Error> {
        Ok(Batch::default())
    }

    pub fn apply_batch(&self, batch: sled::Batch) -> Result<(), sled::Error> {
        self.db.apply_batch(batch)?;
        Ok(())
    }

    pub fn get(&self, key: &K) -> Result<Option<V>, sled::Error> {
        let key_bytes = bincode::encode_to_vec(key, self.binconf).unwrap();
        let value_bytes = self.db.get(key_bytes)?;
        match value_bytes {
            Some(value_bytes) => {
                let value =
                    bincode::decode_from_slice::<V, Configuration>(&value_bytes, self.binconf)
                        .unwrap();
                Ok(Some(value.0))
            }
            None => Ok(None),
        }
    }

    pub fn contains_key(&self, key: &K) -> Result<bool, sled::Error> {
        let key_bytes = bincode::encode_to_vec(key, self.binconf).unwrap();
        Ok(self.db.contains_key(key_bytes)?)
    }

    pub fn flush(&self) -> Result<usize, sled::Error> {
        self.db.flush()
    }

    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn iter(&self) -> sled::Iter {
        self.db.iter()
    }

    pub fn decode_key(&self, key: sled::IVec) -> Result<K, DecodeError> {
        bincode::decode_from_slice::<K, Configuration>(&key, self.binconf).map(|(k, _)| k)
    }
    pub fn decode_value(&self, value: sled::IVec) -> Result<V, DecodeError> {
        bincode::decode_from_slice::<V, Configuration>(&value, self.binconf).map(|(v, _)| v)
    }
}
