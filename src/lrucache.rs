use std::collections::hash_map::{IntoIter, Iter};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub struct LruCache<K, V> {
    capacity: Option<usize>,
    cache: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return LruCache {
                capacity: None,
                cache: HashMap::new(),
                order: VecDeque::new(),
            };
        } else {
            LruCache {
                capacity: Some(capacity),
                cache: HashMap::new(),
                order: VecDeque::new(),
            }
        }
    }
    pub fn insert(&mut self, key: K, value: V) {
        if !self.cache.contains_key(&key) {
            if self.capacity.is_some() && self.cache.len() == self.capacity.unwrap() {
                let oldest_key = self.order.pop_front().unwrap();
                self.cache.remove(&oldest_key);
            }
            self.cache.insert(key.clone(), value);
            self.order.push_back(key);
        }
    }
    pub fn contains_key(&self, key: K) -> bool {
        self.cache.contains_key(&key)
    }
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(value) = self.cache.get(key) {
            let index = self.order.iter().position(|x| *x == *key).unwrap();
            self.order.remove(index);
            self.order.push_back(key.clone());
            Some(value)
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(value) = self.cache.get_mut(key) {
            let index = self.order.iter().position(|x| *x == *key).unwrap();
            self.order.remove(index);
            self.order.push_back(key.clone());
            Some(value)
        } else {
            None
        }
    }

    pub fn iter(&mut self) -> Iter<'_, K, V> {
        self.cache.iter()
    }
    pub fn into_iter(self) -> IntoIter<K, V> {
        self.cache.into_iter()
    }
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }
}
