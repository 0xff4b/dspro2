use rand::{rng, seq::IndexedRandom};
use std::{fs::read_to_string, io};

pub struct Proxies {
  proxies: Vec<String>,
  idx: usize,
}

impl Proxies {
  pub fn new(file: &str) -> Result<Proxies, io::Error> {
    let str_in = read_to_string(file)?;
    let proxies = str_in.lines().map(|s| s.to_string()).collect();

    Ok(Proxies { proxies, idx: 0 })
  }

  pub fn len(&self) -> usize {
    return self.proxies.len();
  }
}

impl Iterator for Proxies {
  type Item = String;

  fn next(&mut self) -> Option<Self::Item> {
    if self.idx >= self.proxies.len() {
      return None;
    }

    let item = self.proxies[self.idx].clone();
    self.idx += 1;

    Some(item)
  }
}
