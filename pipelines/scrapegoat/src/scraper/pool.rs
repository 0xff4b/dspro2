use crate::scraper::proxy::Proxies;
use reqwest::{Client, Proxy};

pub struct ClientPool {
  permits: Permits,
  clients: Vec<Client>,
  idx: usize,
}

impl ClientPool {
  pub fn new(proxies: Proxies, max_concurrent: usize) -> Self {
    let clients: Vec<Client>;
    if proxies.len() == 0 {
      clients = (0..max_concurrent)
        .map(|_| Client::builder().build().unwrap())
        .collect();
    } else {
      // build a client per proxy
      clients = proxies
        .map(|p| {
          Client::builder()
            .proxy(Proxy::all(p).unwrap())
            .build()
            .unwrap()
        })
        .collect();
    }

    Self {
      clients,
      permits: Permits::new(max_concurrent),
      idx: 0,
    }
  }

  pub fn get(&mut self) -> Result<&Client, NoPermitError> {
    self.permits.get()?; // reserve permit

    self.idx = (self.idx + 1) % self.clients.len();
    Ok(&self.clients[self.idx])
  }

  // return permit
  pub fn drop(&mut self) -> Result<(), NoPermitError> {
    self.permits.drop()?;

    Ok(())
  }
}

struct Permits {
  issued_permits: usize,
  max_permits: usize,
}
pub struct NoPermitError;

impl Permits {
  pub fn new(max_permits: usize) -> Self {
    Self {
      max_permits,
      issued_permits: 0,
    }
  }

  pub fn get(&mut self) -> Result<(), NoPermitError> {
    if self.issued_permits == self.max_permits {
      return Err(NoPermitError {});
    }

    self.issued_permits += 1;
    Ok(())
  }

  pub fn drop(&mut self) -> Result<(), NoPermitError> {
    if self.issued_permits == 0 {
      return Err(NoPermitError {});
    };

    self.issued_permits -= 1;
    Ok(())
  }
}
