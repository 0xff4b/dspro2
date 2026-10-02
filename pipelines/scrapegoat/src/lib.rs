mod scraper;

use std::io;

use reqwest::StatusCode;

use crate::scraper::{pool::ClientPool, proxy::Proxies, useragent::UserAgents};

pub struct ScrapeGoat {
  user_agents: UserAgents,
  pool: ClientPool, // client list
}

#[derive(Debug)]
pub struct Error {
  pub status: u16,
  pub msg: String,
}
impl Error {
  pub fn new(status: u16, msg: String) -> Self {
    Self { status, msg }
  }
}

impl ScrapeGoat {
  pub fn new(
    proxy_file: &str,
    useragents_file: &str,
    max_concurrent: usize,
  ) -> Result<Self, io::Error> {
    let proxies = Proxies::new(proxy_file)?;
    let user_agents = UserAgents::new(useragents_file)?;
    Ok(Self {
      user_agents,
      pool: ClientPool::new(proxies, max_concurrent),
    })
  }

  pub async fn get_page(&mut self, url: &str) -> Result<String, Error> {
    // fetch client or error if no permit
    let Ok(client) = self.pool.get() else {
      return Err(Error::new(500, "no permit".to_string()));
    };

    println!("{:?}", client);

    // get page / throw err?
    let res = match client
      .get(url)
      .header("User-Agent", self.user_agents.get_agent())
      .send()
      .await
    {
      Ok(r) => Ok(r.text().await.expect("text not texting?")),
      Err(e) => Err(Error::new(e.status().unwrap_or(StatusCode::from_u16(500).unwrap()).as_u16(), e.to_string())),
    };

    _ = self.pool.drop(); // return permit

    res
  }
}

#[cfg(test)]
mod tests {

  #[tokio::test]
  async fn get_google() {
    todo!()
  }
}
