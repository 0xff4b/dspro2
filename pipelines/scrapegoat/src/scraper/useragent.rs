use rand::{rng, seq::IndexedRandom};
use std::{fs::read_to_string, io};

pub struct UserAgents {
  user_agents: Vec<String>,
}

impl UserAgents {
  pub fn new(file: &str) -> Result<UserAgents, io::Error> {
    let str_in = read_to_string(file)?;
    let user_agents = str_in.lines().map(|s| s.to_string()).collect();

    Ok(UserAgents { user_agents })
  }

  pub fn get_agent(&self) -> &String {
    self.user_agents.choose(&mut rng()).unwrap()
  }
}
