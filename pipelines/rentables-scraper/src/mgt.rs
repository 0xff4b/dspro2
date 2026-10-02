use scrapegoat::ScrapeGoat;
use std::{future::Future, sync::Arc};
use tokio::sync::{Mutex, mpsc};

enum JobResult {
  Page { url: String, html: String },
  Failed { url: String },
}

pub struct Mgt {
  tx: mpsc::Sender<JobResult>,
  rx: mpsc::Receiver<JobResult>,
  in_flight: u32,
  max_concurrent: u32,
  scraper: Arc<Mutex<ScrapeGoat>>,
  queue: Vec<String>,
  done: bool,
}

impl Mgt {
  pub fn new(scraper: ScrapeGoat, max_concurrent: u32, urls: Vec<String>) -> Self {
    let (tx, rx) = mpsc::channel(max_concurrent as usize * 2);
    Self {
      tx,
      rx,
      in_flight: 0,
      max_concurrent,
      scraper: Arc::new(Mutex::new(scraper)),
      queue: urls.into_iter().rev().collect(), // rev so we can pop from end
      done: false,
    }
  }

  fn spawn_next(&mut self) {
    let Some(url) = self.queue.pop() else {
      self.done = true;
      return;
    };

    let tx = self.tx.clone();
    let scraper = self.scraper.clone();
    let url_clone = url.clone();

    tokio::spawn(async move {
      let result = scraper.lock().await.get_page(&url_clone).await;

      let msg = match result {
        Ok(html) => JobResult::Page {
          url: url_clone,
          html,
        },
        Err(e) if e.status == 404 => JobResult::Failed { url: url_clone },
        Err(_) => JobResult::Failed { url: url_clone },
      };

      tx.send(msg).await.ok();
    });

    self.in_flight += 1;
  }

  pub async fn run<F, Fut>(&mut self, mut on_page: F)
  where
    F: FnMut(String, String) -> Fut,
    Fut: Future<Output = ()>,
  {
    while self.in_flight < self.max_concurrent && !self.queue.is_empty() {
      self.spawn_next();
    }

    while self.in_flight > 0 {
      match self.rx.recv().await {
        Some(JobResult::Page { url, html }) => {
          self.in_flight -= 1;
          on_page(url, html).await;
          if !self.done && !self.queue.is_empty() {
            self.spawn_next();
          } else if self.queue.is_empty() {
            self.done = true;
          }
        }
        Some(JobResult::Failed { .. }) => {
          self.in_flight -= 1;
          if !self.queue.is_empty() {
            self.spawn_next();
          } else {
            self.done = true;
          }
        }
        None => break,
      }
    }
  }
}
