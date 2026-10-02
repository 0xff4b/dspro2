use dotenv::dotenv;
use scrapegoat::ScrapeGoat;
use scraper::{Html, Selector};
use std::{collections::HashMap, sync::Arc};

use crate::data::ListingData;
use crate::data::db::Db;
use crate::detail_parser::parse_details;
use crate::mgt::Mgt;

mod data;
mod detail_parser;
mod mgt;

fn parse_listings(html: &str) -> Vec<ListingData> {
  let html = Html::parse_document(html);

  let selector = Selector::parse(r#"#listings > div > div"#).unwrap();
  let details = Selector::parse(r#"div:nth-child(2) > a > ul > li"#).unwrap();
  let price = Selector::parse("div:nth-child(2) > div > strong").unwrap();
  let link = Selector::parse("div:nth-child(2) > div > a").unwrap();

  html
    .select(&selector)
    .filter_map(|elem| {
      let mut data = elem.select(&details);

      let rooms: i32 = data
        .next()?
        .inner_html()
        .trim()
        .split(" ")
        .nth(0)?
        .parse()
        .unwrap_or(0);
      data.next();
      let m_sqrd: i32 = data
        .next()?
        .inner_html()
        .trim()
        .split(" ")
        .nth(0)?
        .parse()
        .unwrap_or(0);

      let price_cold: i32 = elem
        .select(&price)
        .next()?
        .inner_html()
        .trim()
        .replace(".", "")
        .split(" ")
        .nth(1)?
        .parse()
        .unwrap_or(0);

      let slug = elem
        .select(&link)
        .next()?
        .value()
        .attr("href")?
        .replace("/inserate/", "")
        .into();

      Some(ListingData {
        slug,
        rooms,
        m_sqrd,
        price_cold,
      })
    })
    .collect()
}

async fn scrape_listings(db: Arc<Db>, scraper: ScrapeGoat) {
  let urls: Vec<String> = (1..=750)
    .map(|p| format!("https://rentumo.ch/mietobjekte?types=apartment&page={}", p))
    .collect();

  let mut mgt = Mgt::new(scraper, 10, urls);

  mgt
    .run(|_url, html| {
      let db = db.clone();
      async move {
        let listings = parse_listings(&html);
        if listings.is_empty() {
          return;
        }
        for listing in &listings {
          db.insert_listing(listing).await.ok();
        }
        println!("inserted {} listings", listings.len());
      }
    })
    .await;
}

async fn scrape_details(db: Arc<Db>, scraper: ScrapeGoat) {
  let slugs = db.get_all_slugs().await.unwrap();
  println!("scraping details for {} listings", slugs.len());

  let urls: Vec<String> = slugs
    .iter()
    .map(|(_, slug)| format!("https://rentumo.ch/inserate/{}", slug))
    .collect();

  let slug_map: Arc<HashMap<String, i32>> = Arc::new(
    slugs
      .into_iter()
      .map(|(id, slug)| (format!("https://rentumo.ch/inserate/{}", slug), id))
      .collect(),
  );

  let mut mgt = Mgt::new(scraper, 10, urls);

  mgt
    .run(|url, html| {
      let db = db.clone();
      let slug_map = slug_map.clone();
      async move {
        let Some(&listing_id) = slug_map.get(&url) else {
          return;
        };
        let details = parse_details(&html);
        db.insert_listing_details(listing_id, &details).await.ok();
        println!("details saved for listing {}", listing_id);
      }
    })
    .await;
}

#[tokio::main]
async fn main() {
  dotenv().ok();

  let db = Arc::new(
    Db::new(&std::env::var("DATABASE_URL").unwrap())
      .await
      .unwrap(),
  );

  // phase 1: scrape listing pages
  // let scraper = ScrapeGoat::new("./proxies.txt", "./user_agents.txt", 10).unwrap();
  // scrape_listings(db.clone(), scraper).await;

  // phase 2: scrape detail pages
  let scraper = ScrapeGoat::new("./proxies.txt", "./user_agents.txt", 10).unwrap();
  scrape_details(db.clone(), scraper).await;
}
