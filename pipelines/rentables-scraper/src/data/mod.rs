pub mod db;

use scraper::{Html, Selector};
use std::str::FromStr;

#[derive(Debug, Default)]
pub struct ListingData {
  pub slug: String,
  pub rooms: i32,
  pub m_sqrd: i32,
  pub price_cold: i32,
}

#[derive(Debug, Default)]
pub struct ListingDetails {
  pub address: Option<String>,
  pub price_cold: Option<i32>,
  pub area_sqm: Option<i32>,
  pub rooms: Option<i32>,
  pub auxiliary_costs: Option<i32>,
  pub kaution: Option<i32>,
  pub description: Option<String>,
  pub available_from: Option<String>,
}

pub fn parse_field<T: FromStr + Default, F: Fn(&str) -> String>(
  html: &Html,
  selector: &Selector,
  transform: F,
) -> T {
  html
    .select(selector)
    .next()
    .map(|el| {
      transform(el.inner_html().trim())
        .parse()
        .unwrap_or_default()
    })
    .unwrap_or_default()
}
