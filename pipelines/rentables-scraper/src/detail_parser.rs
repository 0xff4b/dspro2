use crate::data::ListingDetails;
use scraper::{Html, Selector};
use serde_json::Value;

pub fn parse_details(html: &str) -> ListingDetails {
  let doc = Html::parse_document(html);
  let mut details = ListingDetails::default();

  // address
  let address_sel = Selector::parse("#address").unwrap();
  details.address = doc
    .select(&address_sel)
    .next()
    .map(|e| e.text().collect::<String>().trim().to_string());

  // JSON-LD for description
  let jsonld_sel = Selector::parse("script[type='application/ld+json']").unwrap();
  for el in doc.select(&jsonld_sel) {
    let text = el.text().collect::<String>();
    if let Ok(val) = serde_json::from_str::<Value>(&text) {
      if val.get("@type").and_then(|t| t.as_str()) == Some("Apartment") {
        details.description = val
          .get("description")
          .and_then(|d| d.as_str())
          .map(|s| s.to_string());
      }
    }
  }

  // property detail rows — label/value pairs
  let row_sel = Selector::parse("div.flex.mt-\\[16px\\]").unwrap();
  let label_sel = Selector::parse("div.flex-1").unwrap();
  let value_sel = Selector::parse("div.pl-3").unwrap();

  for row in doc.select(&row_sel) {
    let label = row
      .select(&label_sel)
      .next()
      .map(|e| e.text().collect::<String>().trim().to_string())
      .unwrap_or_default();
    let value = row
      .select(&value_sel)
      .next()
      .map(|e| e.text().collect::<String>().trim().to_string())
      .unwrap_or_default();

    match label.as_str() {
      "Fläche" => {
        details.area_sqm = value.split_whitespace().next().and_then(|v| v.parse().ok());
      }
      "Zimmer" => {
        details.rooms = value
          .split_whitespace()
          .next()
          .and_then(|v| v.replace(",", ".").parse().ok());
      }
      "Kaltmiete" => {
        details.price_cold = value
          .replace("CHF", "")
          .replace(".", "")
          .trim()
          .parse()
          .ok();
      }
      "Nebenkosten" => {
        details.auxiliary_costs = value
          .replace("CHF", "")
          .replace(".", "")
          .trim()
          .parse()
          .ok();
      }
      "Kaution" => {
        details.kaution = value
          .replace("CHF", "")
          .replace(".", "")
          .trim()
          .parse()
          .ok();
      }
      "Datum verfügbar" => {
        details.available_from = if value == "-" { None } else { Some(value) };
      }
      _ => {}
    }
  }

  details
}
