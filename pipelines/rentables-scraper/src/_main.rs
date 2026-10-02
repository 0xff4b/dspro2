use std::{
  fs::{File, read_to_string},
  io::Write,
  path::PrefixComponent,
};

use scrapegoat::ScrapeGoat;
use scraper::{ElementRef, Html, Selector};

use crate::data::{ListingData, parse_field};

mod data;
mod mgt;

const PAGE_LIMIT: u16 = 375;
/*
get all mehr sehen links
r#"#listings > div > div > div.bg-white.xl\:px-5.px-4.xl\:pt-3\.5.xl\:pb-3\.5.pt-2\.5.pb-2\.5 > div > a.text-sm.font-bold.block.text-center.text-blue_reg"#

get all listing items
r#"#listings > div"# ?

*/

// #[tokio::main]
// async fn main() {
//   let mut scraper = ScrapeGoat::new("./proxies.txt", "./user_agents.txt", 1).expect("fuck you");

//   let res = scraper
//     .get_page("https://rentumo.ch/inserate/2-zimmer-40-m-3-stock-307784")
//     .await
//     .expect("adwoij");

//   let mut file = File::create("output.html").expect("Failed to create file");
//   file.write_all(res.as_bytes())
//     .expect("Failed to write to file");

//   println!("{res}");
// }

fn main() {
  let test = read_to_string("./output_listings.html").unwrap();

  let html = Html::parse_document(&test);

  // get all listing links
  let selector = Selector::parse(r#"#listings > div > div"#).unwrap();

  let details = Selector::parse(r#"div:nth-child(2) > a > ul > li"#).unwrap();

  let price = Selector::parse("div:nth-child(2) > div > strong").unwrap();
  let link = Selector::parse("div:nth-child(2) > div > a").unwrap();

  for elem in html.select(&selector) {
    let mut data = elem.select(&details);

    /*
      "1 Zimmer"
      "Wohnung"
      "36 m²"
      "CHF 770"
      Some("/inserate/1-zimmer-36-m-308921")
    */

    let mut obj = ListingData::default();
    obj.rooms = data
      .next()
      .unwrap()
      .inner_html()
      .trim()
      .split(" ")
      .nth(0)
      .unwrap()
      .parse()
      .unwrap_or(0);

    data.next();

    obj.m_sqrd = data
      .next()
      .unwrap()
      .inner_html()
      .trim()
      .split(" ")
      .nth(0)
      .unwrap()
      .parse()
      .unwrap_or(0);

    obj.price_cold = elem
      .select(&price)
      .next()
      .unwrap()
      .inner_html()
      .trim()
      .replace(".", "")
      .split(" ")
      .nth(1)
      .unwrap()
      .parse()
      .unwrap_or(0);

    obj.slug = elem
      .select(&link)
      .next()
      .unwrap()
      .value()
      .attr("href")
      .unwrap_or("")
      .replace("/inserate/", "")
      .into();

    println!("{:?}", obj);
  }
}

// fn parse_details(data: &mut ListingData, html: &Html) {
//   let address = Selector::parse("#address").unwrap();
//   let desc = Selector::parse("script[type='application/ld+json']").unwrap();
//   let floor = Selector::parse(r#"body > div.out-content.mt-\[68px\].md\:mt-\[172px\].xl\:mt-\[102px\] > section.xs\:pt-4.md\:pt-5 > div > div.flex.flex-wrap.-mx-5 > div.w-full.px-5.lg\:w-2\/3 > div:nth-child(12) > div > div:nth-child(4) > div > div.pl-3.text-lg.font-semibold.text-black.tracking-\[-0\.01em\]"#).unwrap();
//   let aux_costs = Selector::parse(r#"body > div.out-content.mt-\[68px\].md\:mt-\[172px\].xl\:mt-\[102px\] > section.xs\:pt-4.md\:pt-5 > div > div.flex.flex-wrap.-mx-5 > div.w-full.px-5.lg\:w-2\/3 > div:nth-child(14) > div > div:nth-child(2) > div > div.pl-3.text-lg.font-semibold.text-black.tracking-\[-0\.01em\]"#).unwrap();
//   let kaution = Selector::parse(r#"body > div.out-content.mt-\[68px\].md\:mt-\[172px\].xl\:mt-\[102px\] > section.xs\:pt-4.md\:pt-5 > div > div.flex.flex-wrap.-mx-5 > div.w-full.px-5.lg\:w-2\/3 > div:nth-child(14) > div > div:nth-child(3) > div > div.pl-3.text-lg.font-semibold.text-black.tracking-\[-0\.01em\]"#).unwrap();

//   data.address = parse_field(&html, &address);
//   data.desc = parse_field(&html, &desc);
//   data.floor = parse_field(&html, &floor);
//   data.auxiliary_costs = parse_field(&html, &aux_costs);
//   data.kaution = parse_field(&html, &kaution, |e| e.replace("CHF ", ""));
// }
