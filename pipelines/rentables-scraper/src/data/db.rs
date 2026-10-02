use crate::data::{ListingData, ListingDetails};
use sqlx::PgPool;

pub struct Db {
  pool: PgPool,
}

impl Db {
  pub async fn new(url: &str) -> Result<Self, sqlx::Error> {
    let pool = PgPool::connect(url).await?;
    Ok(Self { pool })
  }

  pub async fn insert_listing(&self, listing: &ListingData) -> Result<(), sqlx::Error> {
    sqlx::query!(
      r#"
            INSERT INTO listings (slug, rooms, m_sqrd, price_cold)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (slug) DO NOTHING
            "#,
      listing.slug,
      listing.rooms as i32,
      listing.m_sqrd as i32,
      listing.price_cold as i32,
    )
    .execute(&self.pool)
    .await?;

    Ok(())
  }

  pub async fn insert_listing_details(
    &self,
    listing_id: i32,
    details: &ListingDetails,
  ) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"
        INSERT INTO listing_details (listing_id, address, price_cold, area_sqm, rooms, auxiliary_costs, kaution, description, available_from)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (listing_id) DO UPDATE SET
            address = EXCLUDED.address,
            price_cold = EXCLUDED.price_cold,
            area_sqm = EXCLUDED.area_sqm,
            rooms = EXCLUDED.rooms,
            auxiliary_costs = EXCLUDED.auxiliary_costs,
            kaution = EXCLUDED.kaution,
            description = EXCLUDED.description,
            available_from = EXCLUDED.available_from
        "#,
        listing_id,
        details.address,
        details.price_cold,
        details.area_sqm,
        details.rooms,
        details.auxiliary_costs,
        details.kaution,
        details.description,
        details.available_from,
    )
    .execute(&self.pool)
    .await?;

    Ok(())
  }

  pub async fn get_all_slugs(&self) -> Result<Vec<(i32, String)>, sqlx::Error> {
    let rows = sqlx::query!("SELECT id, slug FROM listings")
      .fetch_all(&self.pool)
      .await?;

    Ok(rows.iter().map(|r| (r.id, r.slug.clone())).collect())
  }
}
