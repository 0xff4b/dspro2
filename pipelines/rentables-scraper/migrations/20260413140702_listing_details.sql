CREATE TABLE listing_details (
    id SERIAL PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    address TEXT,
    price_cold INTEGER,
    area_sqm INTEGER,
    rooms NUMERIC(4,1),
    auxiliary_costs INTEGER,
    kaution INTEGER,
    description TEXT,
    available_from TEXT,
    UNIQUE(listing_id)
);