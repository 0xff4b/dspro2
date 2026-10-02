CREATE TABLE listings (
    id SERIAL PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    rooms INTEGER NOT NULL,
    m_sqrd INTEGER NOT NULL,
    price_cold INTEGER NOT NULL
);