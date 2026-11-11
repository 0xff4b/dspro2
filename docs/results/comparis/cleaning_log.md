| step                                       |   n_rows |   removed |   removed_pct |   retained_pct |
|:-------------------------------------------|---------:|----------:|--------------:|---------------:|
| raw listings                               |    65552 |         0 |          0    |          100.0 |
| rentals with monthly price and coordinates |    25666 |     39886 |         60.8  |           39.2 |
| one row per object (dedup)                 |    23129 |      2537 |          9.88 |           35.3 |
| admin match in Switzerland                 |    23129 |         0 |          0    |           35.3 |
| domain rules (area, rent, rooms)           |    19571 |      3558 |         15.4  |           29.9 |
| price-per-m² outliers                      |    19046 |       525 |          2.68 |           29.1 |
