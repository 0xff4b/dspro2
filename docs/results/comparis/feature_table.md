| group               | feature                     | description                                                 |   missing % |
|:--------------------|:----------------------------|:------------------------------------------------------------|------------:|
| tabular             | area                        | living area (m²)                                            |         0   |
| tabular             | rooms                       | number of rooms (NaN if implausible)                        |         0.2 |
| tabular             | area_per_room               | area / rooms (m² per room)                                  |         0.2 |
| tabular             | rooms_is_integer            | whole room count (missingness indicator on DSPRO1 data)     |         0.2 |
| tabular             | year_built                  | construction year (GWR, period midpoint, else listing)      |        10.9 |
| tabular             | building_age                | 2026 − year built                                           |        10.9 |
| tabular             | apartments                  | dwellings in the building (GWR)                             |        18.6 |
| tabular             | land_area                   | GWR building area (m²)                                      |        14.4 |
| tabular             | land_area_per_apartment     | GWR building area per dwelling (m²)                         |        18.7 |
| tabular             | population                  | STATPOP population of the hectare (geo.admin)               |        12.4 |
| tabular             | oev                         | public-transport quality score                              |        10.3 |
| tabular             | solar                       | solar suitability class                                     |        24.4 |
| tabular             | elevation                   | elevation (m a.s.l.)                                        |        10.3 |
| tabular             | muni_density                | inhabitants per km² of municipal land area (lakes excluded) |         0   |
| tabular             | muni_population             | municipal population (swissBOUNDARIES3D)                    |         0   |
| coordinates         | east                        | LV95 east (m)                                               |         0   |
| coordinates         | north                       | LV95 north (m)                                              |         0   |
| coordinates         | rot45_x                     | LV95 rotated by 45° (x)                                     |         0   |
| coordinates         | rot45_y                     | LV95 rotated by 45° (y)                                     |         0   |
| hierarchy (RE / TE) | re_canton                   | canton key                                                  |         0   |
| hierarchy (RE / TE) | re_district                 | canton/district key                                         |         0   |
| hierarchy (RE / TE) | re_municipality             | canton/district/municipality key                            |         0   |
| text keywords       | kw_balcony                  | kw balcony                                                  |         0   |
| text keywords       | kw_terrace                  | kw terrace                                                  |         0   |
| text keywords       | kw_lift                     | kw lift                                                     |         0   |
| text keywords       | kw_lake_view                | kw lake view                                                |         0   |
| text keywords       | kw_mountain_view            | kw mountain view                                            |         0   |
| text keywords       | kw_renovated                | kw renovated                                                |         0   |
| text keywords       | kw_new_build                | kw new build                                                |         0   |
| text keywords       | kw_furnished                | kw furnished                                                |         0   |
| text keywords       | kw_temporary                | kw temporary                                                |         0   |
| text keywords       | kw_parking                  | kw parking                                                  |         0   |
| text keywords       | kw_garage                   | kw garage                                                   |         0   |
| text keywords       | kw_minergie                 | kw minergie                                                 |         0   |
| text keywords       | kw_loft                     | kw loft                                                     |         0   |
| text keywords       | kw_quiet                    | kw quiet                                                    |         0   |
| text keywords       | kw_pets                     | kw pets                                                     |         0   |
| text keywords       | kw_garden                   | kw garden                                                   |         0   |
| text keywords       | kw_dishwasher               | kw dishwasher                                               |         0   |
| text keywords       | kw_own_washer               | kw own washer                                               |         0   |
| text keywords       | kw_fireplace                | kw fireplace                                                |         0   |
| text keywords       | kw_attic                    | kw attic                                                    |         0   |
| text keywords       | kw_cooperative              | kw cooperative                                              |         0   |
| text keywords       | kw_subsidised               | kw subsidised                                               |         0   |
| text keywords       | kw_shared_flat              | kw shared flat                                              |         0   |
| text keywords       | kw_wheelchair               | kw wheelchair                                               |         0   |
| text attributes     | attr_floor                  | attr floor                                                  |        67.4 |
| text attributes     | attr_has_lift               | attr has lift                                               |        75.4 |
| text attributes     | attr_has_balcony_or_terrace | attr has balcony or terrace                                 |        31.5 |
| text attributes     | attr_view                   | attr view                                                   |         0   |
| text attributes     | attr_renovation_year        | attr renovation year                                        |        94.6 |
| text attributes     | attr_is_furnished           | attr is furnished                                           |        81.7 |
| text attributes     | attr_is_temporary           | attr is temporary                                           |        96.6 |
| text attributes     | attr_parking                | attr parking                                                |         0   |
| text attributes     | attr_is_minergie            | attr is minergie                                            |        97.1 |
| text attributes     | attr_pets_allowed           | attr pets allowed                                           |        93.8 |
| text attributes     | attr_has_garden             | attr has garden                                             |        82.6 |
| text attributes     | attr_has_own_washer         | attr has own washer                                         |        71.2 |
| text attributes     | attr_is_new_build           | attr is new build                                           |        93.4 |
| text attributes     | attr_is_attic               | attr is attic                                               |        89.8 |
| portal attributes   | floor                       | floor                                                       |        23.3 |
| portal attributes   | renovation_year             | renovation year                                             |        82   |
| portal attributes   | balcony_count               | balcony count                                               |         0   |
| portal attributes   | is_house                    | is house                                                    |         0   |
| portal attributes   | area_from_gwr               | area from gwr                                               |         0   |
| portal attributes   | has_balconies               | has balconies                                               |         0   |
| portal attributes   | has_baths                   | has baths                                                   |         0   |
| portal attributes   | has_cable_tv                | has cable tv                                                |         0   |
| portal attributes   | has_cellar                  | has cellar                                                  |         0   |
| portal attributes   | has_dishwasher              | has dishwasher                                              |         0   |
| portal attributes   | has_dryer                   | has dryer                                                   |         0   |
| portal attributes   | has_fireplace               | has fireplace                                               |         0   |
| portal attributes   | has_garden                  | has garden                                                  |         0   |
| portal attributes   | has_lift                    | has lift                                                    |         0   |
| portal attributes   | has_parking_indoor          | has parking indoor                                          |         0   |
| portal attributes   | has_parking_outdoor         | has parking outdoor                                         |         0   |
| portal attributes   | has_terraces                | has terraces                                                |         0   |
| portal attributes   | has_washingmachine          | has washingmachine                                          |         0   |
| portal attributes   | is_child_friendly           | is child friendly                                           |         0   |
