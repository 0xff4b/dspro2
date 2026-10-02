| key | title | provider | licence | reference | resolution | unit | use |
|:---|:---|:---|:---|:---|:---|:---|:---|
| oev | Public-transport quality classes (ÖV-Güteklassen) and classified stops | ARE | OGD (opendata.swiss 'open use, must provide the source') | timetable 2026 | polygons / stop points | class 0-4; m | dimension transport |
| road_day | sonBASE road traffic noise, day (L_r,Tag 06-22 h) | BAFU | OGD (opendata.swiss 'open use, must provide the source'); source 'BAFU 2025, sonBASE' | 2021 (computed 2023) | 10 m raster | dB(A) | dimension noise; LSV/WHO thresholds |
| road_night | sonBASE road traffic noise, night (L_r,Nacht 22-06 h) | BAFU | OGD (opendata.swiss 'open use, must provide the source'); source 'BAFU 2025, sonBASE' | 2021 (computed 2023) | 10 m raster | dB(A) | dimension noise; LSV/WHO thresholds |
| rail_day | sonBASE railway noise, day (L_r,Tag 06-22 h) | BAFU | OGD (opendata.swiss 'open use, must provide the source'); source 'BAFU 2025, sonBASE' | 2021 (computed 2023) | 10 m raster | dB(A) | dimension noise; LSV/WHO thresholds |
| rail_night | sonBASE railway noise, night (L_r,Nacht 22-06 h) | BAFU | OGD (opendata.swiss 'open use, must provide the source'); source 'BAFU 2025, sonBASE' | 2021 (computed 2023) | 10 m raster | dB(A) | dimension noise; LSV/WHO thresholds |
| landuse | Land-use statistics (Arealstatistik), NOAS04 hectare points | BFS | OGD (opendata.swiss 'open use, must provide the source') | survey flights 2013-2020 (release 2024) | 100 m hectare points | share 0-1 | dimension green |
| hectares | Service accessibility per inhabited hectare (STATPOP population, distances) | BFS | OGD (opendata.swiss 'open use, must provide the source') | 2021 | 100 m hectares | inhabitants; m | reference grid (normalisation, municipal aggregates); validation of OSM access |
| no2 | PolluMap nitrogen dioxide (NO2), annual mean | BAFU | OGD (opendata.swiss 'open use, must provide the source') | 2025 | 20 m raster | µg/m³ | dimension air (extended index) |
| pm25 | PolluMap fine particulate matter (PM2.5), annual mean | BAFU | OGD (opendata.swiss 'open use, must provide the source') | 2025 | 100 m raster | µg/m³ | dimension air (extended index) |
| sunshine | Relative sunshine duration, climate normal 1991-2020 | MeteoSwiss | MeteoSwiss OGD (open use, source required) | 1991-2020 | 1 km grid | % of possible | dimension sunshine (extended index) |
| tax | Tax burden in the municipalities (cantonal, municipal and church tax) | ESTV | public statistics (no explicit licence; source required) | 2018 (latest file) | municipality (remapped to 2026) | % of gross income | dimension tax (extended index) |
| vacancy | Vacant dwellings and vacancy rate on 1 June (DF_LWZ_1) | BFS | OGD (opendata.swiss 'open use, must provide the source') | 2026 | municipality | % of dwellings | dimension housing availability (extended index); validation of the core index |
| city_statistics | City Statistics (Urban Audit): selected variables (DF_CITYSTAT_1) | BFS | OGD (opendata.swiss 'open use, must provide the source') | latest year per variable (2018-2025) | 10 core cities | various | validation (descriptive, n = 10) |
| communes | Historicised commune register: correspondence of municipality numbers | BFS | OGD (opendata.swiss 'open use, must provide the source') | 2018 -> 2026 | municipality | - | remap ESTV 2018 to 2026 boundaries |

