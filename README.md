# Driving Jezero -- data stage (fetch.py, cases.py)

Owner: Data & photos lead. Covers R1, R2 (and R8 later).

## Setup
    pip install -r requirements.txt

## Data
- NASA waypoints + traverse GeoJSON: downloaded and cached automatically.
- HiRISE DTM (big): download once from
  https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_dtm_mosaic
  and save as `data/dtm/JEZ_hirise_soc_006_DTM_1m.tif`
- Photos: downloaded per case; if that fails, hand-place 1-2 images in
  `data/photos/<case>/start/` and `.../end/`.

## Run
    python fetch.py       # R1: check/cached data, prints property names
    python cases.py       # R2: picks 50-500 m cases, saves data/cases.json
    python crop_dtm.py    # small per-case DTM crops for the Terrain lead
    python run_data.py    # all three in order

In VS Code: Run and Debug panel has one launch entry per step.

## Hand-off
- Terrain lead: `data/dtm_crops/*.tif` (1 m cells; resample to 2 m). CRS is a Mars
  equirectangular projection, not lon/lat.
- Optimizer lead: `data/cases.json` -- each case has start/end lon/lat/sol,
  `straight_m`, `nasa_path_m`, `nasa_route_lonlat`.

## R1 offline test
Delete `data/cache/`, rerun; then disconnect and rerun. Should use cache or fail with a clear error.
