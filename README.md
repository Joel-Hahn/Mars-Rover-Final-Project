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

---

# Driving Jezero -- terrain stage (terrain.py)

Owner: Terrain lead (Shai). Covers R3.

Turns the HiRISE DTM plus a start and end point into a 2 m grid of elevation
and slope that optimize.py can search.

## What it needs
- `data/cases.json` from cases.py (or any two lon/lat points).
- A DTM. terrain.py uses the first one it finds:
  1. `data/dtm_crops/<case>.tif` (crop_dtm.py; fast, offline, those cases only)
  2. `data/dtm/JEZ_hirise_soc_006_DTM_1m.tif` (the full download; any case)
  3. otherwise the USGS file read over the internet (~20 s per case)

## Use it from code
    import json, terrain
    case = json.load(open("data/cases.json"))[0]
    t = terrain.load_case(case)              # one case from cases.json
    # or: t = terrain.load_terrain(dtm_path, (lon, lat), (lon, lat))

    t.elev, t.slope_deg      # 2D arrays in m and degrees; NaN = no data
    t.cell_size              # 2.0 m (true ground meters)
    t.lonlat_to_cell(lon, lat)   # -> (row, col); row 0 is the north edge
    t.cell_to_lonlat(row, col)   # -> (lon, lat) of the cell centre

## Run
    python terrain.py --case case1          # load one case, print stats, save terrain_check.png
    python terrain.py <dtm.tif> <lon> <lat> <lon> <lat>   # any two points
    python -m pytest -v                     # R3 tests (no download needed)

On Windows, use `py` instead of `python` if `python` opens the Microsoft Store.

## Hand-off notes (optimize.py, plots.py, UI)
- **Cells are true 2 m x 2 m on the ground.** The USGS DTM's map is stretched
  ~5% east-west at Jezero, so each crop is re-drawn on a local true-scale map
  centred on the drive. `t.crs` is that local map, not the DTM file's CRS, so
  use `lonlat_to_cell` / `cell_to_lonlat` rather than the file's transform.
- **NaN spreads one cell.** A no-data cell also makes its 4 neighbours' slope
  NaN, so optimize.py gets a 1-cell buffer around gaps.
- **Errors.** `load_case` / `load_terrain` raise `ValueError` if the start or
  end is off the DTM or on no-data. Catch it and pick another pair.
- **The grid also covers NASA's route** (`nasa_route_lonlat`), so it can be
  drawn on the same map.
- terrain.py only *reports* slope. Blocking (slope > 25 deg or NaN) is
  optimize.py's job.
- DTM coverage: 77.22-77.58 E, 18.31-18.67 N. That is 94% of NASA's waypoints
  (sols 13-2022); the crater-rim drive west of 77.22 E (from sol 1815) is not
  covered.

## R3 tests (`tests/test_terrain.py`)
Run on a small fake Mars DTM written during the test, so no download is needed.
- Plane z = 0.1*x on a 2 m grid -> 5.71 deg; flat plane -> 0 deg.
- A NaN cell stays NaN and doesn't crash the slope.
- lon/lat -> cell -> lon/lat round-trips within 1 m in x and 1 m in y
  (a cell corner is 1.41 m from its centre, so "1 m in a straight line" can't hold).
- Both points land inside the grid; a point off the DTM raises a clear error.
- True ground scale at Jezero's latitude; `terrain_check.png` is the visual check.
