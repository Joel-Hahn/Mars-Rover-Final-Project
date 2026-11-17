Driving Jezero — MCEN 1030 final project
Team of 3 (Shai, Joel, Sam), Python. Due Dec 2; each member defends the code individually in finals week.
What the project does
The user picks two points from a "breadcrumb trail" of Perseverance waypoints in Jezero crater. The program finds routes between them over a terrain grid built from a HiRISE elevation model (DTM) and overlays four routes on a map: least-energy, fastest, trade-off (w = 0.5), and NASA's actual route. Test drives are short: start and end are 50–500 m apart.
Pipeline (one owner per module): fetch.py → cases.py → terrain.py → optimize.py → plots.py, run by main.py.
This session: terrain.py (owner: Shai)
Job: turn a HiRISE DTM GeoTIFF plus a start/end point into a 2 m elevation and slope grid that optimize.py can search.
Contract with the other modules

* Input from cases.py: start and end as (lon, lat) in degrees, east-positive, from NASA's "Where is Perseverance?" GeoJSON.
* Output to optimize.py: one `Terrain` object holding:
   * `elev`: 2D float array in meters, NaN where the DTM has no data
   * `slope_deg`: 2D float array, same shape
   * `cell_size`: 2.0 (meters)
   * the raster transform and CRS
   * `lonlat_to_cell(lon, lat) -> (row, col)` and `cell_to_lonlat(row, col) -> (lon, lat)`
* optimize.py handles neighbors, costs and blocking (slope > 25° or NaN = impassable). terrain.py only reports the terrain.

Steps

1. Crop: read only a window around the two points plus a margin (default 100 m), not the whole file.
2. Coordinates: read the CRS from the GeoTIFF; never hard-code it. Convert lon/lat with `pyproj.Transformer` from the raster CRS's `geodetic_crs` to the raster CRS (`always_xy=True`). HiRISE DTMs are on a Mars body, not Earth, so never use EPSG:4326.
3. Resample to 2 m cells (HiRISE DTMs are usually about 1 m/post) with `Resampling.average`.
4. No-data: convert the file's nodata value to NaN.
5. Slope: `np.gradient(elev, cell_size)` → `slope_deg = degrees(arctan(hypot(dz_dx, dz_dy)))`. Watch the sign/axis convention: rows run north to south.

Requirement R3 (from the Pin-up #1 spec): tests to pass (pytest)

* Synthetic plane z = 0.1·x on a 2 m grid → slope 5.71° within 0.1° (away from the edges).
* Flat plane → slope 0 everywhere.
* A NaN cell stays NaN in `elev` and doesn't crash the slope.
* `lonlat_to_cell` then `cell_to_lonlat` round-trips within half a cell (1 m).
* Both input points land inside the cropped grid; a point off the DTM raises a clear error.
* A visual check: plot the shaded grid with the two points marked.

Libraries
numpy, rasterio, pyproj, matplotlib, pytest. Nothing else without asking.
Data rules

* The full DTM zip goes in `data/raw/` and is gitignored.
* Only small cropped test tiles get committed (`data/tiles/`).
