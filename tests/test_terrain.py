"""Tests for terrain.py (requirement R3)."""

import matplotlib

matplotlib.use("Agg")                     # draw to files, never open a window

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from terrain import (Terrain, compute_slope, crop_window, load_terrain,
                     make_transformers, plot_check, read_resampled)

CELL = 2.0  # meters

# A Mars equirectangular CRS written the way HiRISE DTM files describe it.
# (Tests build their own so they don't need the big DTM file.)
MARS_RADIUS = 3396190.0
MARS_EQC = CRS.from_wkt(
    'PROJCS["Equirectangular MARS",'
    'GEOGCS["GCS_MARS",DATUM["D_MARS",SPHEROID["MARS",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",77.4],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)


# ---------- compute_slope ----------

def test_plane_rising_east_is_5_71_degrees():
    # z = 0.1 * x, with x = column index * 2 m.  Slope = atan(0.1) = 5.71 deg.
    x = np.arange(10) * CELL
    elev = np.tile(0.1 * x, (8, 1))          # 8 identical rows
    slope = compute_slope(elev, CELL)
    interior = slope[1:-1, 1:-1]               # stay away from the edges
    assert np.allclose(interior, 5.71, atol=0.1)


def test_plane_rising_north_is_also_5_71_degrees():
    # Same tilt, but along the rows. Row 0 is north, so z DEcreases
    # as the row index increases. The sign must not change the slope.
    northing = -np.arange(8) * CELL           # row 0 highest
    elev = np.tile((0.1 * northing)[:, None], (1, 10))
    slope = compute_slope(elev, CELL)
    assert np.allclose(slope[1:-1, 1:-1], 5.71, atol=0.1)


def test_flat_plane_is_zero_everywhere():
    elev = np.full((6, 6), 100.0)
    assert np.all(compute_slope(elev, CELL) == 0.0)


def test_nan_cell_stays_nan_and_does_not_crash():
    x = np.arange(5) * CELL
    elev = np.tile(0.1 * x, (5, 1))
    elev[2, 2] = np.nan

    slope = compute_slope(elev, CELL)          # must not raise

    assert np.isnan(elev[2, 2])                # input not "filled in"
    assert np.isnan(slope[2, 2])               # the NaN cell itself
    for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        assert np.isnan(slope[r, c])           # its 4 neighbours (Q6)
    for r, c in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        assert np.isfinite(slope[r, c])        # diagonals are unaffected


# ---------- make_transformers ----------

def test_projection_centre_maps_to_origin():
    to_xy, _ = make_transformers(MARS_EQC)
    x, y = to_xy.transform(77.4, 0.0)          # (lon, lat) order
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_one_degree_east_uses_mars_radius_not_earth():
    # 1 deg of arc on Mars = 2*pi*R/360 = 59,274.7 m  (Earth would be 111,319 m)
    to_xy, _ = make_transformers(MARS_EQC)
    x, _ = to_xy.transform(78.4, 0.0)
    assert abs(x - 2 * np.pi * MARS_RADIUS / 360) < 0.01


def test_lonlat_xy_round_trip():
    to_xy, to_lonlat = make_transformers(MARS_EQC)
    lon, lat = 77.45, 18.44                    # roughly Jezero crater
    x, y = to_xy.transform(lon, lat)
    lon2, lat2 = to_lonlat.transform(x, y)
    assert abs(lon2 - lon) < 1e-9 and abs(lat2 - lat) < 1e-9


# ---------- a small fake DTM for the file-reading steps ----------

HIRISE_NODATA = -3.4028226550889045e38   # the value real HiRISE DTMs use
DTM_W, DTM_H = 600, 400                  # pixels (1 m each)
COLLAR = 50                              # left 50 columns have no data


@pytest.fixture
def dtm(tmp_path):
    """Write a 1 m/pixel Mars GeoTIFF near Jezero; return (path, lonlat_of).

    lonlat_of(col, row) gives the (lon, lat) of that pixel's centre, so tests
    can place points on known pixels.
    """
    path = tmp_path / "fake_dtm.tif"
    transform = from_origin(3000.0, 1093000.0, 1.0, 1.0)   # top-left x, y, pixel size
    cols = np.arange(DTM_W)
    elev = np.tile(-2500.0 + 0.1 * cols, (DTM_H, 1)).astype("float32")
    elev[:, :COLLAR] = HIRISE_NODATA

    with rasterio.open(
        path, "w", driver="GTiff", width=DTM_W, height=DTM_H, count=1,
        dtype="float32", crs=MARS_EQC.to_wkt(), transform=transform,
        nodata=HIRISE_NODATA,
    ) as dst:
        dst.write(elev, 1)

    _, to_lonlat = make_transformers(MARS_EQC)

    def lonlat_of(col, row):
        x, y = transform @ (col + 0.5, row + 0.5)
        return to_lonlat.transform(x, y)

    return path, lonlat_of


# ---------- crop_window ----------

def test_crop_contains_both_points_plus_margin(dtm):
    path, lonlat_of = dtm
    start, end = lonlat_of(200, 150), lonlat_of(400, 250)   # ~224 m apart
    with rasterio.open(path) as src:
        to_xy, _ = make_transformers(src.crs)
        win = crop_window(src, start, end, to_xy, margin_m=100.0)

    # 100 m margin at 1 m/pixel -> about 100 pixels each side of the points
    assert win.col_off <= 100 and win.col_off + win.width >= 501
    assert win.row_off <= 50 and win.row_off + win.height >= 351
    # ...but not the whole file
    assert win.width < DTM_W and win.height < DTM_H


def test_crop_is_clipped_to_the_file_edge(dtm):
    path, lonlat_of = dtm
    start, end = lonlat_of(560, 10), lonlat_of(590, 390)    # near the right edge
    with rasterio.open(path) as src:
        to_xy, _ = make_transformers(src.crs)
        win = crop_window(src, start, end, to_xy, margin_m=100.0)
    assert win.col_off >= 0 and win.row_off >= 0
    assert win.col_off + win.width <= DTM_W
    assert win.row_off + win.height <= DTM_H


def test_point_outside_the_file_raises_clear_error(dtm):
    path, lonlat_of = dtm
    start = lonlat_of(200, 150)
    end = lonlat_of(900, 150)                                # 300 m past the right edge
    with rasterio.open(path) as src:
        to_xy, _ = make_transformers(src.crs)
        with pytest.raises(ValueError, match="outside the DTM"):
            crop_window(src, start, end, to_xy)


def test_point_on_nodata_raises_clear_error(dtm):
    path, lonlat_of = dtm
    start = lonlat_of(10, 150)                               # inside the file, in the collar
    end = lonlat_of(300, 150)
    with rasterio.open(path) as src:
        to_xy, _ = make_transformers(src.crs)
        with pytest.raises(ValueError, match="no elevation data"):
            crop_window(src, start, end, to_xy)


# ---------- read_resampled ----------

def test_resample_gives_2m_cells_and_right_shape(dtm):
    path, _ = dtm
    with rasterio.open(path) as src:
        elev, transform = read_resampled(src, Window(100, 50, 200, 100), CELL)
    assert elev.shape == (50, 100)               # 100 m x 200 m at 2 m/cell
    assert transform.a == 2.0 and transform.e == -2.0   # +2 m east, 2 m south per cell
    assert elev.dtype == np.float64


def test_resample_averages_the_1m_pixels(dtm):
    # Fake DTM: pixel in column c has z = -2500 + 0.1*c.
    # 2 m cell j (inside the window) averages columns 100+2j and 101+2j.
    path, _ = dtm
    with rasterio.open(path) as src:
        elev, _ = read_resampled(src, Window(100, 50, 200, 100), CELL)
    j = np.arange(100)
    expected = -2500.0 + 0.1 * (100 + 2 * j + 0.5)
    assert np.allclose(elev[0], expected, atol=1e-3)
    # ...and the slope module agrees: 0.1 m per m -> 5.71 deg
    assert np.allclose(compute_slope(elev, CELL)[1:-1, 1:-1], 5.71, atol=0.1)


def test_resample_turns_nodata_into_nan(dtm):
    path, _ = dtm
    with rasterio.open(path) as src:
        elev, _ = read_resampled(src, Window(0, 0, 200, 100), CELL)
    collar_cells = COLLAR // 2                   # 50 px of no-data = 25 cells
    assert np.all(np.isnan(elev[:, :collar_cells]))
    assert np.all(np.isfinite(elev[:, collar_cells:]))
    assert np.nanmin(elev) > -3000               # no -3.4e38 left anywhere


def test_resample_keeps_exact_2m_for_odd_windows(dtm):
    path, _ = dtm
    with rasterio.open(path) as src:
        elev, transform = read_resampled(src, Window(100, 50, 201, 101), CELL)
    assert transform.a == 2.0 and transform.e == -2.0
    assert elev.shape == (50, 100)               # the odd last metre is dropped


def test_resample_transform_is_georeferenced(dtm):
    # The new transform must put cell (0, 0) where the window really is.
    path, _ = dtm
    with rasterio.open(path) as src:
        _, transform = read_resampled(src, Window(100, 50, 200, 100), CELL)
        window_corner = src.transform @ (100, 50)    # top-left of the window
    assert transform @ (0, 0) == pytest.approx(window_corner)


def test_resample_partial_nodata_cell_uses_only_valid_pixels(tmp_path):
    # Real HiRISE strips are tilted, so the no-data edge cuts through cells.
    # A cell that is half no-data must average only its real pixels.
    elev_1m = np.full((4, 6), 100.0, dtype="float32")
    elev_1m[:, :3] = HIRISE_NODATA        # 3-px collar splits the 2nd cell
    elev_1m[:, 3] = 200.0                 # the valid half of that cell
    path = tmp_path / "edge.tif"
    with rasterio.open(path, "w", driver="GTiff", width=6, height=4, count=1,
                       dtype="float32", crs=MARS_EQC.to_wkt(),
                       transform=from_origin(0, 4, 1, 1), nodata=HIRISE_NODATA) as dst:
        dst.write(elev_1m, 1)
    with rasterio.open(path) as src:
        elev, _ = read_resampled(src, Window(0, 0, 6, 4), CELL)
    assert np.all(np.isnan(elev[:, 0]))   # fully no-data cell
    assert np.allclose(elev[:, 1], 200.0) # half cell: only the valid pixels
    assert np.allclose(elev[:, 2], 100.0)


# ---------- Terrain object ----------

WINDOW = Window(100, 50, 400, 300)    # 400 m x 300 m of the fake DTM


@pytest.fixture
def terrain(dtm):
    path, _ = dtm
    with rasterio.open(path) as src:
        elev, transform = read_resampled(src, WINDOW, CELL)
        crs = src.crs
    return Terrain(elev, compute_slope(elev, CELL), CELL, transform, crs)


def test_terrain_holds_the_contract_fields(terrain):
    assert terrain.elev.shape == (150, 200)
    assert terrain.slope_deg.shape == terrain.elev.shape
    assert terrain.cell_size == 2.0
    assert terrain.transform.a == 2.0
    assert terrain.crs is not None


def test_lonlat_to_cell_finds_the_right_cell(dtm, terrain):
    # 1 m pixel (col 200, row 150) is 100 px right and 100 px down from the
    # window corner -> 2 m cell (row 50, col 50).
    _, lonlat_of = dtm
    assert terrain.lonlat_to_cell(*lonlat_of(200, 150)) == (50, 50)


def test_rows_run_north_to_south(dtm, terrain):
    _, lonlat_of = dtm
    north = terrain.lonlat_to_cell(*lonlat_of(300, 60))
    south = terrain.lonlat_to_cell(*lonlat_of(300, 300))
    assert north[0] < south[0]                 # smaller row = further north


def test_cell_to_lonlat_gives_the_cell_centre(terrain):
    to_xy, _ = make_transformers(terrain.crs)
    x, y = to_xy.transform(*terrain.cell_to_lonlat(0, 0))
    corner_x, corner_y = terrain.transform.c, terrain.transform.f   # top-left corner
    assert x == pytest.approx(corner_x + 1.0)  # 1 m east of the corner
    assert y == pytest.approx(corner_y - 1.0)  # 1 m south of the corner


def test_round_trip_within_half_a_cell_per_axis(dtm, terrain):
    # R3: lon/lat -> cell -> lon/lat lands within 1 m in x AND within 1 m in y.
    # (Not 1 m in a straight line: a corner point is 1.41 m from the centre.)
    _, lonlat_of = dtm
    to_xy, _ = make_transformers(terrain.crs)
    rng = np.random.default_rng(0)
    for col, row in rng.uniform([101, 51], [499, 349], size=(50, 2)):
        lon, lat = lonlat_of(col, row)
        lon2, lat2 = terrain.cell_to_lonlat(*terrain.lonlat_to_cell(lon, lat))
        x, y = to_xy.transform(lon, lat)
        x2, y2 = to_xy.transform(lon2, lat2)
        assert abs(x2 - x) <= 1.0 and abs(y2 - y) <= 1.0


def test_lonlat_to_cell_off_the_grid_raises(dtm, terrain):
    _, lonlat_of = dtm
    with pytest.raises(ValueError, match="outside the terrain grid"):
        terrain.lonlat_to_cell(*lonlat_of(550, 200))   # in the file, not the crop


def test_cells_are_plain_ints(dtm, terrain):
    _, lonlat_of = dtm
    row, col = terrain.lonlat_to_cell(*lonlat_of(200, 150))
    assert type(row) is int and type(col) is int   # safe to use as indices


# ---------- load_terrain (the one call main.py makes) ----------

def test_load_terrain_end_to_end(dtm):
    path, lonlat_of = dtm
    start, end = lonlat_of(200, 150), lonlat_of(400, 250)   # ~224 m apart
    t = load_terrain(path, start, end)

    assert isinstance(t, Terrain)
    assert t.cell_size == 2.0
    # R3: both input points land inside the cropped grid
    for lon, lat in (start, end):
        row, col = t.lonlat_to_cell(lon, lat)
        assert np.isfinite(t.elev[row, col])
    # cropped (300 m x 400 m area + margin), not the whole 400 x 600 m file
    assert t.elev.shape[0] < DTM_H // 2 and t.elev.shape[1] < DTM_W // 2
    # fake DTM is a 0.1 m/m ramp -> 5.71 deg wherever slope is defined
    finite = np.isfinite(t.slope_deg)
    assert np.allclose(t.slope_deg[finite], 5.71, atol=0.1)


def test_load_terrain_margin_is_adjustable(dtm):
    path, lonlat_of = dtm
    start, end = lonlat_of(250, 200), lonlat_of(350, 200)
    small = load_terrain(path, start, end, margin_m=20.0)
    big = load_terrain(path, start, end, margin_m=100.0)
    assert small.elev.size < big.elev.size


def test_load_terrain_off_dtm_point_raises(dtm):
    path, lonlat_of = dtm
    with pytest.raises(ValueError, match="outside the DTM"):
        load_terrain(path, lonlat_of(200, 150), lonlat_of(900, 150))


def test_load_terrain_closes_the_file(dtm):
    # The DTM must not stay open (on Windows an open file can't be moved/deleted).
    path, lonlat_of = dtm
    load_terrain(path, lonlat_of(200, 150), lonlat_of(400, 250))
    path.rename(path.with_suffix(".moved.tif"))             # fails if still open


# ---------- plot_check (R3 visual check) ----------

def _marker(ax, label):
    """The (x, y) of the plotted marker with this legend label."""
    (line,) = [ln for ln in ax.get_lines() if ln.get_label() == label]
    return line.get_xdata()[0], line.get_ydata()[0]


def test_plot_check_marks_both_points_in_the_right_place(dtm):
    path, lonlat_of = dtm
    start, end = lonlat_of(200, 150), lonlat_of(400, 250)
    t = load_terrain(path, start, end)
    fig, ax = plot_check(t, start, end)

    # Axes are meters east / north of the grid's bottom-left corner.
    n_rows, n_cols = t.elev.shape
    height_m = n_rows * t.cell_size
    for label, (lon, lat) in (("Start", start), ("End", end)):
        x, y = make_transformers(t.crs)[0].transform(lon, lat)
        col_f, row_f = ~t.transform @ (x, y)
        expected = (col_f * t.cell_size, height_m - row_f * t.cell_size)
        assert _marker(ax, label) == pytest.approx(expected)

    # Start is 200 m west and 100 m north of End.
    sx, sy = _marker(ax, "Start")
    ex, ey = _marker(ax, "End")
    assert ex - sx == pytest.approx(200.0) and sy - ey == pytest.approx(100.0)
    matplotlib.pyplot.close(fig)


def test_plot_check_saves_a_png_even_with_nodata(dtm, tmp_path):
    path, lonlat_of = dtm
    start, end = lonlat_of(80, 150), lonlat_of(250, 200)   # crop includes the no-data collar
    t = load_terrain(path, start, end)
    assert np.isnan(t.elev).any()
    out = tmp_path / "check.png"
    fig, _ = plot_check(t, start, end, out_path=out)
    assert out.exists() and out.stat().st_size > 10_000
    matplotlib.pyplot.close(fig)
