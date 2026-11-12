"""Tests for terrain.py (requirement R3)."""

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from terrain import compute_slope, crop_window, make_transformers, read_resampled

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
