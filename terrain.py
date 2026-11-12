"""terrain.py: turn a HiRISE DTM into a 2 m elevation + slope grid.

Owner: Shai. Output is a Terrain object consumed by optimize.py.
"""

import math

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window


def compute_slope(elev, cell_size):
    """Return slope in degrees for every cell of a 2D elevation grid.

    elev      : 2D array of elevations in meters (NaN = no data)
    cell_size : distance between neighbouring cells in meters

    Rows run north -> south and columns run west -> east, so the row
    derivative is -dz/dy. The sign doesn't matter here because hypot()
    squares both parts.
    """
    # np.gradient returns derivatives in axis order: rows first, then columns.
    dz_drow, dz_dcol = np.gradient(elev, cell_size)

    # Steepness = size of the gradient vector; slope angle = arctan of it.
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dcol, dz_drow)))

    # A central difference skips the cell itself, so a NaN cell would get a
    # made-up slope from its neighbours. Put the NaN back.
    slope_deg[np.isnan(elev)] = np.nan
    return slope_deg


def make_transformers(crs):
    """Return (lonlat_to_xy, xy_to_lonlat) transformers for a raster's CRS.

    crs : the DTM's projected CRS (rasterio's raster.crs or a pyproj CRS).
          Always read it from the file; never hard-code it.

    lon/lat are degrees, east-positive, on the SAME body the map is drawn
    on (Mars for HiRISE). That body comes from crs.geodetic_crs, so we never
    use Earth's EPSG:4326.
    """
    crs = CRS.from_user_input(crs)     # accept rasterio's CRS too
    geodetic = crs.geodetic_crs        # Mars lon/lat for this projection

    # always_xy=True: inputs/outputs are (lon, lat) and (x, y), never (lat, lon).
    lonlat_to_xy = Transformer.from_crs(geodetic, crs, always_xy=True)
    xy_to_lonlat = Transformer.from_crs(crs, geodetic, always_xy=True)
    return lonlat_to_xy, xy_to_lonlat


def crop_window(src, start, end, lonlat_to_xy, margin_m=100.0):
    """Return the rasterio Window covering start, end and a margin around them.

    src          : an open rasterio dataset (the DTM)
    start, end   : (lon, lat) in degrees, east-positive
    lonlat_to_xy : transformer from make_transformers(src.crs)
    margin_m     : extra ground (meters) on every side, so routes can detour

    Raises ValueError if either point is outside the file or on no-data.
    """
    # 1. Both points into map meters (x east, y north).
    points_xy = [lonlat_to_xy.transform(lon, lat) for lon, lat in (start, end)]

    # 2. Each point must be on real elevation data, not just inside the file.
    for name, lonlat, (x, y) in zip(("start", "end"), (start, end), points_xy):
        _check_on_dtm(src, name, lonlat, x, y)

    # 3. Bounding box of the two points, grown by the margin (in meters).
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    left, right = min(xs) - margin_m, max(xs) + margin_m
    bottom, top = min(ys) - margin_m, max(ys) + margin_m

    # 4. Meters -> fractional pixel positions. ~transform is the inverse
    #    of the raster transform, i.e. (x, y) -> (col, row).
    inverse = ~src.transform
    col_a, row_a = inverse @ (left, top)
    col_b, row_b = inverse @ (right, bottom)

    # 5. Round OUTWARD to whole pixels so nothing gets cut off, then clip to
    #    the file so a margin past the edge doesn't ask for missing pixels.
    col_start = max(0, math.floor(min(col_a, col_b)))
    row_start = max(0, math.floor(min(row_a, row_b)))
    col_stop = min(src.width, math.ceil(max(col_a, col_b)))
    row_stop = min(src.height, math.ceil(max(row_a, row_b)))

    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def _check_on_dtm(src, name, lonlat, x, y):
    """Raise a clear ValueError if map point (x, y) has no DTM elevation."""
    row, col = src.index(x, y)        # which pixel holds (x, y)
    if not (0 <= row < src.height and 0 <= col < src.width):
        raise ValueError(
            f"{name} point {lonlat} is outside the DTM "
            f"(pixel row {row}, col {col}; file is {src.height} x {src.width})."
        )

    # HiRISE DTMs are tilted strips with a no-data border, so a point can
    # be inside the file's rectangle and still have no elevation.
    pixel = src.read(1, window=Window(col, row, 1, 1), masked=True)
    if pixel.mask.any():
        raise ValueError(
            f"{name} point {lonlat} is inside the DTM file but has no "
            f"elevation data there. Pick a waypoint on the DTM strip."
        )


def read_resampled(src, window, cell_size=2.0):
    """Read `window` of the DTM, averaged onto square cells of `cell_size` m.

    Returns (elev, transform):
      elev      : 2D float64 array in meters, NaN where the DTM has no data
      transform : affine transform of the NEW grid (cell (0,0) = window corner)
    """
    res_x, res_y = src.res                     # source pixel size in m (~1 for HiRISE)

    # How many whole 2 m cells fit in the window. A leftover strip narrower
    # than one cell is dropped, so every cell is exactly cell_size wide.
    n_cols = int(window.width * res_x // cell_size)
    n_rows = int(window.height * res_y // cell_size)
    window = Window(window.col_off, window.row_off,
                    n_cols * cell_size / res_x, n_rows * cell_size / res_y)

    # Resampling.average: each 2 m cell = mean of the 1 m pixels under it
    # (no-data pixels are left out of the mean). masked=True marks no-data.
    data = src.read(1, window=window, out_shape=(n_rows, n_cols),
                    resampling=Resampling.average, masked=True)
    elev = data.astype(np.float64).filled(np.nan)

    # Grid geometry: start at the window's top-left corner, then stretch the
    # pixel size from res to cell_size.
    transform = src.window_transform(window) @ Affine.scale(
        cell_size / res_x, cell_size / res_y)
    return elev, transform
