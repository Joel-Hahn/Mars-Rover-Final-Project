"""terrain.py: turn a HiRISE DTM into a 2 m elevation + slope grid.

Owner: Shai. Output is a Terrain object consumed by optimize.py.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import rasterio
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


@dataclass
class Terrain:
    """The terrain grid handed to optimize.py.

    elev      : 2D float array, meters, NaN = no data
    slope_deg : 2D float array, same shape as elev
    cell_size : meters per cell (2.0)
    transform : affine transform of the grid: (col, row) -> map (x, y)
    crs       : the DTM's projected CRS (Mars)

    Row 0 is the north edge; rows increase southward, columns eastward.
    """
    elev: np.ndarray
    slope_deg: np.ndarray
    cell_size: float
    transform: Affine
    crs: object
    # Built once from crs in __post_init__; not passed in by the caller.
    _to_xy: Transformer = field(init=False, repr=False)
    _to_lonlat: Transformer = field(init=False, repr=False)

    def __post_init__(self):
        if self.elev.shape != self.slope_deg.shape:
            raise ValueError("elev and slope_deg must have the same shape.")
        self._to_xy, self._to_lonlat = make_transformers(self.crs)

    def lonlat_to_cell(self, lon, lat):
        """(lon, lat) in degrees -> (row, col) of the cell that contains it."""
        x, y = self._to_xy.transform(lon, lat)
        col_f, row_f = ~self.transform @ (x, y)   # fractional cell position
        row, col = math.floor(row_f), math.floor(col_f)  # which cell it's IN
        n_rows, n_cols = self.elev.shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(
                f"Point ({lon}, {lat}) is outside the terrain grid "
                f"(cell row {row}, col {col}; grid is {n_rows} x {n_cols})."
            )
        return row, col

    def cell_to_lonlat(self, row, col):
        """(row, col) -> (lon, lat) in degrees of that cell's CENTRE."""
        x, y = self.transform @ (col + 0.5, row + 0.5)
        return self._to_lonlat.transform(x, y)


def load_terrain(dtm_path, start, end, margin_m=100.0, cell_size=2.0):
    """Build the Terrain for one test drive. This is what main.py calls.

    dtm_path : path to the HiRISE DTM GeoTIFF
    start    : (lon, lat) in degrees, east-positive, from cases.py
    end      : (lon, lat) in degrees, east-positive, from cases.py
    margin_m : extra ground around the two points (meters)
    cell_size: output cell size in meters (2.0 by contract)

    Raises ValueError if either point is off the DTM or on no-data.
    """
    # 'with' closes the file again as soon as the block ends, even on an error.
    with rasterio.open(dtm_path) as src:
        to_xy, _ = make_transformers(src.crs)            # CRS read from the file
        window = crop_window(src, start, end, to_xy, margin_m)
        elev, transform = read_resampled(src, window, cell_size)
        crs = src.crs

    terrain = Terrain(elev, compute_slope(elev, cell_size), cell_size,
                      transform, crs)

    # R3: both points must land inside the final grid. Resampling can drop a
    # strip under one cell wide at the far edge; this catches that rare case
    # with a clear message (lonlat_to_cell raises ValueError).
    terrain.lonlat_to_cell(*start)
    terrain.lonlat_to_cell(*end)
    return terrain


# ---------- visual check (R3) ----------

# Colours: one-hue blue ramp for elevation (light = low, dark = high);
# start/end take the next two categorical hues so they never blend into it.
_ELEV_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_START_COLOR, _END_COLOR = "#eb6834", "#1baf7a"
_NODATA_COLOR = "#e1e0d9"
_INK, _MUTED = "#0b0b0b", "#898781"


def plot_check(terrain, start, end, out_path=None, title=None):
    """Draw the shaded terrain grid with start and end marked; return (fig, ax).

    Axes are meters east / north of the grid's bottom-left corner, so you can
    read distances straight off the plot. If out_path is given, save a PNG.
    """
    import matplotlib.pyplot as plt                  # only needed for this check
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import (LightSource, LinearSegmentedColormap,
                                   Normalize, to_rgb)
    from matplotlib.patches import Patch

    elev, cs = terrain.elev, terrain.cell_size
    n_rows, n_cols = elev.shape
    width_m, height_m = n_cols * cs, n_rows * cs
    extent = (0, width_m, 0, height_m)               # row 0 drawn at the top (north)
    nodata = np.isnan(elev)

    # Colour each cell by elevation (blue ramp: light = low, dark = high).
    ramp = LinearSegmentedColormap.from_list("elev", _ELEV_RAMP)
    norm = Normalize(np.nanmin(elev), np.nanmax(elev))
    rgb = ramp(norm(np.where(nodata, np.nanmin(elev), elev)))[..., :3]

    # Hillshade (sun from the north-west) shows the shape of the ground.
    # It can't handle NaN, so fill holes first; cells next to a hole (NaN
    # slope) get neutral light so the fill doesn't draw a fake cliff.
    # vert_exag=3 exaggerates relief for the eye only; the data is unchanged.
    light = LightSource(azdeg=315, altdeg=45)
    filled = np.where(nodata, np.nanmean(elev), elev)
    shade = light.hillshade(filled, vert_exag=3, dx=cs, dy=cs)
    shade[np.isnan(terrain.slope_deg)] = 0.5
    rgb = light.blend_soft_light(rgb, shade[..., None])
    rgb[nodata] = to_rgb(_NODATA_COLOR)

    fig, ax = plt.subplots(figsize=(8.5, 7 * height_m / width_m + 1.0),
                           layout="constrained")
    ax.imshow(rgb, extent=extent, origin="upper")

    # Mark the exact points (not just their cell) in plot meters.
    to_xy, _ = make_transformers(terrain.crs)
    for label, (lon, lat), color, shape in (("Start", start, _START_COLOR, "o"),
                                             ("End", end, _END_COLOR, "s")):
        x, y = to_xy.transform(lon, lat)
        col_f, row_f = ~terrain.transform @ (x, y)
        px, py = col_f * cs, height_m - row_f * cs
        ax.plot(px, py, shape, ms=11, color=color, mec="white", mew=2,
                label=label, zorder=3)
        ax.annotate(label, (px, py), xytext=(9, 9), textcoords="offset points",
                    color=_INK, fontsize=10, weight="bold")

    handles = ax.get_legend_handles_labels()[0]
    if nodata.any():
        handles.append(Patch(color=_NODATA_COLOR, label="No data"))
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    cbar = fig.colorbar(ScalarMappable(norm, ramp), ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Elevation (m)", color=_MUTED)
    cbar.ax.tick_params(colors=_MUTED, labelsize=9)
    cbar.outline.set_edgecolor(_NODATA_COLOR)
    ax.set_xlabel("meters east", color=_MUTED)
    ax.set_ylabel("meters north", color=_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_NODATA_COLOR)
    ax.set_title(title or f"Terrain check: {n_rows} x {n_cols} cells at {cs:g} m",
                 color=_INK, loc="left")

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, ax


if __name__ == "__main__":
    # Quick check on a real DTM, e.g.
    #   py terrain.py data/raw/DTM.tif 77.45 18.44 77.452 18.441 --out check.png
    import argparse

    parser = argparse.ArgumentParser(description="Load and plot a terrain grid.")
    parser.add_argument("dtm")
    parser.add_argument("start_lon", type=float)
    parser.add_argument("start_lat", type=float)
    parser.add_argument("end_lon", type=float)
    parser.add_argument("end_lat", type=float)
    parser.add_argument("--margin", type=float, default=100.0, help="meters")
    parser.add_argument("--out", default="terrain_check.png")
    args = parser.parse_args()

    start = (args.start_lon, args.start_lat)
    end = (args.end_lon, args.end_lat)
    t = load_terrain(args.dtm, start, end, margin_m=args.margin)
    finite = np.isfinite(t.slope_deg)
    print(f"Grid {t.elev.shape[0]} x {t.elev.shape[1]} cells at {t.cell_size:g} m")
    print(f"Elevation {np.nanmin(t.elev):.1f} to {np.nanmax(t.elev):.1f} m; "
          f"{np.isnan(t.elev).mean():.1%} no data")
    print(f"Slope: median {np.median(t.slope_deg[finite]):.1f} deg, "
          f"{(t.slope_deg[finite] > 25).mean():.1%} of cells over 25 deg")
    print(f"Start cell {t.lonlat_to_cell(*start)}, end cell {t.lonlat_to_cell(*end)}")
    plot_check(t, start, end, out_path=args.out)
    print(f"Saved {args.out}")
