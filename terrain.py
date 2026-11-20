"""terrain.py: turn a HiRISE DTM into a 2 m elevation + slope grid.

Owner: Shai. Output is a Terrain object consumed by optimize.py.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Proj, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import EquidistantCylindricalConversion
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
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


def make_local_crs(crs, lon, lat):
    """Return a flat map of the same planet that is true to scale at (lon, lat).

    The DTM's own map (equirectangular, true scale at the equator) stretches
    east-west by 1/cos(latitude): about 5.4% at Jezero. This builds the same
    kind of map but true to scale at the drive's own latitude, centred on it,
    so 1 map meter = 1 ground meter in BOTH directions near the drive.
    """
    crs = CRS.from_user_input(crs)
    conversion = EquidistantCylindricalConversion(
        latitude_first_parallel=lat,        # true scale along THIS latitude
        longitude_natural_origin=lon,       # x = 0 at the drive
    )
    # Same planet/datum as the file: reuse its geodetic CRS, never a new one.
    return ProjectedCRS(conversion=conversion, geodetic_crs=crs.geodetic_crs,
                        name="Local true-scale map at the drive")


def crop_window(src, start, end, lonlat_to_xy, margin_m=100.0, extra=()):
    """Return the rasterio Window covering start, end and a margin around them.

    src          : an open rasterio dataset (the DTM)
    start, end   : (lon, lat) in degrees, east-positive
    lonlat_to_xy : transformer from make_transformers(src.crs)
    margin_m     : extra ground (meters) on every side, so routes can detour
    extra        : more (lon, lat) points the crop must also cover, e.g.
                   NASA's route (not checked for data, only included)

    Raises ValueError if start or end is outside the file or on no-data.
    """
    # 1. Both points into map meters (x east, y north).
    points_xy = [lonlat_to_xy.transform(lon, lat) for lon, lat in (start, end)]

    # 2. Each point must be on real elevation data, not just inside the file.
    for name, lonlat, (x, y) in zip(("start", "end"), (start, end), points_xy):
        _check_on_dtm(src, name, lonlat, x, y)

    # 3. Bounding box of all the points, grown by the margin (in meters).
    points_xy += [lonlat_to_xy.transform(lon, lat) for lon, lat in extra]
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


def read_resampled(src, window, dst_crs, dst_bounds, cell_size=2.0):
    """Read `window` of the DTM and re-draw it as square cell_size-m cells.

    src        : the open DTM
    window     : the piece of the file to read (from crop_window)
    dst_crs    : the map to draw the new grid on (from make_local_crs)
    dst_bounds : (left, bottom, right, top) of the new grid, in dst_crs meters
    cell_size  : side of each new cell in meters

    Returns (elev, transform):
      elev      : 2D float64 array in meters, NaN where the DTM has no data
      transform : affine transform of the NEW grid, in dst_crs meters
    """
    # 1. Read only the cropped piece. masked=True marks no-data; turn it into
    #    NaN so the fake -3.4e38 value never gets averaged into a cell.
    data = src.read(1, window=window, masked=True)
    data = data.astype(np.float64).filled(np.nan)

    # 2. The new, empty grid: enough whole cells to cover dst_bounds.
    left, bottom, right, top = dst_bounds
    n_cols = math.ceil((right - left) / cell_size)
    n_rows = math.ceil((top - bottom) / cell_size)
    transform = from_origin(left, top, cell_size, cell_size)   # top-left corner
    elev = np.full((n_rows, n_cols), np.nan)

    # 3. Re-draw: for each new cell, average the source pixels that fall
    #    inside it (Resampling.average). NaN pixels are left out of the mean;
    #    a cell with no valid pixels stays NaN.
    reproject(
        source=data, destination=elev,
        src_transform=src.window_transform(window), src_crs=src.crs,
        src_nodata=np.nan,
        dst_transform=transform, dst_crs=RioCRS.from_wkt(CRS(dst_crs).to_wkt()),
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return elev, transform


@dataclass
class Terrain:
    """The terrain grid handed to optimize.py.

    elev      : 2D float array, meters, NaN = no data
    slope_deg : 2D float array, same shape as elev
    cell_size : meters per cell (2.0), true ground meters in both directions
    transform : affine transform of the grid: (col, row) -> map (x, y)
    crs       : the local true-scale Mars map the grid is drawn on
                (from make_local_crs), not the DTM file's own map

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


def load_terrain(dtm_path, start, end, margin_m=100.0, cell_size=2.0,
                 include=()):
    """Build the Terrain for one test drive. This is what main.py calls.

    dtm_path : path to the HiRISE DTM GeoTIFF (or fetch.dtm_source())
    start    : (lon, lat) in degrees, east-positive, from cases.py
    end      : (lon, lat) in degrees, east-positive, from cases.py
    margin_m : extra ground around the points (meters)
    cell_size: output cell size in meters (2.0 by contract)
    include  : more (lon, lat) points the grid must cover, e.g. NASA's route,
               so plots.py can draw it on the same grid

    Raises ValueError if start or end is off the DTM or on no-data.
    """
    include = [tuple(p[:2]) for p in include]
    mid_lon = (start[0] + end[0]) / 2
    mid_lat = (start[1] + end[1]) / 2

    # 'with' closes the file again as soon as the block ends, even on an error.
    with rasterio.open(dtm_path) as src:
        # The output grid lives on a local map that is true to scale at the
        # drive, so every cell is cell_size x cell_size meters ON THE GROUND.
        local_crs = make_local_crs(src.crs, mid_lon, mid_lat)
        local_to_xy, _ = make_transformers(local_crs)
        xs, ys = zip(*(local_to_xy.transform(lon, lat)
                       for lon, lat in [start, end, *include]))
        dst_bounds = (min(xs) - margin_m, min(ys) - margin_m,
                      max(xs) + margin_m, max(ys) + margin_m)

        # Crop the file a little wider than the margin: the file's map is
        # stretched (scale factor k > 1), and the edge cells need whole pixels.
        k = Proj(CRS.from_user_input(src.crs)).get_factors(mid_lon, mid_lat)
        stretch = max(k.parallel_scale, k.meridional_scale)
        to_xy, _ = make_transformers(src.crs)            # CRS read from the file
        window = crop_window(src, start, end, to_xy,
                             margin_m * stretch + 2 * cell_size, extra=include)

        elev, transform = read_resampled(src, window, local_crs, dst_bounds,
                                         cell_size)

    terrain = Terrain(elev, compute_slope(elev, cell_size), cell_size,
                      transform, local_crs)

    # R3: both points must land inside the final grid (they always should,
    # since the grid is built around them; this fails loudly if not).
    terrain.lonlat_to_cell(*start)
    terrain.lonlat_to_cell(*end)
    return terrain


# ---------- connecting to fetch.py / cases.py ----------

def case_dtm_path(case):
    """Which DTM file to open for one case from cases.json.

    Prefers the small per-case crop that crop_dtm.py writes (fast, works
    offline); otherwise fetch.dtm_source(): the full DTM on disk, or the
    USGS file read over the internet.
    """
    import crop_dtm                      # imported here so terrain.py and its
    import fetch                         # tests don't need the network code

    crop = crop_dtm.OUT / f"{case['id']}.tif"
    return str(crop) if crop.exists() else fetch.dtm_source()


def load_case(case, dtm_path=None, margin_m=100.0, cell_size=2.0):
    """Build the Terrain for one case dict from cases.py's cases.json.

    Uses the case's start/end lon/lat, and makes the grid also cover NASA's
    actual route (nasa_route_lonlat) so it can be drawn on the same map.
    """
    start = (case["start"]["lon"], case["start"]["lat"])
    end = (case["end"]["lon"], case["end"]["lat"])
    route = case.get("nasa_route_lonlat", [])
    return load_terrain(dtm_path or case_dtm_path(case), start, end,
                        margin_m, cell_size, include=route)


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
    # Quick check on a real DTM. Either a case from cases.py's cases.json:
    #   py terrain.py --case case1
    # or any DTM and two points:
    #   py terrain.py data/dtm/DTM.tif 77.45 18.44 77.452 18.441 --out check.png
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and plot a terrain grid.")
    parser.add_argument("--case", help="case id from data/cases.json, e.g. case1")
    parser.add_argument("dtm", nargs="?")
    parser.add_argument("coords", nargs="*", type=float,
                        help="start_lon start_lat end_lon end_lat")
    parser.add_argument("--margin", type=float, default=100.0, help="meters")
    parser.add_argument("--out", default="terrain_check.png")
    args = parser.parse_args()

    if args.case:
        import cases                                       # Sam's module
        if not cases.CASES_FILE.exists():
            parser.error(f"{cases.CASES_FILE} not found. Run `py cases.py` "
                         f"(or `py run_data.py`) first.")
        all_cases = json.loads(cases.CASES_FILE.read_text())
        case = next((c for c in all_cases if c["id"] == args.case), None)
        if case is None:
            parser.error(f"no {args.case!r} in {cases.CASES_FILE}; have "
                         f"{[c['id'] for c in all_cases]}")
        start = (case["start"]["lon"], case["start"]["lat"])
        end = (case["end"]["lon"], case["end"]["lat"])
        print(f"{args.case}: sols {case['start']['sol']} -> {case['end']['sol']}, "
              f"{case['straight_m']} m apart; DTM: {args.dtm or case_dtm_path(case)}")
        t = load_case(case, dtm_path=args.dtm, margin_m=args.margin)
    elif args.dtm and len(args.coords) == 4:
        start, end = tuple(args.coords[:2]), tuple(args.coords[2:])
        t = load_terrain(args.dtm, start, end, margin_m=args.margin)
    else:
        parser.error("give --case ID, or a DTM path and 4 coordinates")
    finite = np.isfinite(t.slope_deg)
    print(f"Grid {t.elev.shape[0]} x {t.elev.shape[1]} cells at {t.cell_size:g} m")
    print(f"Elevation {np.nanmin(t.elev):.1f} to {np.nanmax(t.elev):.1f} m; "
          f"{np.isnan(t.elev).mean():.1%} no data")
    print(f"Slope: median {np.median(t.slope_deg[finite]):.1f} deg, "
          f"{(t.slope_deg[finite] > 25).mean():.1%} of cells over 25 deg")
    print(f"Start cell {t.lonlat_to_cell(*start)}, end cell {t.lonlat_to_cell(*end)}")
    plot_check(t, start, end, out_path=args.out)
    print(f"Saved {args.out}")
