"""Desktop Mars Rover path comparison using full NASA mission traverse and 2 m terrain.

Put this file beside main.py, terrain.py, traversability.py, optimization.py,
and data/cases.json in your Mars-Rover-Final-Project checkout.

Run: python main.py   (existing main.py calls launch_ui())
     python mars_rover_ui.py

Data provenance and limits
--------------------------
* Uses NASA's full Perseverance traverse cached by fetch.py at
  data/cache/M20_traverse.json (downloads it if missing), not just four test
  cases. The case tracks remain available in the drop-down for focused tests.
  The ~50 m selectable markers are INTERPOLATED ALONG NASA's recorded geometry;
  they are NOT additional NASA rover location measurements.
  Missing gaps in NASA segments are not drawn as a fictional rover drive.
* With a real DTM, the zoomable map spans the source GeoTIFF's available
  extent and is an elevation colorization, NOT a satellite photograph.
* Click anywhere in the mapped terrain or choose recorded-track markers at
  ~50 m intervals. Only the second selection has a 500 m maximum radius;
  there is no 50 m minimum separation requirement.
* Drag route information cards to rearrange them within the map view.
  At close zoom a toggleable 2 x 2 m grid is drawn, based on the active
  loaded terrain grid (it shows only where that grid has data).
* A NASA path segment exists only when both selections are on/very close to
  the recorded trace; other map locations can still be optimized.
* Fastest/lowest-energy routes use optimization.PathOptimizer and the
  direction-dependent TraversabilityModel.evaluate_move() costs (no charging).
* An Energy graphs button opens total energy, cumulative energy vs distance,
  and energy per meter plots. Graphs use the exact per-move costs computed by
  traversability.py, not separate assumed numbers. The graph window supports
  route selection, zoom/pan and saving a PNG via Matplotlib's toolbar.
* NASA-route time and energy are MODEL ESTIMATES, computed by snapping the
  recorded polyline to the 2 m grid. These are NOT measured NASA telemetry.
  A recorded path can cross terrain the conservative model rejects, in which
  case NASA time and energy are displayed as unavailable, not fabricated.
* Slope thresholds, speeds and energy coefficients are modeling assumptions;
  this is not a flight-safe rover navigator.

Dependencies: pip install -r requirements.txt; pip install pillow
Pillow is optional for display of hillshade/image; computations work without it.
The full NASA track is fetched using the repository's fetch.py and cached
locally, so it is available offline after the first download. Place the
NASA/USGS HiRISE DTM at data/dtm/JEZ_hirise_soc_006_DTM_1m.tif, or generate
case crops with python crop_dtm.py, or choose a DTM TIFF in the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Mapping, Sequence

import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "data" / "cases.json"
DTM_FULL = HERE / "data" / "dtm" / "JEZ_hirise_soc_006_DTM_1m.tif"
DTM_CROPS = HERE / "data" / "dtm_crops"
NASA_TRAVERSE_CACHE = HERE / "data" / "cache" / "M20_traverse.json"
FULL_MISSION = "Full Perseverance mission (NASA)"
MARS_RADIUS_M = 3_396_190.0  # same sphere used in cases.py
POINT_SPACING_M = 50.0
MAX_DISTANCE_M = 500.0

ACTUAL = "NASA recorded path"
FASTEST = "Fastest path"
LOWEST = "Lowest-energy path"
ROUTE_NAMES = (ACTUAL, FASTEST, LOWEST)
ROUTE_STYLES = {
    ACTUAL: ("#ea580c", ()),
    FASTEST: ("#0284c7", (8, 4)),
    LOWEST: ("#8b5cf6", (5, 4)),
}

XY = tuple[float, float]


@dataclass(frozen=True)
class RoverPoint:
    """Point at a sampled distance along a *recorded* NASA route."""

    name: str
    x_m: float
    y_m: float
    along_m: float | None = None
    lon: float | None = None
    lat: float | None = None
    segment_id: int = 0  # which continuous portion of the recorded traverse


@dataclass(frozen=True)
class EnergySample:
    """One directional tile move's modeled energy and cumulative path distance.

    The initial sample is (0, 0, 0, 0). Energy is Joules; distance is
    meters. Plotting converts Joules to watt-hours and never recalculates
    movement costs independently from traversability.py.
    """

    distance_m: float
    cumulative_energy_j: float
    move_energy_j: float
    move_distance_m: float


def profile_from_moves(moves: Sequence) -> tuple[EnergySample, ...]:
    """Build an energy trace directly from traversability/optimizer moves."""
    samples = [EnergySample(0.0, 0.0, 0.0, 0.0)]
    distance = energy = 0.0
    for move in moves:
        d, e = float(move.distance_m), float(move.energy_j)
        if (not math.isfinite(d) or not math.isfinite(e) or
                d <= 0 or e < 0 or not getattr(move, "passable", True)):
            return ()  # no valid per-move energy data; never fabricate it
        distance += d
        energy += e
        samples.append(EnergySample(distance, energy, e, d))
    return tuple(samples)


@dataclass(frozen=True)
class RouteDisplay:
    name: str
    coordinates: tuple[XY, ...]
    distance_m: float | None
    time_s: float | None
    energy_j: float | None
    note: str
    ascent_m: float | None = None
    descent_m: float | None = None
    energy_profile: tuple[EnergySample, ...] = ()

    @property
    def energy_wh(self) -> float | None:
        return None if self.energy_j is None else self.energy_j / 3600.0


def local_xy(lonlat: Sequence[Sequence[float]], ref_lon: float, ref_lat: float) -> list[XY]:
    """Local Mars tangent-plane coordinates for previews without a DTM.

    Once terrain loads, use the precise projected Terrain._to_xy instead.
    """
    cos_lat = math.cos(math.radians(ref_lat))
    return [
        (MARS_RADIUS_M * math.radians(float(lon) - ref_lon) * cos_lat,
         MARS_RADIUS_M * math.radians(float(lat) - ref_lat))
        for lon, lat in lonlat
    ]


def cumulative_distances(coords: Sequence[XY]) -> list[float]:
    if not coords:
        return []
    cum = [0.0]
    for a, b in zip(coords, coords[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum


def interpolate_at(coords: Sequence[XY], cumulative: Sequence[float], pos: float) -> XY:
    if not coords:
        raise ValueError("Recorded path must not be empty")
    if pos <= 0:
        return coords[0]
    if pos >= cumulative[-1]:
        return coords[-1]
    import bisect
    i = min(max(bisect.bisect_right(cumulative, pos) - 1, 0), len(coords) - 2)
    while i + 1 < len(cumulative) - 1 and cumulative[i + 1] <= cumulative[i]:
        i += 1
    span = cumulative[i + 1] - cumulative[i]
    t = (pos - cumulative[i]) / span if span else 0.0
    a, b = coords[i], coords[i + 1]
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def sample_recorded_track(
    coords: Sequence[XY], spacing_m: float = POINT_SPACING_M,
) -> list[RoverPoint]:
    """Resample points *on* an observed polyline, without inventing a route.

    Always include end. If the last interval is <40% spacing, merge it into
    the previous interval rather than adding almost-overlapping markers.
    """
    if len(coords) < 2 or spacing_m <= 0:
        raise ValueError("Need a recorded line with >=2 points and spacing > 0")
    cum = cumulative_distances(coords)
    length = cum[-1]
    if length <= 0:
        raise ValueError("Recorded track has zero measured length")
    # Evenly distribute the markers so the first and last are real endpoints;
    # each intermediate one is approximately spacing_m along the NASA line.
    # This avoids a tiny final interval (e.g. 4 m after a 400 m marker).
    intervals = max(1, int(math.floor(length / spacing_m + 0.5)))
    targets = [length * k / intervals for k in range(intervals + 1)]
    return [
        RoverPoint(f"P{i:02d}", *interpolate_at(coords, cum, s), along_m=s)
        for i, s in enumerate(targets, start=1)
    ]


def recorded_subpath(coords: Sequence[XY], start_m: float, end_m: float) -> list[XY]:
    """Precise sub-polyline between two arclength positions, in requested order."""
    if not coords:
        return []
    cum = cumulative_distances(coords)
    reverse = start_m > end_m
    lo, hi = (end_m, start_m) if reverse else (start_m, end_m)
    if lo < -1e-6 or hi > cum[-1] + 1e-6:
        raise ValueError("Selection is outside NASA recorded track")
    middle = [coords[i] for i, s in enumerate(cum) if lo < s < hi]
    result = [interpolate_at(coords, cum, lo), *middle,
              interpolate_at(coords, cum, hi)]
    return list(reversed(result)) if reverse else result


def read_cases(path: Path = CASES_FILE) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("data/cases.json should be a JSON list of case objects")
    return [case for case in data
            if isinstance(case, dict) and len(case.get("nasa_route_lonlat", [])) >= 2]


def _mars_ground_distance(a: Sequence[float], b: Sequence[float]) -> float:
    lat = math.radians((float(a[1]) + float(b[1])) / 2)
    return MARS_RADIUS_M * math.hypot(
        math.radians(float(a[0]) - float(b[0])) * math.cos(lat),
        math.radians(float(a[1]) - float(b[1])),
    )


def parse_nasa_traverse(data: dict, join_gap_m: float = 1.0) -> list[list[XY]]:
    """Read the real NASA M20 traverse GeoJSON, preserving unrecorded gaps.

    NASA's traverse is made of LineString and MultiLineString drive geometries.
    Adjacent pieces are joined only when their endpoints are essentially
    connected; never draw or sample a fictitious line between distant pieces.
    Original feature/line ordering is preserved, as in repository cases.py.
    """
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("Expected NASA traverse FeatureCollection")
    pieces: list[list[XY]] = []
    for feat in data["features"]:
        geom = feat.get("geometry") or {}
        shape = geom.get("type")
        if shape == "LineString":
            lines = [geom.get("coordinates", [])]
        elif shape == "MultiLineString":
            lines = geom.get("coordinates", [])
        else:
            continue
        for part in lines:
            coords: list[XY] = []
            for pt in part:
                if not isinstance(pt, (tuple, list)) or len(pt) < 2:
                    continue
                try:
                    lon, lat = float(pt[0]), float(pt[1])
                except (ValueError, TypeError):
                    continue
                if not (-90 <= lat <= 90) or not all(map(math.isfinite, (lon, lat))):
                    continue
                if not coords or _mars_ground_distance(coords[-1], (lon, lat)) > .002:
                    coords.append((lon, lat))
            if len(coords) >= 2:
                if pieces and _mars_ground_distance(pieces[-1][-1], coords[0]) <= join_gap_m:
                    # Remove duplicate seam vertices, but retain the actual
                    # short connection if coordinates differ slightly.
                    pieces[-1].extend(coords[1:] if _mars_ground_distance(pieces[-1][-1], coords[0]) < .002 else coords)
                else:
                    pieces.append(coords)
    pieces = [part for part in pieces if len(part) >= 2 and
              sum(_mars_ground_distance(a, b) for a, b in zip(part, part[1:])) >= 1.0]
    if not pieces:
        raise ValueError("NASA traverse contains no usable recorded segments")
    return pieces


def full_mission_case(data: dict, reference_case: dict | None = None) -> dict:
    """Adapt the full recorded NASA route to the existing UI case interface."""
    parts = parse_nasa_traverse(data)
    base = reference_case or {}
    anchor = base.get("start", {"lon": parts[0][0][0], "lat": parts[0][0][1]})
    nearby = base.get("end")
    if nearby is None or _mars_ground_distance(
            (anchor["lon"], anchor["lat"]), (nearby["lon"], nearby["lat"])) > 1000:
        nearby = {"lon": parts[0][1][0], "lat": parts[0][1][1]}
    return {"id": FULL_MISSION,
            "start": {"lon": float(anchor["lon"]), "lat": float(anchor["lat"])},
            "end": {"lon": float(nearby["lon"]), "lat": float(nearby["lat"])},
            # Keep load_case()'s include= geometry local to its small anchor.
            # The ACTUAL full track lives only in _segments_lonlat below.
            "nasa_route_lonlat": [(float(anchor["lon"]), float(anchor["lat"])),
                                  (float(nearby["lon"]), float(nearby["lat"]))],
            "_segments_lonlat": parts,
            "_full_mission": True}


def load_full_mission_case(*, refresh: bool = False,
                           reference_case: dict | None = None) -> dict:
    """Reuse the repository fetch.py, including its offline NASA traverse cache.

    This is called in a worker thread, never while Tkinter draws frames.
    """
    if NASA_TRAVERSE_CACHE.is_file() and not refresh:
        with NASA_TRAVERSE_CACHE.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        from fetch import load_geojson
        data = load_geojson("traverse", refresh=refresh)
    return full_mission_case(data, reference_case)


def find_local_dtm(case_id: str) -> Path | None:
    # Crop wins for quick loading; all choices stay local and offline.
    crop = DTM_CROPS / f"{case_id}.tif"
    if crop.is_file():
        return crop
    if DTM_FULL.is_file():
        return DTM_FULL
    return None


def map_center(track_xy: Sequence[XY]) -> XY:
    return ((min(p[0] for p in track_xy) + max(p[0] for p in track_xy)) / 2,
            (min(p[1] for p in track_xy) + max(p[1] for p in track_xy)) / 2)


def projected_track(terrain, lonlat: Sequence[Sequence[float]]) -> list[XY]:
    """Use terrain.py's true-scale local Mars projection, not Earth lat/lon."""
    projector = terrain._to_xy  # cached pyproj transformer created by Terrain
    return [tuple(map(float, projector.transform(float(p[0]), float(p[1]))))
            for p in lonlat]


def route_cells_near_recorded_line(terrain, xy_line: Sequence[XY]) -> list[tuple[int, int]]:
    """Snap a real polyline to a sequence of adjacent grid centers at <=0.5 m steps.

    This is ONLY for model-cost estimation; the displayed NASA path remains
    the unmodified recorded NASA polyline. The snapped route is not claimed
    to be a true driven 2 m cell-center path.
    """
    inv = ~terrain.transform
    rows, cols = terrain.elev.shape
    step_m = min(float(terrain.cell_size) * 0.24, 0.5)
    out: list[tuple[int, int]] = []
    for a, b in zip(xy_line, xy_line[1:]):
        n = max(1, int(math.ceil(math.hypot(b[0]-a[0], b[1]-a[1]) / step_m)))
        for i in range(n + 1):
            t = i / n
            xf, yf = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            cf, rf = inv * (xf, yf)
            cell = (math.floor(rf), math.floor(cf))
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                raise ValueError("Recorded NASA path leaves the loaded terrain grid")
            if not out or out[-1] != cell:
                out.append(cell)
    return out


def evaluate_recorded_path_with_profile(
    model, terrain, xy_line: Sequence[XY]
) -> tuple[float | None, float | None, str, tuple[EnergySample, ...]]:
    """Model time, energy and graph data for NASA geometry if safely modelable.

    A modeled NASA energy profile is NOT telemetry. The display path remains
    recorded geometry; its modeled graph distances follow snapped 2 m moves.
    """
    try:
        cells = route_cells_near_recorded_line(terrain, xy_line)
        result = model.evaluate_path(cells)
        if not result.passable:
            return (None, None,
                    "Model rejects at least one recorded-track tile/transition; estimates unavailable.", ())
        profile = profile_from_moves(result.moves)
        if not profile:
            return None, None, "Invalid modeled movement costs along NASA track.", ()
        return (result.time_s, result.energy_j,
                "Modeled from NASA track snapped to 2 m cells (NOT measured).", profile)
    except (ValueError, IndexError) as exc:
        return None, None, f"Model estimate unavailable: {exc}", ()


def evaluate_recorded_path(model, terrain, xy_line: Sequence[XY]) -> tuple[float | None, float | None, str]:
    """Backward-compatible three-value summary for existing callers."""
    time_s, energy_j, note, _profile = evaluate_recorded_path_with_profile(
        model, terrain, xy_line)
    return time_s, energy_j, note


def cell_xy(terrain, cells: Sequence[tuple[int, int]]) -> tuple[XY, ...]:
    return tuple(tuple(map(float, terrain.transform * (c + 0.5, r + 0.5)))
                 for r, c in cells)


def nearest_passable(model, terrain, xy: XY, max_offset_m: float = 4.0) -> tuple[tuple[int, int], float]:
    """Explicitly snap a NASA marker to a nearby passable 2 m cell if required.

    Endpoints can be interpolated on a path and fall in a conservative blocked
    tile. Snap must be reported in the UI and is limited to 4 m, not silent.
    """
    col_f, row_f = (~terrain.transform) * xy
    base = (math.floor(row_f), math.floor(col_f))
    radius = int(math.ceil(max_offset_m / float(terrain.cell_size))) + 1
    options = []
    rows, cols = model.elev.shape
    for r in range(base[0] - radius, base[0] + radius + 1):
        for c in range(base[1] - radius, base[1] + radius + 1):
            if 0 <= r < rows and 0 <= c < cols:
                p = terrain.transform * (c + 0.5, r + 0.5)
                d = math.hypot(p[0] - xy[0], p[1] - xy[1])
                if d <= max_offset_m and model.evaluate_tile(r, c).passable:
                    options.append((d, (r, c)))
    if not options:
        raise ValueError(f"No passable tile within {max_offset_m:g} m of a selected rover point")
    options.sort()
    return options[0][1], options[0][0]


def calculate_real_routes(terrain, track_xy: Sequence[XY], start: RoverPoint,
                          end: RoverPoint) -> dict[str, RouteDisplay]:
    """Pure worker-side route calculation; also easy to unit-test headlessly."""
    from optimization import PathOptimizer
    from traversability import TraversabilityModel

    model = TraversabilityModel(terrain)
    if (start.along_m is not None and end.along_m is not None and
            start.segment_id == end.segment_id and len(track_xy) >= 2):
        actual = recorded_subpath(track_xy, start.along_m, end.along_m)
        recorded_length = cumulative_distances(actual)[-1]
        if recorded_length > 2500:
            t = e = None
            energy_profile = ()
            actual_note = "Recorded path exceeds the local terrain model; modeled estimates unavailable."
        else:
            t, e, actual_note, energy_profile = evaluate_recorded_path_with_profile(
                model, terrain, actual)
        if start.along_m > end.along_m:
            actual_note = "Reverse of NASA recorded chronology. " + actual_note
        actual_result = RouteDisplay(ACTUAL, tuple(actual), recorded_length, t, e,
                                     actual_note, energy_profile=energy_profile)
    else:
        actual_result = RouteDisplay(
            ACTUAL, (), None, None, None,
            "No continuous NASA recorded route links these selected positions.")
    result = {ACTUAL: actual_result}

    def not_found(name: str, reason: str) -> RouteDisplay:
        return RouteDisplay(name, (), None, None, None, reason)

    try:
        start_cell, start_snap = nearest_passable(model, terrain, (start.x_m, start.y_m))
        end_cell, end_snap = nearest_passable(model, terrain, (end.x_m, end.y_m))
    except ValueError as exc:
        result[FASTEST] = not_found(FASTEST, str(exc))
        result[LOWEST] = not_found(LOWEST, str(exc))
        return result

    optimized = PathOptimizer(model).optimize(start_cell, end_cell)
    snap_note = f"Grid endpoints offset {start_snap:.1f} m / {end_snap:.1f} m from NASA markers."
    for name, route in ((FASTEST, optimized.fastest),
                        (LOWEST, optimized.lowest_energy)):
        if route.found:
            result[name] = RouteDisplay(
                name, cell_xy(terrain, route.cells), route.distance_m,
                route.time_s, route.energy_j,
                "A* on 2 m tiles; no charging. " + snap_note,
                route.ascent_m, route.descent_m,
                profile_from_moves(route.moves),
            )
        else:
            result[name] = not_found(name, route.reason)
    return result


def closest_along_track(point: XY, line: Sequence[XY]) -> tuple[float | None, float]:
    """Return nearest arclength along NASA track and its distance in meters."""
    if len(line) < 2:
        return None, float("inf")
    best_dist, best_along, along = float("inf"), 0.0, 0.0
    for a, b in zip(line, line[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length_sq = dx*dx+dy*dy
        length = math.sqrt(length_sq)
        if length_sq:
            t = max(0.0, min(1.0,
                   ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length_sq))
            d = math.hypot(point[0]-(a[0]+t*dx), point[1]-(a[1]+t*dy))
            if d < best_dist:
                best_dist, best_along = d, along+t*length
        along += length
    return best_along, best_dist


def closest_on_segments(point: XY, segments: Sequence[Sequence[XY]]) -> tuple[int, float | None, float]:
    """Nearest *recorded* segment and arclength, without bridging data gaps."""
    best = (0, None, float("inf"))
    for segment_id, line in enumerate(segments):
        along, dist = closest_along_track(point, line)
        if dist < best[2]:
            best = segment_id, along, dist
    return best


def nice_grid_spacing(target_m: float) -> float:
    """Return an aesthetically useful scale in meters near target_m."""
    target_m = max(1e-5, float(target_m))
    order = 10**math.floor(math.log10(target_m))
    return min((order*x for x in (1, 2, 5, 10)),
               key=lambda v: abs(math.log(v/target_m)))


def source_dtm_bounds(path: Path, terrain) -> tuple[float, float, float, float]:
    """Full source GeoTIFF extent in the loaded terrain's local Mars meters."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(path) as src:
        tr = Transformer.from_crs(src.crs, terrain.crs, always_xy=True)
        b = src.bounds
        # Include middle points in case of minor projection bowing.
        points = [(x,y) for x in (b.left, (b.left+b.right)/2, b.right)
                 for y in (b.bottom, (b.bottom+b.top)/2, b.top)]
        xy = [tr.transform(x,y) for x,y in points]
    finite = [(x,y) for x,y in xy if math.isfinite(x) and math.isfinite(y)]
    if not finite:
        raise ValueError("Could not locate DTM bounds in Mars map coordinates")
    xs, ys = zip(*finite)
    return min(xs), min(ys), max(xs), max(ys)


def _colorize_dtm(elev: np.ndarray, bounds: tuple[float,float,float,float]) -> np.ndarray:
    """Draw georeferenced elevation as a Mars-toned heightmap, NOT a photo."""
    h, w = elev.shape
    valid = np.isfinite(elev)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = (56, 51, 48)  # no DTM data
    if not np.any(valid):
        return rgb
    low, high = np.percentile(elev[valid], [3, 97])
    norm = np.clip((elev-low)/max(float(high-low), .01), 0, 1)
    land = np.stack([152+90*norm, 92+75*norm, 64+52*norm], axis=-1)
    filled = np.where(valid, elev, np.nanmedian(elev[valid]))
    left, right, bottom, top = bounds
    pixel_x = max((right-left)/w, 1e-4)
    pixel_y = max((top-bottom)/h, 1e-4)
    gy, gx = np.gradient(filled, pixel_y, pixel_x)
    light = np.clip(.84 - .65*gx + .75*gy, .46, 1.26)
    land *= light[..., None]
    rgb[valid] = np.clip(land[valid], 0, 255).astype(np.uint8)
    return rgb


def terrain_rgb(terrain, bounds: tuple[float, float, float, float],
                width: int, height: int | None = None) -> np.ndarray:
    """Render the in-memory 2 m terrain crop, without allocating a full DTM."""
    height = width if height is None else height
    left, right, bottom, top = bounds
    inv = ~terrain.transform
    xs = left + (np.arange(width)+.5)*(right-left)/width
    ys = top - (np.arange(height)+.5)*(top-bottom)/height
    xx, yy = np.meshgrid(xs,ys)
    cc_f = inv.a*xx + inv.b*yy + inv.c
    rr_f = inv.d*xx + inv.e*yy + inv.f
    rr, cc = np.floor(rr_f).astype(int), np.floor(cc_f).astype(int)
    nrows,ncols = terrain.elev.shape
    inside = (rr>=0)&(rr<nrows)&(cc>=0)&(cc<ncols)
    z = np.full((height,width), np.nan)
    z[inside] = np.asarray(terrain.elev)[rr[inside],cc[inside]]
    return _colorize_dtm(z, bounds)


def source_dtm_rgb(path: Path, dst_crs, bounds: tuple[float,float,float,float],
                   width: int, height: int) -> np.ndarray:
    """Read only the visible source DTM pixels at screen resolution.

    This permits zooming and panning across the *entire* DTM GeoTIFF without
    loading the mosaic into RAM. Its overview shading is separate from the
    small 2 m grid built by terrain.py for each optimized route.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    left, right, bottom, top = bounds
    z = np.full((height,width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src,1), destination=z,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_bounds(left,bottom,right,top,width,height),
                  dst_crs=dst_crs, src_nodata=src.nodata, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, num_threads=2,
                  warp_mem_limit=64)
    return _colorize_dtm(z.astype(float), bounds)


class MarsRoverUI(ttk.Frame):
    """Interactive Tkinter map with route calculations and on-map comparisons.

    Full mission NASA geometry supplies selectable ~50 m markers by default.
    Four short test cases remain selectable. Any *map location* can also be
    selected, subject to the 500 m maximum
    start/end separation. A real NASA route only exists for on-track points.
    """

    MIN_METERS_PER_PIXEL = 0.12
    MAX_METERS_PER_PIXEL = 150.0
    MAP_PADDING = 1.14

    def __init__(self, parent: tk.Misc, *, cases_path: str | Path = CASES_FILE,
                 dtm_path: str | Path | None = None) -> None:
        super().__init__(parent, padding=6)
        self.parent = parent
        self.cases = read_cases(Path(cases_path))
        self.case_map: dict[str, dict] = {}
        self.case: dict | None = None
        self.selected_dtm = Path(dtm_path) if dtm_path else None
        self.active_dtm_path: Path | None = None
        self.terrain = None
        self.source_extent: tuple[float, float, float, float] | None = None
        self.track_xy: list[XY] = []
        self.track_segments_xy: list[list[XY]] = []
        self.points: list[RoverPoint] = []
        self.by_name: dict[str, RoverPoint] = {}
        self.free_points: dict[str, RoverPoint] = {}
        self.center_xy: XY = (0.0, 0.0)
        self.meters_per_pixel = 1.0
        self.routes: dict[str, RouteDisplay] = {}
        self._energy_window: tk.Toplevel | None = None
        self._energy_figure = None
        self._energy_canvas = None
        self._energy_visibility: dict[str, tk.BooleanVar] = {}
        self._energy_checkboxes: dict[str, ttk.Checkbutton] = {}
        self._map_geometry = (1, 1)
        self._marker_pixels: dict[str, XY] = {}
        self._canvas_image = None
        self._image_cache_key = None
        self._image_cache = None
        self._result_queue: queue.Queue = queue.Queue()
        self._generation = 0
        self._map_generation = 0
        self._map_render_key = None
        self._render_after = None
        self._draw_after = None
        self._fit_when_ready = True
        self._drag_origin: tuple[int, int] | None = None
        self._drag_was_pan = False
        self._active_card: str | None = None
        self._card_drag_origin: XY | None = None
        self._card_drag_start: XY | None = None
        self._card_drag_moved = False
        self._card_positions: dict[str, XY] = {}
        self.grid_terrain = None
        self._user_selected_case = False
        self._mission_load_id = 0
        self._mission_requested = False
        self.loading = False
        self.calculating = False
        self.selected_case = tk.StringVar()
        self.start_name = tk.StringVar()
        self.end_name = tk.StringVar()
        self.distance_text = tk.StringVar(value="Click a map location or choose a recorded marker")
        self.status_text = tk.StringVar(value="Loading recorded case tracks...")
        self.show_route = {name: tk.BooleanVar(value=(name != ACTUAL)) for name in ROUTE_NAMES}
        self.show_grid = tk.BooleanVar(value=True)
        self._build_ui()
        if self.cases:
            self.case_map = {c.get("id", f"case{i+1}"): c for i, c in enumerate(self.cases)}
            self.case_combo.configure(values=list(self.case_map))
            self.selected_case.set(next(iter(self.case_map)))
            self._change_case()
        else:
            self.case_combo.configure(state="disabled")
            self.calculate_button.configure(state="disabled")
            self.status_text.set("No NASA cases: run python cases.py to create data/cases.json")
        self.after(125, self._poll_worker_results)
        self.after_idle(self.center_map)
        self.after(350, lambda: self._load_full_mission(refresh=False, select_when_ready=True))

    def _build_ui(self) -> None:
        self.pack(fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(header, text="PERSEVERANCE | ROUTE EXPLORER",
                  font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text="Wheel: zoom  |  Drag map: pan  |  Drag route cards: move  |  Click: select",
                  foreground="#546275").pack(side="right", padx=9)

        # Map gets all the available vertical space; controls are on the bottom.
        self.canvas = tk.Canvas(self, bg="#332d2a", highlightthickness=0,
                                width=1150, height=680, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._begin_drag)
        self.canvas.bind("<B1-Motion>", self._drag_map)
        self.canvas.bind("<ButtonRelease-1>", self._finish_drag)
        self.canvas.bind("<ButtonPress-2>", self._begin_drag)
        self.canvas.bind("<B2-Motion>", self._drag_map)
        self.canvas.bind("<ButtonRelease-2>", self._finish_drag)
        self.canvas.bind("<ButtonPress-3>", self._begin_drag)
        self.canvas.bind("<B3-Motion>", self._drag_map)
        self.canvas.bind("<ButtonRelease-3>", self._finish_drag)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(.80, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(1.25, e.x, e.y))

        controls = ttk.Frame(self, padding=(4, 9, 4, 3))
        controls.grid(row=2, column=0, sticky="ew")
        for i in (1, 2):
            controls.columnconfigure(i, weight=1)
        col = ttk.Frame(controls)
        col.grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Label(col, text="Recorded drive area").pack(anchor="w")
        self.case_combo = ttk.Combobox(col, textvariable=self.selected_case,
                                       state="readonly", width=15)
        self.case_combo.pack(anchor="w", pady=(3, 0))
        self.case_combo.bind("<<ComboboxSelected>>", self._user_change_case)

        col = ttk.Frame(controls)
        col.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(col, text="Start").pack(anchor="w")
        self.start_combo = ttk.Combobox(col, textvariable=self.start_name,
                                        state="readonly", width=22)
        self.start_combo.pack(fill="x", pady=(3, 0))
        self.start_combo.bind("<<ComboboxSelected>>", lambda _e: self._selection_changed())

        col = ttk.Frame(controls)
        col.grid(row=0, column=2, sticky="ew", padx=(0, 15))
        ttk.Label(col, text="End (within 500 m of start)").pack(anchor="w")
        self.end_combo = ttk.Combobox(col, textvariable=self.end_name,
                                      state="readonly", width=22)
        self.end_combo.pack(fill="x", pady=(3, 0))
        self.end_combo.bind("<<ComboboxSelected>>", lambda _e: self._selection_changed())

        buttons = ttk.Frame(controls)
        buttons.grid(row=0, column=3, sticky="e")
        for label, cmd in (("+", lambda: self._zoom_at(.75)),
                           ("-", lambda: self._zoom_at(1.3333)),
                           ("Fit map", self.center_map),
                           ("DTM...", self.choose_dtm),
                           ("Refresh NASA", self.refresh_full_mission),
                           ("Clear", self.clear_selection)):
            ttk.Button(buttons, text=label, width=(7 if len(label) < 4 else
                        13 if label == "Refresh NASA" else 8),
                       command=cmd).pack(side="left", padx=2)
        self.calculate_button = ttk.Button(buttons, text="Calculate routes",
                                            command=self.calculate_paths)
        self.calculate_button.pack(side="left", padx=(7, 1))

        bottom = ttk.Frame(self, padding=(4, 2, 4, 2))
        bottom.grid(row=3, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.distance_text,
                  foreground="#334155").pack(side="left")
        ttk.Checkbutton(bottom, text="2 m grid (close zoom)", variable=self.show_grid,
                        command=self._schedule_redraw).pack(side="left", padx=15)
        self.energy_button = ttk.Button(bottom, text="Energy graphs",
                                        command=self.open_energy_graphs, state="disabled")
        self.energy_button.pack(side="left", padx=(0, 12))
        ttk.Label(bottom, textvariable=self.status_text,
                  foreground="#6d5947", wraplength=780).pack(side="right", padx=(10, 0))

    def _user_change_case(self, _event=None) -> None:
        self._user_selected_case = True
        self._change_case()

    def refresh_full_mission(self) -> None:
        self._load_full_mission(refresh=True, select_when_ready=True)

    def _load_full_mission(self, *, refresh: bool, select_when_ready: bool) -> None:
        if self._mission_requested and not refresh:
            return
        self._mission_requested = True
        self._mission_load_id += 1
        mission_id = self._mission_load_id
        reference = self.cases[0] if self.cases else None
        self.status_text.set("Reading NASA's full traverse (cached after download)...")

        def work():
            try:
                new_case = load_full_mission_case(refresh=refresh, reference_case=reference)
                self._result_queue.put((mission_id, "mission", new_case, select_when_ready))
            except Exception as exc:
                self._result_queue.put((mission_id, "mission_error", str(exc), None))
        threading.Thread(target=work, daemon=True).start()

    def _change_case(self) -> None:
        case = self.case_map.get(self.selected_case.get())
        if case is None:
            return
        self._generation += 1
        self._map_generation += 1
        self.case = case
        self.terrain = None
        self.grid_terrain = None
        self.source_extent = None
        self.active_dtm_path = None
        self.loading = self.calculating = False
        self.routes.clear()
        self._refresh_energy_graphs()
        self.free_points.clear()
        self._card_positions.clear()
        self._image_cache_key = None
        self._map_render_key = None
        self._refresh_track(projected=False)
        self.start_name.set("")
        self.end_name.set("")
        self._refresh_combos()
        self._fit_when_ready = True
        self.center_map()
        self.distance_text.set("Select a start point; a 500 m limit will appear")
        self.status_text.set(
            f"{len(self.points):,} selectable positions (~50 m along recorded NASA tracks). "
            "You can also click anywhere on the map.")
        self._update_button()
        local_dtm = self.selected_dtm or find_local_dtm(str(case.get("id", "")))
        if local_dtm is not None and local_dtm.is_file():
            self._start_terrain_load(local_dtm)
        else:
            self.status_text.set("No DTM loaded: select DTM... to enable route calculations.")
        self._schedule_redraw()

    def _refresh_track(self, projected: bool) -> None:
        assert self.case is not None
        lines = self.case.get("_segments_lonlat") or [self.case["nasa_route_lonlat"]]
        if projected:
            self.track_segments_xy = [projected_track(self.terrain, part) for part in lines]
        else:
            start, end = self.case["start"], self.case["end"]
            ref_lon = (float(start["lon"]) + float(end["lon"])) / 2
            ref_lat = (float(start["lat"]) + float(end["lat"])) / 2
            self.track_segments_xy = [local_xy(part, ref_lon, ref_lat) for part in lines]
        self.track_xy = [pt for part in self.track_segments_xy for pt in part]
        self.points = []
        counter = 0
        for segment_id, coords in enumerate(self.track_segments_xy):
            try:
                sample = sample_recorded_track(coords)
            except ValueError:
                continue
            for pt in sample:
                counter += 1
                self.points.append(RoverPoint(f"P{counter:04d}", pt.x_m, pt.y_m,
                                              pt.along_m, segment_id=segment_id))
        self.by_name = {p.name: p for p in self.points}
        self.by_name.update(self.free_points)

    def _refresh_combos(self) -> None:
        names = list(self.by_name)
        self.start_combo.configure(values=names)
        start = self.by_name.get(self.start_name.get())
        if start is None:
            self.end_combo.configure(values=names)
            return
        eligible = [p.name for p in self.by_name.values()
                    if p.name != start.name and
                    math.hypot(p.x_m - start.x_m, p.y_m - start.y_m) <= MAX_DISTANCE_M + 1e-6]
        self.end_combo.configure(values=eligible)

    def _start_terrain_load(self, path: Path) -> None:
        from copy import deepcopy
        if self.case is None:
            return
        generation = self._generation
        case = deepcopy(self.case)
        self.terrain = None
        self.loading = True
        self._update_button()
        self.status_text.set(f"Loading Mars elevation: {path.name}...")

        def work():
            try:
                from affine import Affine
                if getattr(Affine, "__matmul__", None) is None:
                    Affine.__matmul__ = Affine.__mul__
                from terrain import load_case
                # For the full mission, load only an existing short case as a
                # projection/terrain anchor, never a 45 km wide 2 m raster.
                t = load_case(case, dtm_path=str(path), margin_m=255.0, cell_size=2.0)
                try:
                    extent = source_dtm_bounds(path, t)
                except Exception:
                    # A georeferencing quirk should not disable the usable
                    # local 2 m terrain grid and route calculations.
                    extent = None
                self._result_queue.put((generation, "terrain", (t, extent), str(path)))
            except Exception as exc:
                self._result_queue.put((generation, "terrain_error", str(exc), str(path)))

        threading.Thread(target=work, daemon=True).start()

    def choose_dtm(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.parent, title="Choose Mars DTM GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if filename and self.case is not None:
            self.selected_dtm = Path(filename)
            self._generation += 1
            self._map_generation += 1
            self.routes.clear()
            self._refresh_energy_graphs()
            self._image_cache_key = None
            self._start_terrain_load(self.selected_dtm)
            self._schedule_redraw()

    def _poll_worker_results(self) -> None:
        try:
            while True:
                generation, kind, payload, detail = self._result_queue.get_nowait()
                if kind in ("mission", "mission_error"):
                    if generation != self._mission_load_id:
                        continue
                    if kind == "mission_error":
                        self.status_text.set(
                            "Full NASA traverse unavailable: " + str(payload)[:170] +
                            ". Run python fetch.py to download/cache it.")
                    else:
                        self.case_map[FULL_MISSION] = payload
                        self.case_combo.configure(values=[FULL_MISSION] +
                                                  [k for k in self.case_map if k != FULL_MISSION],
                                                  state="readonly")
                        if detail and not self._user_selected_case:
                            self.selected_case.set(FULL_MISSION)
                            self._change_case()
                        else:
                            self.status_text.set(
                                "Full NASA traverse loaded. Select Full Perseverance mission in the menu.")
                    continue
                if kind == "map":
                    if generation == self._map_generation and detail == self._map_render_key:
                        self._image_cache = Image.fromarray(payload, "RGB")
                        self._image_cache_key = detail
                        self._schedule_redraw()
                    continue
                if generation != self._generation:
                    continue
                if kind == "terrain":
                    self.terrain, self.source_extent = payload
                    self.grid_terrain = self.terrain
                    self.active_dtm_path = Path(detail)
                    self.loading = False
                    first, second = self.start_name.get(), self.end_name.get()
                    old_first = self.by_name.get(first)
                    old_second = self.by_name.get(second)
                    self.free_points.clear()
                    self._refresh_track(projected=True)
                    # Reproject any manually clicked points to the newly loaded frame.
                    if old_first is not None and first.startswith("Click "):
                        self.free_points[first] = self._reproject_free(old_first, first)
                    if old_second is not None and second.startswith("Click "):
                        self.free_points[second] = self._reproject_free(old_second, second)
                    self.by_name.update(self.free_points)
                    self.start_name.set(first if first in self.by_name else "")
                    self.end_name.set(second if second in self.by_name else "")
                    self._refresh_combos()
                    self._map_generation += 1
                    self._map_render_key = None
                    self._image_cache_key = None
                    self.center_map()
                    self._selection_changed()
                    self.status_text.set("DTM ready. Scroll to zoom, drag to pan; select any two points <=500 m apart.")
                elif kind == "terrain_error":
                    self.terrain = None
                    self.loading = False
                    self.status_text.set(f"Could not read {Path(detail).name}: {payload}")
                elif kind == "routes":
                    self.calculating = False
                    self.routes, self.grid_terrain = payload
                    self.status_text.set("Routes calculated. Times and energy are modeled, not NASA telemetry.")
                    self._refresh_energy_graphs()
                elif kind == "routes_error":
                    self.calculating = False
                    self.routes.clear()
                    self.status_text.set(f"Route calculation failed: {payload}")
                    self._refresh_energy_graphs()
                self._update_button()
                self._schedule_redraw()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(125, self._poll_worker_results)

    def _reproject_free(self, p: RoverPoint, name: str) -> RoverPoint:
        # Free-point coordinates chosen before DTM was loaded cannot be exactly
        # tied to Mars CRS unless stored as lon/lat. Prefer those when known.
        if p.lon is not None and p.lat is not None and self.terrain is not None:
            x, y = self.terrain._to_xy.transform(p.lon, p.lat)
            return RoverPoint(name, float(x), float(y), p.along_m, p.lon, p.lat, p.segment_id)
        # Without a georeference, retain the old approximate local offset.
        return RoverPoint(name, p.x_m, p.y_m, p.along_m, p.lon, p.lat, p.segment_id)

    def _selected_points(self) -> tuple[RoverPoint | None, RoverPoint | None]:
        return self.by_name.get(self.start_name.get()), self.by_name.get(self.end_name.get())

    def _valid_selection(self) -> tuple[bool, float]:
        a, b = self._selected_points()
        if a is None or b is None:
            return False, 0.0
        d = math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
        # No 50 m minimum: distinct points can be just centimeters apart.
        return a.name != b.name and d <= MAX_DISTANCE_M + 1e-6, d

    def _update_button(self) -> None:
        valid, _ = self._valid_selection()
        self.calculate_button.configure(
            state="normal" if valid and self.terrain is not None
            and not self.loading and not self.calculating else "disabled")
        self.energy_button.configure(
            state="normal" if any(route.energy_profile and route.energy_j is not None
                                  for route in self.routes.values()) else "disabled")

    def _selection_changed(self) -> None:
        if self.calculating:
            self._generation += 1  # invalidate in-progress route work
        self.calculating = False
        self.routes.clear()
        self._refresh_energy_graphs()
        a, b = self._selected_points()
        if a is None and self.end_name.get():
            self.end_name.set("")
            b = None
        if a is not None and b is not None:
            dist = math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
            if dist > MAX_DISTANCE_M + 1e-6 or a.name == b.name:
                self.end_name.set("")
                b = None
                self.status_text.set("End must be a different point within 500 m of the start.")
        if a is None:
            self.distance_text.set("Choose a start anywhere on the map")
        elif b is None:
            self.distance_text.set(f"Start {a.name} | choose an end within 500 m (no minimum)")
        else:
            self.distance_text.set(f"{a.name} to {b.name} | {dist:.1f} m straight-line distance")
        self._refresh_combos()
        self._update_button()
        self._schedule_redraw()

    def _select_map_point(self, px: float, py: float) -> None:
        if self.case is None:
            return
        world = self._pixel_to_world(px, py)
        # Prefer exactly sampled recorded-route positions when within 12 px.
        winner = None
        best = 12.0
        for point in self.points:
            sx, sy = self._world_to_pixel(point.x_m, point.y_m)
            distance = math.hypot(px - sx, py - sy)
            if distance <= best:
                winner, best = point, distance
        start, end = self._selected_points()
        new_start = start is None or end is not None
        if winner is None:
            name = "Click A" if new_start else "Click B"
            lon = lat = None
            if self.terrain is not None:
                lon, lat = self.terrain._to_lonlat.transform(*world)
            else:
                # Store geographical coordinates even in no-DTM preview mode.
                case_a, case_b = self.case["start"], self.case["end"]
                ref_lon = (float(case_a["lon"]) + float(case_b["lon"])) / 2
                ref_lat = (float(case_a["lat"]) + float(case_b["lat"])) / 2
                lon = ref_lon + math.degrees(world[0] /
                    (MARS_RADIUS_M * math.cos(math.radians(ref_lat))))
                lat = ref_lat + math.degrees(world[1] / MARS_RADIUS_M)
            segment_id, along, offset = closest_on_segments(world, self.track_segments_xy)
            # Only attribute a free click to a NASA-recorded segment if
            # it is within two meters of that real recorded line.
            if offset > 2.0:
                along = None
            winner = RoverPoint(name, world[0], world[1], along, lon, lat,
                               segment_id)
        if new_start:
            self.free_points.clear()
            self.by_name = {p.name: p for p in self.points}
            if winner.name.startswith("Click "):
                self.free_points["Click A"] = winner
                self.by_name["Click A"] = winner
            self.start_name.set(winner.name)
            self.end_name.set("")
            self.center_xy = (winner.x_m, winner.y_m)
            # When initially browsing a huge DTM, focus on the 500 m
            # selection zone. Do NOT zoom out if the user was already close:
            # they may want to choose points only centimeters apart.
            w, h = self._canvas_size()
            fit_radius_scale = 1040.0/max(1, min(w, h))
            if self.meters_per_pixel > fit_radius_scale:
                self.meters_per_pixel = fit_radius_scale
        else:
            assert start is not None
            if math.hypot(winner.x_m - start.x_m, winner.y_m - start.y_m) > MAX_DISTANCE_M + 1e-6:
                self.status_text.set("That point is more than 500 m from your start. Pick closer.")
                self._schedule_redraw()
                return
            if winner.name == start.name:
                self.status_text.set("Choose a second, distinct point (there is no minimum distance).")
                return
            if winner.name.startswith("Click "):
                self.free_points["Click B"] = winner
                self.by_name["Click B"] = winner
            self.end_name.set(winner.name)
            self.status_text.set("Selection ready. Click Calculate routes.")
        self._selection_changed()

    def calculate_paths(self) -> None:
        valid, distance = self._valid_selection()
        if not valid:
            messagebox.showwarning("Select points", "Choose two distinct points no more than 500 m apart.")
            return
        if self.terrain is None:
            messagebox.showwarning("DTM needed", "Load a HiRISE DTM before calculating paths.")
            return
        start, end = self._selected_points()
        assert start is not None and end is not None
        self._generation += 1
        generation = self._generation
        display_terrain = self.terrain
        dtm_path = self.active_dtm_path
        case = self.case
        track_segments = tuple(tuple(seg) for seg in self.track_segments_xy)
        self.routes.clear()
        self._refresh_energy_graphs()
        self.calculating = True
        self._update_button()
        self.status_text.set(f"Calculating routes for {distance:.1f} m separation...")
        self._schedule_redraw()

        def work():
            try:
                # Crop just the chosen 500 m area for A*. This keeps search
                # cost manageable even if the source DTM covers many kilometers.
                if dtm_path is not None:
                    from terrain import load_terrain
                    ll = []
                    for p in (start, end):
                        if p.lon is not None and p.lat is not None:
                            ll.append((float(p.lon), float(p.lat)))
                        else:
                            ll.append(display_terrain._to_lonlat.transform(p.x_m, p.y_m))
                    planning = load_terrain(str(dtm_path), ll[0], ll[1],
                                            margin_m=150.0, cell_size=2.0)
                    same_recorded = (start.along_m is not None and end.along_m is not None
                                     and start.segment_id == end.segment_id)
                    if same_recorded:
                        selected_line = (case.get("_segments_lonlat") or
                                         [case["nasa_route_lonlat"]])[start.segment_id]
                        planning_track = projected_track(planning, selected_line)
                    else:
                        planning_track = ()
                    def move_point(p: RoverPoint, lonlat: XY) -> RoverPoint:
                        x, y = planning._to_xy.transform(*lonlat)
                        return RoverPoint(p.name, x, y, p.along_m,
                                          *lonlat, p.segment_id)
                    s = move_point(start, ll[0]); e = move_point(end, ll[1])
                    results = calculate_real_routes(planning, planning_track, s, e)
                    from pyproj import Transformer
                    converter = Transformer.from_crs(planning.crs, display_terrain.crs,
                                                     always_xy=True)
                    rendered = {}
                    for name, route in results.items():
                        xy = tuple(tuple(map(float, converter.transform(*point)))
                                   for point in route.coordinates)
                        rendered[name] = RouteDisplay(name, xy, route.distance_m,
                                                      route.time_s, route.energy_j,
                                                      route.note, route.ascent_m, route.descent_m,
                                                      route.energy_profile)
                else:
                    selected_track = (
                        track_segments[start.segment_id]
                        if start.along_m is not None and end.along_m is not None and
                           start.segment_id == end.segment_id else ())
                    results = calculate_real_routes(display_terrain, selected_track, start, end)
                    rendered = results
                    planning = display_terrain
                self._result_queue.put((generation, "routes", (rendered, planning), None))
            except Exception as exc:
                self._result_queue.put((generation, "routes_error", str(exc), None))

        threading.Thread(target=work, daemon=True).start()

    def clear_selection(self) -> None:
        self.start_name.set("")
        self.end_name.set("")
        self.free_points.clear()
        self.by_name = {p.name: p for p in self.points}
        self._selection_changed()
        self.status_text.set("Selection cleared; actual NASA path remains hidden.")

    def center_map(self) -> None:
        if not self.track_xy:
            return
        xs, ys = zip(*self.track_xy)
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        if self.source_extent is not None:
            left, bottom, right, top = self.source_extent
            minx, miny = min(minx, left), min(miny, bottom)
            maxx, maxy = max(maxx, right), max(maxy, top)
        pad = max(70.0, max(maxx-minx, maxy-miny)*.06)
        minx -= pad; maxx += pad; miny -= pad; maxy += pad
        self.center_xy = ((minx + maxx)/2.0, (miny + maxy)/2.0)
        width, height = self._canvas_size()
        self.meters_per_pixel = min(self.MAX_METERS_PER_PIXEL, max(
            self.MIN_METERS_PER_PIXEL,
            self.MAP_PADDING * max((maxx - minx)/width, (maxy - miny)/height, .03)))
        self._schedule_redraw()

    def _on_canvas_configure(self, event) -> None:
        # Tk does not know the actual canvas size during the constructor.
        # Fit only once when a real geometry becomes available, without
        # resetting the user's later zoom whenever the window is resized.
        if self._fit_when_ready and event.width >= 150 and event.height >= 100:
            self._fit_when_ready = False
            self.center_map()
        else:
            self._schedule_redraw()

    def _canvas_size(self) -> tuple[int, int]:
        if hasattr(self, "canvas"):
            return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        return (1000, 600)

    def _map_bounds(self) -> tuple[float, float, float, float]:
        w, h = self._canvas_size()
        cx, cy = self.center_xy
        dx, dy = w*self.meters_per_pixel/2, h*self.meters_per_pixel/2
        return cx-dx, cx+dx, cy-dy, cy+dy

    def _world_to_pixel(self, x: float, y: float) -> XY:
        w, h = self._canvas_size()
        return (w/2 + (x-self.center_xy[0])/self.meters_per_pixel,
                h/2 - (y-self.center_xy[1])/self.meters_per_pixel)

    def _pixel_to_world(self, px: float, py: float) -> XY:
        w, h = self._canvas_size()
        return (self.center_xy[0] + (px-w/2)*self.meters_per_pixel,
                self.center_xy[1] - (py-h/2)*self.meters_per_pixel)

    def _zoom_at(self, factor: float, px: float | None = None,
                 py: float | None = None) -> None:
        w, h = self._canvas_size()
        px = w/2 if px is None else px
        py = h/2 if py is None else py
        wx, wy = self._pixel_to_world(px, py)
        new_scale = min(self.MAX_METERS_PER_PIXEL,
                        max(self.MIN_METERS_PER_PIXEL, self.meters_per_pixel*factor))
        if new_scale == self.meters_per_pixel:
            return
        self.meters_per_pixel = new_scale
        self.center_xy = (wx-(px-w/2)*new_scale, wy+(py-h/2)*new_scale)
        self._schedule_redraw()

    def _wheel_zoom(self, e) -> None:
        self._zoom_at(.80 if e.delta > 0 else 1.25, e.x, e.y)

    def _card_dimensions(self) -> tuple[int, int]:
        _w, h = self._canvas_size()
        return 272, min(110, max(85, (h - 110)//3))

    def _card_rect(self, name: str) -> tuple[float, float, float, float]:
        """Cards have screen positions and do NOT move with the map itself."""
        w, h = self._canvas_size()
        card_w, card_h = self._card_dimensions()
        if name not in self._card_positions:
            i = ROUTE_NAMES.index(name)
            self._card_positions[name] = (float(w-card_w-15), float(60+i*(card_h+10)))
        x, y = self._card_positions[name]
        x = min(max(x, 4.0), max(4.0, w-card_w-4.0))
        y = min(max(y, 37.0), max(37.0, h-card_h-6.0))
        self._card_positions[name] = (x, y)
        return x, y, x+card_w, y+card_h

    def _card_at(self, x: float, y: float) -> str | None:
        if not self.routes:
            return None
        # Last drawn card is on top if cards overlap.
        for name in reversed(ROUTE_NAMES):
            if name not in self.routes:
                continue
            x1, y1, x2, y2 = self._card_rect(name)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def _begin_drag(self, e) -> None:
        self._active_card = (self._card_at(e.x, e.y)
                             if getattr(e, "num", 1) == 1 else None)
        if self._active_card is not None:
            self._card_drag_origin = (e.x, e.y)
            self._card_drag_start = self._card_positions[self._active_card]
            self._card_drag_moved = False
            self._drag_origin = None
        else:
            self._drag_origin = (e.x, e.y)
            self._drag_was_pan = False

    def _drag_map(self, e) -> None:
        if self._active_card is not None:
            assert self._card_drag_origin is not None and self._card_drag_start is not None
            dx = e.x-self._card_drag_origin[0]
            dy = e.y-self._card_drag_origin[1]
            if abs(dx) + abs(dy) > 4:
                self._card_drag_moved = True
            if self._card_drag_moved:
                self._card_positions[self._active_card] = (
                    self._card_drag_start[0] + dx,
                    self._card_drag_start[1] + dy)
                self._card_rect(self._active_card)  # clamp position to map screen
                self._schedule_redraw()
            return
        if self._drag_origin is None:
            return
        ox, oy = self._drag_origin
        dx, dy = e.x-ox, e.y-oy
        if abs(dx) < 2 and abs(dy) < 2:
            return
        self.center_xy = (self.center_xy[0]-dx*self.meters_per_pixel,
                          self.center_xy[1]+dy*self.meters_per_pixel)
        self._drag_origin = (e.x, e.y)
        self._drag_was_pan = True
        self._schedule_redraw()

    def _finish_drag(self, e) -> None:
        if self._active_card is not None:
            name = self._active_card
            moved = self._card_drag_moved
            self._active_card = None
            self._card_drag_origin = None
            self._card_drag_start = None
            self._card_drag_moved = False
            if not moved:
                x1, y1, _x2, _y2 = self._card_rect(name)
                if x1+9 <= e.x <= x1+40 and y1+6 <= e.y <= y1+32:
                    self._toggle_route(name)
            return  # clicking cards never selects a rover point
        dragging = self._drag_was_pan
        self._drag_origin = None
        self._drag_was_pan = False
        if not dragging and getattr(e, "num", 1) == 1:
            self._select_map_point(e.x, e.y)

    def _schedule_redraw(self) -> None:
        if not hasattr(self, "canvas") or not self.canvas.winfo_exists():
            return
        if self._draw_after is not None:
            self.after_cancel(self._draw_after)
        self._draw_after = self.after(50, self._redraw)

    def _schedule_background(self, key, bounds, render_w: int, render_h: int) -> None:
        if self.active_dtm_path is None or self.terrain is None or Image is None:
            return
        if key != self._map_render_key:
            self._map_render_key = key
            self._map_generation += 1
        generation = self._map_generation
        path = self.active_dtm_path
        crs = self.terrain.crs
        if self._render_after is not None:
            self.after_cancel(self._render_after)

        def spawn():
            self._render_after = None
            # Ignore obsolete zoom/pan requests before accessing a large DTM.
            if generation != self._map_generation:
                return
            def work():
                try:
                    rgb = source_dtm_rgb(path, crs, bounds, render_w, render_h)
                    self._result_queue.put((generation, "map", rgb, key))
                except Exception:
                    try:
                        rgb = terrain_rgb(self.terrain, bounds, render_w, render_h)
                        self._result_queue.put((generation, "map", rgb, key))
                    except Exception:
                        pass
            threading.Thread(target=work, daemon=True).start()
        # Debounce mouse-wheel/drag events instead of launching a raster
        # reprojection for each individual pixel of movement.
        self._render_after = self.after(180, spawn)

    def _draw_map_background(self, w: int, h: int) -> None:
        canvas = self.canvas
        canvas.create_rectangle(0, 0, w, h, fill="#765743", outline="")
        if self.terrain is not None and Image is not None:
            bounds = self._map_bounds()
            # Compute at <= 1000 px along the longest dimension, then scale
            # on the Tk thread. The source DTM is read in a background worker.
            scale = min(1.0, 1000.0/max(w, h))
            rw = max(2, int(w*scale)); rh = max(2, int(h*scale))
            key = (str(self.active_dtm_path), id(self.terrain),
                   tuple(round(v, 2) for v in bounds), rw, rh)
            if key != self._image_cache_key:
                self._schedule_background(key, bounds, rw, rh)
            if key == self._image_cache_key and self._image_cache is not None:
                image = self._image_cache
                if image.size != (w, h):
                    image = image.resize((w, h), Image.Resampling.BILINEAR)
                self._canvas_image = ImageTk.PhotoImage(image, master=canvas)
                canvas.create_image(0, 0, anchor="nw", image=self._canvas_image)
            elif (self._image_cache is not None and self._image_cache_key
                  and key[:2] == self._image_cache_key[:2]):
                # While panning, show the last terrain image translated and
                # rescaled until the latest raster window finishes loading.
                old_l, old_r, old_b, old_t = self._image_cache_key[2]
                new_l, new_r, new_b, new_t = bounds
                new_scale = self.meters_per_pixel
                old_w = round((old_r-old_l)/new_scale)
                old_h = round((old_t-old_b)/new_scale)
                if 2 <= old_w <= 4000 and 2 <= old_h <= 4000:
                    image = self._image_cache.resize((old_w,old_h), Image.Resampling.BILINEAR)
                    x0 = (old_l-new_l)/new_scale
                    y0 = (new_t-old_t)/new_scale
                    self._canvas_image = ImageTk.PhotoImage(image, master=canvas)
                    canvas.create_image(x0, y0, anchor="nw", image=self._canvas_image)
            else:
                canvas.create_text(12, 42, text="Rendering Mars elevation...",
                                   anchor="w", fill="white",
                                   font=("Segoe UI", 11, "bold"))
        elif self.terrain is None:
            canvas.create_text(12, 42, text="NASA track coordinates | Load DTM for actual elevation",
                               anchor="w", fill="white", font=("Segoe UI", 11, "bold"))
        # At close zoom, show EXACT 2 m squares aligned with the most recent
        # terrain/planning grid transform. This is only a spatial reference:
        # it does NOT imply passability has been computed across the full DTM.
        if self.show_grid.get() and self.terrain is not None and self.meters_per_pixel <= 0.40:
            self._draw_two_meter_grid()
        step = nice_grid_spacing(110*self.meters_per_pixel)
        canvas.create_text(w-15, 15, text="N ^", anchor="ne", fill="#ffffff",
                           font=("Segoe UI", 12, "bold"))
        bar_px = step/self.meters_per_pixel
        canvas.create_line(20, h-26, 20+bar_px, h-26, fill="white", width=4)
        canvas.create_text(20, h-33, anchor="sw", fill="white",
                           text=f"{step:g} m", font=("Segoe UI", 10, "bold"))

    def _draw_two_meter_grid(self) -> None:
        """Draw a 2 x 2 m projected reference lattice at close zoom.

        The origin is taken from terrain.py's 2 m Affine transform, or from
        the latest route-planning crop. Each drawn cell is 2 m wide/high in
        Mars local projected meters. This is NOT a computed traversability map.
        """
        reference = self.grid_terrain or self.terrain
        if reference is None:
            return
        tr = reference.transform
        step_x = abs(float(tr.a))
        step_y = abs(float(tr.e))
        # terrain.py grids are north-up, with no rotated/skewed map axes.
        if (step_x < 1e-6 or step_y < 1e-6 or
                abs(float(tr.b)) > 1e-7 or abs(float(tr.d)) > 1e-7):
            return
        if step_x / self.meters_per_pixel < 4 or step_y / self.meters_per_pixel < 4:
            return
        left, right, bottom, top = self._map_bounds()
        w, h = self._canvas_size()
        ox, oy = float(tr.c), float(tr.f)
        first_x = math.floor((left-ox)/step_x)
        last_x = math.ceil((right-ox)/step_x)
        first_y = math.floor((bottom-oy)/step_y)
        last_y = math.ceil((top-oy)/step_y)
        if last_x-first_x > 500 or last_y-first_y > 500:
            return
        for k in range(first_x, last_x+1):
            x, _ = self._world_to_pixel(ox+k*step_x, self.center_xy[1])
            self.canvas.create_line(x, 0, x, h, fill="#b4ded6", width=1,
                                    stipple="gray50", tags="two_meter_grid")
        for k in range(first_y, last_y+1):
            _, y = self._world_to_pixel(self.center_xy[0], oy+k*step_y)
            self.canvas.create_line(0, y, w, y, fill="#b4ded6", width=1,
                                    stipple="gray50", tags="two_meter_grid")
        self.canvas.create_rectangle(9, 39, 258, 59, fill="#182631", outline="#a2cfc5")
        self.canvas.create_text(17, 49, text="2 m x 2 m reference grid (not hazards)",
                                anchor="w", fill="#c8e5e0", font=("Segoe UI", 9))

    def _draw_selection_zone(self) -> None:
        a, b = self._selected_points()
        if a is None or b is not None:
            return
        x, y = self._world_to_pixel(a.x_m, a.y_m)
        radius = MAX_DISTANCE_M/self.meters_per_pixel
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                outline="#6ee7b7", width=2, dash=(9, 6))
        label_y = max(52, min(self._canvas_size()[1]-50, y-radius-12))
        self.canvas.create_text(x, label_y, text="500 m end-point limit",
                                fill="#bafbe2", font=("Segoe UI", 10, "bold"))

    def _draw_markers(self) -> None:
        c = self.canvas
        w, h = self._canvas_size()
        a, b = self._selected_points()
        self._marker_pixels.clear()
        occupied: set[tuple[int, int]] = set()
        for p in self.points:
            x, y = self._world_to_pixel(p.x_m, p.y_m)
            if -10 <= x <= w+10 and -10 <= y <= h+10:
                # All points remain selectable and available in dropdowns,
                # but overlapped markers are drawn only once at overview zoom.
                bucket = (int(x//12), int(y//12))
                if bucket in occupied:
                    continue
                occupied.add(bucket)
                self._marker_pixels[p.name] = (x, y)
                radius = 4 if self.meters_per_pixel > 3 else 5
                color = "#263340"
                if a is not None and b is None and math.hypot(p.x_m-a.x_m, p.y_m-a.y_m) > MAX_DISTANCE_M:
                    color = "#6f6664"  # unavailable marker
                c.create_oval(x-radius, y-radius, x+radius, y+radius,
                              fill=color, outline="#f4e4cb", width=1)
                if self.meters_per_pixel < 1.2:
                    c.create_text(x+7, y-7, text=p.name, anchor="w", fill="white",
                                  font=("Segoe UI", 8, "bold"))
        for p, color, tag in ((a, "#34efd8", "START"), (b, "#ffe779", "END")):
            if p is None:
                continue
            x, y = self._world_to_pixel(p.x_m, p.y_m)
            c.create_oval(x-10, y-10, x+10, y+10,
                          fill=color, outline="white", width=3)
            c.create_text(x+14, y-14, text=f"{tag}: {p.name}", anchor="w",
                          fill="white", font=("Segoe UI", 10, "bold"))

    def _draw_route_lines(self) -> None:
        c = self.canvas
        for name in ROUTE_NAMES:
            route = self.routes.get(name)
            if route is None or not self.show_route[name].get() or len(route.coordinates) < 2:
                continue
            color, dash = ROUTE_STYLES[name]
            pixels = [v for xy in route.coordinates for v in self._world_to_pixel(*xy)]
            c.create_line(*pixels, fill="#17212b", width=7, smooth=False)
            c.create_line(*pixels, fill=color, width=4, dash=dash,
                          capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _toggle_route(self, name: str) -> None:
        self.show_route[name].set(not self.show_route[name].get())
        self._schedule_redraw()

    def open_energy_graphs(self) -> None:
        """Show energy comparisons without shrinking or obscuring the map canvas.

        These plots reuse the existing cost model's directional moves and are
        created lazily, so the UI still loads if Matplotlib is not installed.
        """
        if not any(r.energy_profile and r.energy_j is not None
                   for r in self.routes.values()):
            messagebox.showinfo("Energy graphs", "Calculate routes with valid energy estimates first.")
            return
        if self._energy_window is not None and self._energy_window.winfo_exists():
            self._energy_window.lift()
            self._energy_window.focus_set()
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except ImportError:
            messagebox.showerror("Matplotlib required",
                                 "Install Matplotlib in your VS Code environment:\n"
                                 "python -m pip install matplotlib")
            return

        window = tk.Toplevel(self)
        window.title("Perseverance | Modeled route energy comparison")
        window.geometry("1320x660")
        window.minsize(950, 480)
        self._energy_window = window
        panel = ttk.Frame(window, padding=(12, 10))
        panel.pack(fill="both", expand=True)
        top = ttk.Frame(panel)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="ENERGY COMPARISON", font=("Segoe UI", 15, "bold")).pack(side="left")
        ttk.Label(top, text="Show routes:", padding=(20, 0, 5, 0)).pack(side="left")
        self._energy_visibility = {}
        self._energy_checkboxes = {}
        for name in ROUTE_NAMES:
            route = self.routes.get(name)
            var = tk.BooleanVar(master=window, value=self.show_route[name].get())
            self._energy_visibility[name] = var
            cb = ttk.Checkbutton(top, text=name, variable=var,
                                 command=self._refresh_energy_graphs)
            cb.pack(side="left", padx=(4, 6))
            self._energy_checkboxes[name] = cb
            if route is None or not route.energy_profile:
                cb.configure(state="disabled")
        ttk.Button(top, text="Close", command=window.destroy).pack(side="right")

        fig = Figure(figsize=(13, 5.3), dpi=96, facecolor="#ffffff")
        self._energy_figure = fig
        self._energy_canvas = FigureCanvasTkAgg(fig, master=panel)
        self._energy_canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self._energy_canvas, panel, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", side="bottom")
        ttk.Label(panel,
                  text=("All energy estimates use the traversability model (no charging). "
                        "NASA-track energy is modeled from 2 m grid moves, NOT measured telemetry. "
                        "Checkboxes affect this graph window only. Use the chart toolbar to save images."),
                  wraplength=1200, foreground="#64748b").pack(side="bottom", fill="x", pady=(6, 2))

        def clear_window_refs(event) -> None:
            if event.widget is window:
                self._energy_window = None
                self._energy_figure = None
                self._energy_canvas = None
                self._energy_visibility = {}
                self._energy_checkboxes = {}
        window.bind("<Destroy>", clear_window_refs, add="+")
        self._refresh_energy_graphs()

    def _refresh_energy_graphs(self) -> None:
        """Refresh all plots on the Tk main thread after new route results."""
        if (self._energy_window is None or not self._energy_window.winfo_exists()
                or self._energy_figure is None or self._energy_canvas is None):
            return
        for name, cb in self._energy_checkboxes.items():
            route = self.routes.get(name)
            cb.configure(state="normal" if route is not None and route.energy_profile
                         and route.energy_j is not None else "disabled")
        fig = self._energy_figure
        fig.clear()
        ax_total, ax_cumulative, ax_intensity = fig.subplots(1, 3)
        available = [name for name in ROUTE_NAMES
                     if (self._energy_visibility.get(name) is not None
                         and self._energy_visibility[name].get()
                         and name in self.routes
                         and self.routes[name].energy_j is not None
                         and self.routes[name].energy_profile)]
        if not available:
            for ax in (ax_total, ax_cumulative, ax_intensity):
                ax.text(.5, .5, "No energy data selected.\nCalculate routes or enable a route.",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, color="#64748b")
                ax.set_axis_off()
        else:
            for index, name in enumerate(available):
                route = self.routes[name]
                color = ROUTE_STYLES[name][0]
                samples = route.energy_profile
                xs = [s.distance_m for s in samples]
                cumulative_wh = [s.cumulative_energy_j / 3600.0 for s in samples]
                totals_wh = route.energy_j / 3600.0
                ax_total.bar(index, totals_wh, color=color, width=.58)
                ax_total.text(index, totals_wh, f"{totals_wh:.2f}",
                              ha="center", va="bottom", fontsize=9)
                ax_cumulative.plot(xs, cumulative_wh, color=color, linewidth=2,
                                   label=name)
                step_points = [(s.distance_m, s.move_energy_j / 3600.0 / s.move_distance_m)
                               for s in samples[1:] if s.move_distance_m > 0]
                if step_points:
                    step_x, intensity = zip(*step_points)
                    ax_intensity.plot(step_x, intensity, color=color, linewidth=1.4,
                                      alpha=.85, label=name)
            ax_total.set_xticks(list(range(len(available))),
                                [name.replace(" path", "") for name in available],
                                rotation=18, ha="right")
            ax_total.set_ylabel("Energy (Wh)")
            ax_total.set_title("Total modeled energy")
            ax_total.set_ylim(bottom=0)
            ax_total.margins(y=.18)
            ax_cumulative.set_title("Cumulative energy vs. distance")
            ax_cumulative.set_xlabel("Traveled distance (m)")
            ax_cumulative.set_ylabel("Cumulative energy (Wh)")
            ax_intensity.set_title("Energy required per meter")
            ax_intensity.set_xlabel("Traveled distance (m)")
            ax_intensity.set_ylabel("Energy intensity (Wh/m)")
            for ax in (ax_cumulative, ax_intensity):
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.legend(loc="best", fontsize=8)
        for ax in (ax_total, ax_cumulative, ax_intensity):
            if ax.axison:
                ax.grid(alpha=.22, axis="y")
                ax.set_axisbelow(True)
        fig.subplots_adjust(left=.06, right=.985, bottom=.18, top=.88, wspace=.35)
        self._energy_canvas.draw_idle()

    def _draw_route_cards(self) -> None:
        if not self.routes:
            return
        w, h = self._canvas_size()
        c = self.canvas
        for name in ROUTE_NAMES:
            route = self.routes.get(name)
            if route is None:
                continue
            color, _ = ROUTE_STYLES[name]
            x1, y1, x2, y2 = self._card_rect(name)
            if self.show_route[name].get() and len(route.coordinates) >= 2:
                # Point from line to the closest edge of the movable card.
                target_x, target_y = (x1+x2)/2, (y1+y2)/2
                possible = [self._world_to_pixel(*point) for point in
                            route.coordinates[::max(1, len(route.coordinates)//150)]]
                visible = [(px, py) for px, py in possible
                           if -8 <= px <= w+8 and -8 <= py <= h+8]
                if visible:
                    px, py = min(visible,
                                 key=lambda pt: (pt[0]-target_x)**2+(pt[1]-target_y)**2)
                    edge_x = min(max(px, x1), x2)
                    edge_y = min(max(py, y1), y2)
                    if x1 < edge_x < x2 and y1 < edge_y < y2:
                        distances = ((abs(edge_x-x1), (x1,edge_y)),
                                     (abs(edge_x-x2), (x2,edge_y)),
                                     (abs(edge_y-y1), (edge_x,y1)),
                                     (abs(edge_y-y2), (edge_x,y2)))
                        edge_x,edge_y = min(distances,key=lambda p:p[0])[1]
                    c.create_line(px,py,edge_x,edge_y,fill=color,width=2,dash=(5,4))
                    c.create_oval(px-4,py-4,px+4,py+4,fill=color,outline="white")
            c.create_rectangle(x1, y1, x2, y2, fill="#182631", outline=color, width=2)
            c.create_rectangle(x1+10, y1+10, x1+26, y1+26,
                               outline="white", width=2,
                               fill=color if self.show_route[name].get() else "#182631")
            if self.show_route[name].get():
                c.create_text(x1+18, y1+18, text="✓", fill="white",
                              font=("Segoe UI", 10, "bold"))
            c.create_text(x1+36,y1+18,text=name,anchor="w",fill="white",
                          font=("Segoe UI",11,"bold"))
            c.create_text(x2-10,y1+18,text="drag",anchor="e",
                          fill="#9aafbe",font=("Segoe UI",8,"italic"))
            if route.distance_m is None:
                info = "No route found"
                detail = route.note
            else:
                tm = "N/A" if route.time_s is None else f"{route.time_s/60:.1f} min"
                en = "N/A" if route.energy_wh is None else f"{route.energy_wh:.2f} Wh"
                info = f"{route.distance_m:.1f} m   |   {tm}"
                detail = f"Energy: {en}"
            c.create_text(x1+11, y1+46, anchor="w", fill="#e6f1f8",
                          text=info, font=("Segoe UI", 10))
            c.create_text(x1+11, y1+67, anchor="w", fill="#e6f1f8",
                          text=detail[:40], font=("Segoe UI", 10))
            if y2-y1 >= 96:
                note = ("Modeled, not NASA telemetry" if name == ACTUAL else
                        "2 m terrain / no charging")
                if not route.coordinates and name == ACTUAL:
                    note = "Available for continuous NASA track"
                c.create_text(x1+11,y2-12,anchor="w",fill="#b6c7d2",
                              text=note,font=("Segoe UI",8))

    def _redraw(self) -> None:
        self._draw_after = None
        if not self.canvas.winfo_exists():
            return
        w, h = self._canvas_size()
        if w < 150 or h < 100:
            return
        self.canvas.delete("all")
        self._draw_map_background(w, h)
        self._draw_selection_zone()
        self._draw_route_lines()
        self._draw_markers()
        self._draw_route_cards()
        # Top-left, always-on help/legend. This is NOT a NASA path line.
        self.canvas.create_rectangle(8, 8, 335, 31, fill="#182631", outline="#596574")
        self.canvas.create_text(17, 19, anchor="w",
                                text="Wheel: zoom  |  Map: pan  |  Cards: drag",
                                fill="white", font=("Segoe UI", 10))
        if self.routes and not self.show_route[ACTUAL].get():
            self.canvas.create_text(12, 55, anchor="w", fill="#fbd0af",
                                    text="NASA recorded path is OFF (use its card to enable)",
                                    font=("Segoe UI", 9))


def launch_ui(*, cases_path: str | Path = CASES_FILE,
              dtm_path: str | Path | None = None) -> None:
    """Keep the existing main.py import and entry point unchanged."""
    root = tk.Tk()
    root.title("Perseverance | Interactive Route Explorer")
    root.geometry("1400x890")
    root.minsize(970, 650)
    MarsRoverUI(root, cases_path=cases_path, dtm_path=dtm_path)
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
