"""cases.py -- Stage 2 of Driving Jezero. Owner: Data & photos lead. Covers R2.

Picks start/end waypoint pairs that are 50-500 m apart, both inside the DTM,
saves them to data/cases.json, downloads endpoint photos, and prints the case
table (R2's "verified by").  Also provides nasa_route() for R8's comparison.

    python cases.py            # pick 3+ cases, fetch photos
    python cases.py --no-photos
"""
import argparse
import json
import math
import sys

import numpy as np

import fetch

MARS_R = 3_396_190.0          # m; sphere used by the USGS Eqc DTM (not 3389.5 km)
MIN_M, MAX_M = 50.0, 500.0
CASES_FILE = fetch.DATA / "cases.json"


# ---------------------------------------------------------------- helpers ---
def waypoint_table(geojson: dict) -> list[dict]:
    """Flatten waypoint Point features into dicts, in drive order."""
    rows = []
    for f in geojson["features"]:
        if f["geometry"]["type"] != "Point":
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        rows.append({"lon": float(lon), "lat": float(lat),
                     "sol": _num(p.get("sol")), "props": p})
    rows.sort(key=lambda r: (r["sol"] if r["sol"] is not None else 1e9))
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def _num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def dist_m(a: dict, b: dict) -> float:
    """Straight-line distance (m), local equirectangular -- fine below ~1 km."""
    lat0 = math.radians((a["lat"] + b["lat"]) / 2)
    dx = MARS_R * math.cos(lat0) * math.radians(b["lon"] - a["lon"])
    dy = MARS_R * math.radians(b["lat"] - a["lat"])
    return math.hypot(dx, dy)


def traverse_lines(traverse: dict) -> np.ndarray:
    """All traverse vertices as an (N, 2) array of [lon, lat], in file order."""
    pts = []
    for f in traverse["features"]:
        g = f["geometry"]
        if g["type"] == "LineString":
            pts.extend(g["coordinates"])
        elif g["type"] == "MultiLineString":
            for line in g["coordinates"]:
                pts.extend(line)
    return np.array([p[:2] for p in pts], dtype=float)


def nasa_route(traverse_xy: np.ndarray, a: dict, b: dict) -> np.ndarray:
    """The part of NASA's traverse between a and b (for the 4th route in R8).
    Snaps each endpoint to the nearest traverse vertex and slices between them.
    Returns [lon, lat] rows. Check this visually -- the overlay plot is the test.
    """
    def nearest(w):
        d = (traverse_xy[:, 0] - w["lon"]) ** 2 + (traverse_xy[:, 1] - w["lat"]) ** 2
        return int(np.argmin(d))
    i, j = nearest(a), nearest(b)
    lo, hi = sorted((i, j))
    seg = traverse_xy[lo:hi + 1]
    return seg if i <= j else seg[::-1]


def path_length_m(xy: np.ndarray) -> float:
    return sum(dist_m({"lon": p[0], "lat": p[1]}, {"lon": q[0], "lat": q[1]})
               for p, q in zip(xy[:-1], xy[1:]))


# -------------------------------------------------------------- DTM check ---
def make_dtm_checker():
    """Returns f(lon, lat, margin_m) -> True if that point plus a margin lies
    inside the DTM footprint AND on valid (non-nodata) data. Uses rasterio +
    pyproj so the Mars projection comes from the file itself.
    If rasterio is missing or the DTM is unreachable, returns None (warns).
    """
    try:
        import rasterio
        from pyproj import CRS, Transformer
        ds = rasterio.open(fetch.dtm_source())
        tf = Transformer.from_crs(CRS.from_wkt(ds.crs.to_wkt()).geodetic_crs,
                                  CRS.from_wkt(ds.crs.to_wkt()), always_xy=True)
    except Exception as e:  # noqa: BLE001 - any failure -> skip check
        print(f"[dtm] coverage check skipped ({e}). Install rasterio/pyproj or "
              f"download the DTM to {fetch.DTM_LOCAL}")
        return None

    def covered(lon, lat, margin_m=50.0):
        x, y = tf.transform(lon, lat)
        l, b, r, t = ds.bounds
        if not (l + margin_m < x < r - margin_m and b + margin_m < y < t - margin_m):
            return False
        win = rasterio.windows.from_bounds(x - margin_m, y - margin_m,
                                           x + margin_m, y + margin_m, ds.transform)
        arr = ds.read(1, window=win, masked=True)
        return arr.size > 0 and not np.ma.is_masked(arr) and not arr.mask.any()
    return covered


# ----------------------------------------------------------- case picking ---
def pick_cases(n_cases: int = 4, wanted=(80, 180, 300, 450)) -> list[dict]:
    wp = waypoint_table(fetch.load_geojson("waypoints"))
    tr = traverse_lines(fetch.load_geojson("traverse"))
    covered = make_dtm_checker()
    ok_cache: dict[int, bool] = {}

    def in_dtm(w):
        if covered is None:
            return True
        if w["idx"] not in ok_cache:
            ok_cache[w["idx"]] = covered(w["lon"], w["lat"])
        return ok_cache[w["idx"]]

    # Look at pairs a few drives apart; keep those 50-500 m apart.
    cands = []
    for i, a in enumerate(wp):
        for j in range(i + 1, min(i + 15, len(wp))):
            b = wp[j]
            d = dist_m(a, b)
            if MIN_M <= d <= MAX_M:
                cands.append((a, b, d))

    chosen, used = [], set()
    for target in list(wanted)[:n_cases]:           # spread over the 50-500 range
        for a, b, d in sorted(cands, key=lambda c: abs(c[2] - target)):
            if a["idx"] in used or b["idx"] in used:
                continue
            if not (in_dtm(a) and in_dtm(b)):
                continue
            seg = nasa_route(tr, a, b)
            if len(seg) < 2:
                continue
            used |= {a["idx"], b["idx"]}
            chosen.append({
                "id": f"case{len(chosen) + 1}",
                "start": {k: a[k] for k in ("idx", "lon", "lat", "sol")},
                "end": {k: b[k] for k in ("idx", "lon", "lat", "sol")},
                "straight_m": round(d, 1),
                "nasa_path_m": round(path_length_m(seg), 1),
                "nasa_route_lonlat": seg.tolist(),
            })
            break
    return chosen


def print_table(cases: list[dict]) -> None:
    print(f"{'id':7}{'start sol':>10}{'end sol':>9}{'straight m':>12}"
          f"{'NASA path m':>13}  photos")
    for c in cases:
        n = len(list((fetch.PHOTOS / c['id']).rglob("*.*")))
        print(f"{c['id']:7}{c['start']['sol']!s:>10}{c['end']['sol']!s:>9}"
              f"{c['straight_m']:>12}{c['nasa_path_m']:>13}  {n}")
    if len(cases) < 3:
        print("\nWARNING: fewer than 3 cases (R2 needs 3+). Widen the search "
              "window or check DTM coverage.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-photos", action="store_true")
    args = ap.parse_args([] if "ipykernel" in sys.modules else None)
    try:
        cases = pick_cases()
        CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CASES_FILE.write_text(json.dumps(cases, indent=1))
        if not args.no_photos:
            for c in cases:
                for end in ("start", "end"):
                    if c[end]["sol"] is not None:
                        fetch.save_photos(c["id"], end, c[end]["sol"])
    except fetch.FetchError as e:
        raise SystemExit(f"ERROR: {e}")
    print_table(cases)
    print(f"\nSaved {CASES_FILE}")
