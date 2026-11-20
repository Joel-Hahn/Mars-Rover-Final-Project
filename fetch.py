"""fetch.py -- Stage 1 of Driving Jezero. Owner: Data & photos lead. Covers R1.

Downloads and caches:
  * M20_waypoints.json  (rover position at the end of each drive)
  * M20_traverse.json   (the rover's path as lines)
  * endpoint photos from NASA's raw-images feed (untested -> manual fallback)
The USGS HiRISE DTM is NOT downloaded whole by default; see dtm_source().

R1 pass condition: runs from cache; clear error if the site is down.
    python fetch.py            # fetch (or reuse cache)
    python fetch.py --refresh  # force re-download
"""
import argparse
import json
import sys
from pathlib import Path

import requests

try:
    HERE = Path(__file__).parent      # running as a .py script
except NameError:
    HERE = Path.cwd()                  # running in a Jupyter notebook
DATA = HERE / "data"
CACHE = DATA / "cache"
PHOTOS = DATA / "photos"

BASE = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json"
URLS = {
    "waypoints": f"{BASE}/M20_waypoints.json",
    "traverse": f"{BASE}/M20_traverse.json",
}

# USGS Mars 2020 TRN HiRISE DTM mosaic, 1 m/px, Eqc projection (CC0, large file).
DTM_URL = (
    "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/"
    "JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif"
)
DTM_LOCAL = DATA / "dtm" / "JEZ_hirise_soc_006_DTM_1m.tif"

RAW_API = "https://mars.nasa.gov/rss/api/"
HEADERS = {"User-Agent": "MCEN1030-driving-jezero/0.1 (student project)"}


class FetchError(RuntimeError):
    """Raised with a human-readable message when the network or a file fails."""


def _download_json(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        r.json()  # validate before we cache it
    except (requests.RequestException, ValueError) as e:
        raise FetchError(
            f"Could not download {url}\n  reason: {e}\n"
            f"  no cached copy at {dest}. Check your connection, or copy the "
            f"file into {dest.parent} by hand."
        ) from e
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest


def load_geojson(name: str, refresh: bool = False) -> dict:
    """name is 'waypoints' or 'traverse'. Returns the parsed GeoJSON dict."""
    path = _download_json(URLS[name], CACHE / f"M20_{name}.json", refresh)
    return json.loads(path.read_text())


def describe(geojson: dict, n: int = 3) -> None:
    """Print the property names -- run this first (the Nov 16 spike)."""
    feats = geojson["features"]
    print(f"{len(feats)} features; geometry types: "
          f"{sorted({f['geometry']['type'] for f in feats})}")
    print("property keys:", sorted(feats[0]["properties"].keys()))
    for f in feats[:n]:
        print(" ", f["geometry"]["type"], f["properties"])


def dtm_source() -> str:
    """Path/URL rasterio should open for the DTM.

    Local file if you downloaded it (DTM_LOCAL), otherwise the remote file via
    GDAL's /vsicurl/, which reads only the windows you ask for (needs the
    server to support HTTP range requests -- test it in the spike).
    """
    return str(DTM_LOCAL) if DTM_LOCAL.exists() else f"/vsicurl/{DTM_URL}"


# ---------------------------------------------------------------- photos ----
def photos_for_sol(sol: int, n: int = 3) -> list[dict]:
    """Query NASA's raw-image feed for one sol. UNTESTED against the live site:
    the parameter names come from the site's own web app. Returns a list of
    {'url', 'camera', 'sol', 'id'} dicts, Navcam/Mastcam-Z first.
    """
    params = {
        "feed": "raw_images", "category": "mars2020", "feedtype": "json",
        "num": 100, "page": 0, "order": "sol desc",
        "condition_2": f"{sol}:sol:gte", "condition_3": f"{sol}:sol:lte",
    }
    try:
        r = requests.get(RAW_API, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        images = r.json().get("images", [])
    except (requests.RequestException, ValueError) as e:
        raise FetchError(f"Raw-images query failed for sol {sol}: {e}") from e

    out = []
    for im in images:
        files = im.get("image_files", {})
        url = files.get("large") or files.get("full_res") or files.get("medium")
        cam = (im.get("camera") or {}).get("instrument", "")
        if url:
            out.append({"url": url, "camera": cam, "sol": im.get("sol", sol),
                        "id": im.get("imageid", "")})
    pref = ("NAVCAM", "MCZ", "FRONT_HAZCAM")  # wide, rover-eye views first
    out.sort(key=lambda d: 0 if d["camera"].startswith(pref) else 1)
    return out[:n]


def save_photos(case_id: str, end: str, sol: int, n: int = 2) -> list[Path]:
    """Download photos for one endpoint ('start'/'end') into data/photos/.
    Falls back to anything you hand-placed in data/photos/<case_id>/<end>/.
    """
    folder = PHOTOS / case_id / end
    folder.mkdir(parents=True, exist_ok=True)
    have = sorted(folder.glob("*.jp*g")) + sorted(folder.glob("*.png"))
    if have:
        return have  # cached or hand-picked
    try:
        found = photos_for_sol(sol, n)
    except FetchError as e:
        print(f"[photos] {e}\n[photos] fallback: drop 1-2 images in {folder}")
        return []
    saved = []
    for i, p in enumerate(found):
        ext = p["url"].rsplit(".", 1)[-1].split("?")[0]
        dest = folder / f"sol{p['sol']}_{i}.{ext}"
        try:
            r = requests.get(p["url"], headers=HEADERS, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            saved.append(dest)
        except requests.RequestException as e:
            print(f"[photos] skipped {p['url']}: {e}")
    if not saved:
        print(f"[photos] nothing saved; drop 1-2 images in {folder}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args([] if "ipykernel" in sys.modules else None)
    try:
        for name in URLS:
            print(f"== {name} ==")
            describe(load_geojson(name, args.refresh))
    except FetchError as e:
        raise SystemExit(f"ERROR: {e}")
    print("\nDTM will be read from:", dtm_source())
