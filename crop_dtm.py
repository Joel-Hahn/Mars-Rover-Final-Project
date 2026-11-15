"""crop_dtm.py -- cut the big HiRISE DTM down to one small GeoTIFF per case,
so the Terrain lead can start without the full download.

Run AFTER cases.py has written data/cases.json:
    import crop_dtm; crop_dtm.crop_all()
Output: data/dtm_crops/<case_id>.tif  (case endpoints + margin, original 1 m cells)
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import from_bounds

import fetch

OUT = fetch.DATA / "dtm_crops"
MARGIN_M = 150.0   # extra ground around the endpoints so routes can detour


def crop_all(margin_m: float = MARGIN_M) -> list[Path]:
    cases = json.loads((fetch.DATA / "cases.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    with rasterio.open(fetch.dtm_source()) as ds:
        crs = CRS.from_wkt(ds.crs.to_wkt())
        tf = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        for c in cases:
            xs, ys = zip(*(tf.transform(c[k]["lon"], c[k]["lat"])
                           for k in ("start", "end")))
            # also include the NASA route so it never falls off the crop
            rx, ry = zip(*(tf.transform(lon, lat)
                           for lon, lat in c["nasa_route_lonlat"]))
            xs, ys = list(xs) + list(rx), list(ys) + list(ry)
            win = from_bounds(min(xs) - margin_m, min(ys) - margin_m,
                              max(xs) + margin_m, max(ys) + margin_m,
                              ds.transform).round_offsets().round_lengths()
            data = ds.read(1, window=win)
            profile = ds.profile | {"height": data.shape[0], "width": data.shape[1],
                                    "transform": ds.window_transform(win)}
            dest = OUT / f"{c['id']}.tif"
            with rasterio.open(dest, "w", **profile) as dst:
                dst.write(data, 1)
            nodata = ds.nodata
            bad = int(np.sum(data == nodata)) if nodata is not None else 0
            print(f"{c['id']}: {data.shape[1]}x{data.shape[0]} px, "
                  f"{bad} nodata px -> {dest}")
            written.append(dest)
    return written


if __name__ == "__main__":
    crop_all()
