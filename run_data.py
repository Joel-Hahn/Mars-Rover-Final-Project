"""run_data.py -- runs the whole data stage in order: fetch -> cases -> crop.
    python run_data.py
Stops with a clear message at the first step that fails.
"""
import json
import sys

import cases
import crop_dtm
import fetch


def main():
    print("== 1/3 fetch (R1) ==")
    for name in fetch.URLS:
        fetch.describe(fetch.load_geojson(name))

    print("\n== 2/3 cases (R2) ==")
    cs = cases.pick_cases()
    cases.CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    cases.CASES_FILE.write_text(json.dumps(cs, indent=1))
    for c in cs:
        for end in ("start", "end"):
            if c[end]["sol"] is not None:
                fetch.save_photos(c["id"], end, c[end]["sol"])
    cases.print_table(cs)

    print("\n== 3/3 crop DTM for teammates ==")
    crop_dtm.crop_all()
    print("\nDone. Share data/cases.json, data/dtm_crops/, data/photos/.")


if __name__ == "__main__":
    try:
        main()
    except fetch.FetchError as e:
        sys.exit(f"ERROR: {e}")
    except FileNotFoundError as e:
        sys.exit(f"ERROR: missing file: {e}\nDid you save the DTM to {fetch.DTM_LOCAL}?")
