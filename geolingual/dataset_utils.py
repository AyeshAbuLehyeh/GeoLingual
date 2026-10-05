import json
import os
import re

import pandas as pd

from geolingual.xml_utils import load_and_parse, is_valid
from geolingual.config import CITIES, DESCRIPTIONS_DIR, VIGOR_ROOT, VLM_DIR_NAME

CACHE_PATH = "cache/valid_pairs.jsonl"
STATS_PATH = "cache/valid_pairs_stats.json"

SAT_LATLON_RE = re.compile(r"satellite_(-?\d+\.\d+)_(-?\d+\.\d+)")


def sat_latlon(filename: str):
    m = SAT_LATLON_RE.search(filename)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def pano_latlon(filename: str):
    parts = filename.split(",")
    if len(parts) < 3:
        return None, None
    try:
        return float(parts[1]), float(parts[2])
    except ValueError:
        return None, None


def txt_path(modality: str, city: str, filename: str) -> str:
    base = os.path.basename(filename)
    txt_name = re.sub(r"\.(png|jpg)$", ".txt", base)
    return os.path.join(DESCRIPTIONS_DIR, modality, VLM_DIR_NAME, city, modality, txt_name)


def pano_image_path(city: str, filename: str) -> str:
    return os.path.join(VIGOR_ROOT, city, "panorama", filename)


def sat_image_path(city: str, filename: str) -> str:
    return os.path.join(VIGOR_ROOT, city, "satellite", filename)


def build(force: bool = False):
    if os.path.exists(CACHE_PATH) and not force:
        return

    os.makedirs("cache", exist_ok=True)
    stats = {}
    n_written = 0

    with open(CACHE_PATH, "w") as out:
        for city in CITIES:
            df = pd.read_csv(f"data/{city}.csv")
            n_rows = len(df)
            n_pano_valid = 0
            n_sat_valid = 0
            n_pair_valid = 0

            sat_cache = {}

            for _, row in df.iterrows():
                pano_file = row["panorama"]
                sat_file = row["satellite"]

                p_path = txt_path("panorama", city, pano_file)
                pano_fields = load_and_parse(p_path, "ground") if os.path.exists(p_path) else None
                pano_ok = is_valid(pano_fields, "ground")
                if pano_ok:
                    n_pano_valid += 1

                if sat_file not in sat_cache:
                    s_path = txt_path("satellite", city, sat_file)
                    sat_fields = load_and_parse(s_path, "satellite") if os.path.exists(s_path) else None
                    sat_ok = is_valid(sat_fields, "satellite")
                    sat_cache[sat_file] = (sat_fields, sat_ok)
                sat_fields, sat_ok = sat_cache[sat_file]

                if pano_ok and sat_ok:
                    n_pair_valid += 1
                    plat, plon = pano_latlon(pano_file)
                    slat, slon = sat_latlon(sat_file)
                    rec = {
                        "city": city,
                        "pano_file": os.path.basename(pano_file),
                        "sat_file": sat_file,
                        "pano_fields": pano_fields,
                        "sat_fields": sat_fields,
                        "pano_lat": plat, "pano_lon": plon,
                        "sat_lat": slat, "sat_lon": slon,
                    }
                    out.write(json.dumps(rec) + "\n")
                    n_written += 1

            n_sat_valid = sum(1 for _, ok in sat_cache.values() if ok)
            stats[city] = {
                "csv_rows": n_rows,
                "unique_satellite_tiles": len(sat_cache),
                "panorama_valid": n_pano_valid,
                "satellite_valid": n_sat_valid,
                "pair_valid": n_pair_valid,
            }
            print(f"{city}: {n_pair_valid} valid pairs (of {n_rows} rows, "
                  f"{n_pano_valid} pano-valid, {n_sat_valid}/{len(sat_cache)} sat-valid)")

    stats["TOTAL"] = {
        "pair_valid": sum(s["pair_valid"] for c, s in stats.items() if c != "TOTAL"),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nTotal valid pairs: {n_written} -> {CACHE_PATH}")
    print(f"Stats -> {STATS_PATH}")


def load():
    build(force=False)
    records = []
    with open(CACHE_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    build(force=True)
