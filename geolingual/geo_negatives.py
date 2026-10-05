import json
import math
import random
from collections import defaultdict

from geolingual.dataset_utils import load

random.seed(0)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_pools(k_hard=9, k_random=0, seed_records=None):
    records = seed_records if seed_records is not None else load()

    sat_by_city = defaultdict(dict)
    for r in records:
        sat_id = f"{r['city']}_{r['sat_file']}"
        if sat_id not in sat_by_city[r["city"]]:
            sat_by_city[r["city"]][sat_id] = {
                "lat": r["sat_lat"], "lon": r["sat_lon"], "fields": r["sat_fields"],
            }

    pools = []
    for r in records:
        city = r["city"]
        query_id = f"{city}_{r['pano_file']}"
        true_sat_id = f"{city}_{r['sat_file']}"
        plat, plon = r["pano_lat"], r["pano_lon"]
        if plat is None:
            continue

        others = [(sid, d) for sid, d in sat_by_city[city].items() if sid != true_sat_id]
        dists = [(sid, haversine_m(plat, plon, d["lat"], d["lon"])) for sid, d in others]
        dists.sort(key=lambda x: x[1])
        hard = [sid for sid, _ in dists[:k_hard]]

        rand_pool = [sid for sid, _ in dists[k_hard:]]
        random_negs = random.sample(rand_pool, min(k_random, len(rand_pool))) if k_random else []

        candidates = [true_sat_id] + hard + random_negs
        random.shuffle(candidates)

        dist_map = {sid: d for sid, d in dists}
        pools.append({
            "query_id": query_id,
            "city": city,
            "true_sat_id": true_sat_id,
            "candidates": candidates,
            "hard_neg_distances_m": {sid: round(dist_map[sid], 1) for sid in hard},
        })
    return pools, sat_by_city


if __name__ == "__main__":
    pools, sat_by_city = build_pools(k_hard=9, k_random=0)
    print(f"Built {len(pools)} candidate pools")
    for city in ["Chicago", "NewYork", "SanFrancisco", "Seattle"]:
        city_pools = [p for p in pools if p["city"] == city]
        n = len(city_pools)
        avg_nearest = sum(min(p["hard_neg_distances_m"].values()) for p in city_pools) / n
        print(f"{city}: {n} pools, avg nearest hard-negative distance = {avg_nearest:.1f} m")

    with open("cache/hard_negative_pools.json", "w") as f:
        json.dump(pools, f)
    print("Saved -> cache/hard_negative_pools.json")
