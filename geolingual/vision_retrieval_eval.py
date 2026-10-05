import argparse
import json

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from geolingual.dataset_utils import load
from geolingual.vision_embed import load_model, get_transforms, embed_images
from geolingual.dataset_utils import pano_image_path, sat_image_path


def build_unique_image_lists(records):
    pano_db = {}
    sat_db = {}
    gt = {}
    city_of = {}
    for r in records:
        pano_id = f"{r['city']}_{r['pano_file']}"
        sat_id = f"{r['city']}_{r['sat_file']}"
        if pano_id not in pano_db:
            pano_db[pano_id] = pano_image_path(r["city"], r["pano_file"])
        if sat_id not in sat_db:
            sat_db[sat_id] = sat_image_path(r["city"], r["sat_file"])
        gt[pano_id] = sat_id
        city_of[pano_id] = r["city"]
    return pano_db, sat_db, gt, city_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_prefix", type=str, default="cache/vision_retrieval")
    args = ap.parse_args()

    records = load()
    if args.limit:
        records = records[:args.limit]

    pano_db, sat_db, gt, city_of = build_unique_image_lists(records)
    pano_ids = list(pano_db.keys())
    sat_ids = list(sat_db.keys())
    print(f"Queries (panorama): {len(pano_ids)}   References (satellite): {len(sat_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading AuxGeo model...")
    model = load_model(device=device)
    sat_transform, ground_transform = get_transforms()

    print("Embedding satellite (reference) images...")
    sat_emb = embed_images([sat_db[i] for i in sat_ids], sat_transform, model, device=device, desc="sat embed")
    print("Embedding panorama (query) images...")
    pano_emb = embed_images([pano_db[i] for i in pano_ids], ground_transform, model, device=device, desc="pano embed")

    sims = cosine_similarity(pano_emb, sat_emb)

    top_k = args.top_k
    recall = {1: 0, 5: 0, 10: 0, 20: 0}
    pools = []
    for qi, qid in enumerate(pano_ids):
        order = np.argsort(sims[qi])[::-1]
        top_ids = [sat_ids[i] for i in order[:max(20, top_k)]]
        gt_id = gt[qid]
        rank = top_ids.index(gt_id) + 1 if gt_id in top_ids else None
        for k in recall:
            if rank is not None and rank <= k:
                recall[k] += 1
        pools.append({
            "query_id": qid, "city": city_of[qid], "true_sat_id": gt_id,
            "vision_top_k": top_ids[:top_k],
            "vision_rank_of_true": rank,
        })

    n = len(pano_ids)
    metrics = {f"Recall@{k}": v / n * 100 for k, v in recall.items()}
    metrics["num_queries"] = n
    print("\n=== Vision retrieval sanity check (compare to checkpoint's reported R@1=80.43%) ===")
    print(json.dumps(metrics, indent=2))

    json.dump(metrics, open(f"{args.out_prefix}_metrics.json", "w"), indent=2)
    json.dump(pools, open(f"{args.out_prefix}_top{top_k}_pools.json", "w"))
    print(f"\nSaved -> {args.out_prefix}_metrics.json, {args.out_prefix}_top{top_k}_pools.json")


if __name__ == "__main__":
    main()
