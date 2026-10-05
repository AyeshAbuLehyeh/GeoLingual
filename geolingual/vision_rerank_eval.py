import argparse
import json

from geolingual.dataset_utils import load
from geolingual.geo_negatives import build_pools
from geolingual.rerank_model import RerankJudge, ground_text_block, sat_text_block, parse_ranking
from geolingual.dataset_utils import pano_image_path, sat_image_path


def build_lookups():
    records = load()
    pano_fields_by_id = {f"{r['city']}_{r['pano_file']}": r["pano_fields"] for r in records}
    pano_file_by_id = {f"{r['city']}_{r['pano_file']}": (r["city"], r["pano_file"]) for r in records}
    _, sat_by_city = build_pools(k_hard=1, k_random=0, seed_records=records)
    return pano_fields_by_id, pano_file_by_id, sat_by_city


def sat_lookup(sat_id, sat_by_city):
    city, filename = sat_id.split("_", 1)
    return city, filename, sat_by_city[city][sat_id]["fields"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "image", "both"], required=True)
    ap.add_argument("--subset", choices=["hard", "easy", "all", "sample"], default="hard")
    ap.add_argument("--sample_size", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pools_file", type=str, default="cache/vision_retrieval_top10_pools.json")
    ap.add_argument("--out_prefix", type=str, default="cache/vision_rerank")
    args = ap.parse_args()

    import random
    random.seed(0)

    pools = json.load(open(args.pools_file))
    if args.subset == "hard":
        pools = [p for p in pools if p["vision_rank_of_true"] != 1]
        random.shuffle(pools)
    elif args.subset == "easy":
        pools = [p for p in pools if p["vision_rank_of_true"] == 1]
        random.shuffle(pools)
    elif args.subset == "sample":
        pools = random.sample(pools, min(args.sample_size, len(pools)))
    if args.limit:
        pools = pools[:args.limit]

    print(f"Evaluating mode={args.mode} subset={args.subset} n_pools={len(pools)}")

    pano_fields_by_id, pano_file_by_id, sat_by_city = build_lookups()
    judge = RerankJudge()

    results = []
    hit1 = hit3 = hit5 = 0
    recovered = 0
    still_findable = 0

    for i, p in enumerate(pools):
        qid = p["query_id"]
        city, pano_file = pano_file_by_id[qid]
        query_text = ground_text_block(pano_fields_by_id[qid])
        query_image = pano_image_path(city, pano_file)

        candidates = []
        cand_id_by_num = {}
        for j, sat_id in enumerate(p["vision_top_k"], 1):
            c_city, c_filename, c_fields = sat_lookup(sat_id, sat_by_city)
            candidates.append({
                "text": sat_text_block(c_fields),
                "image_path": sat_image_path(c_city, c_filename),
            })
            cand_id_by_num[j] = sat_id

        raw = judge.rerank(args.mode, query_text, query_image, candidates)
        ranking = parse_ranking(raw, len(candidates))

        true_sat_id = p["true_sat_id"]
        true_num = next((num for num, sid in cand_id_by_num.items() if sid == true_sat_id), None)
        new_rank = ranking.index(true_num) + 1 if true_num is not None else None

        if true_num is not None:
            still_findable += 1
            if new_rank == 1:
                hit1 += 1
                recovered += 1
            if new_rank is not None and new_rank <= 3:
                hit3 += 1
            if new_rank is not None and new_rank <= 5:
                hit5 += 1

        results.append({
            "query_id": qid, "true_sat_id": true_sat_id, "true_num": true_num,
            "new_rank": new_rank, "vision_rank_of_true": p["vision_rank_of_true"],
            "raw": raw[:500],
        })

        if i % 25 == 0:
            print(f"  {i+1}/{len(pools)} done", flush=True)

    n = len(pools)
    metrics = {
        "mode": args.mode, "subset": args.subset, "n": n,
        "R@1_in_pool": hit1 / n * 100 if n else 0,
        "R@3_in_pool": hit3 / n * 100 if n else 0,
        "R@5_in_pool": hit5 / n * 100 if n else 0,
        "true_candidate_shown_rate": still_findable / n * 100 if n else 0,
    }
    if args.subset == "hard":
        metrics["recovery_rate_of_hard_queries"] = recovered / n * 100 if n else 0
    elif args.subset == "easy":
        metrics["retention_rate_of_easy_queries"] = recovered / n * 100 if n else 0

    print("\n=== Reranking results ===")
    print(json.dumps(metrics, indent=2))

    out_prefix = f"{args.out_prefix}_{args.mode}_{args.subset}"
    json.dump(metrics, open(f"{out_prefix}_metrics.json", "w"), indent=2)
    json.dump(results, open(f"{out_prefix}_detail.json", "w"))
    print(f"Saved -> {out_prefix}_metrics.json / _detail.json")


if __name__ == "__main__":
    main()
