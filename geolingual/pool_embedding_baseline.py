import argparse
import json

import torch
from sklearn.metrics.pairwise import cosine_similarity

from geolingual.spatial_verification import sample_queries, rank_metrics
from geolingual.similarity_and_retrieval import embed_bert, embed_tfidf, embed_mpnet, ground_text, sat_text


def evaluate_embedding_baseline(pools, query_lookup, sat_by_city, method="bert", device="cuda"):
    query_texts = {}
    for p in pools:
        r = query_lookup[p["query_id"]]
        query_texts[p["query_id"]] = ground_text(r["pano_fields"])

    cand_texts = {}
    for p in pools:
        for cand in p["candidates"]:
            if cand not in cand_texts:
                cand_texts[cand] = sat_text(sat_by_city[p["city"]][cand]["fields"])

    q_ids = list(query_texts.keys())
    c_ids = list(cand_texts.keys())
    q_texts = [query_texts[i] for i in q_ids]
    c_texts = [cand_texts[i] for i in c_ids]

    if method == "bert":
        q_emb = embed_bert(q_texts, device)
        c_emb = embed_bert(c_texts, device)
    elif method == "mpnet":
        q_emb = embed_mpnet(q_texts, device)
        c_emb = embed_mpnet(c_texts, device)
    elif method == "tfidf":
        c_emb, q_emb = embed_tfidf(c_texts, q_texts)
    else:
        raise ValueError(method)

    q_idx = {i: n for n, i in enumerate(q_ids)}
    c_idx = {i: n for n, i in enumerate(c_ids)}

    sims = cosine_similarity(q_emb, c_emb)

    pools_with_scores = []
    for p in pools:
        qi = q_idx[p["query_id"]]
        scored = [(cand, float(sims[qi, c_idx[cand]])) for cand in p["candidates"]]
        pools_with_scores.append({"query_id": p["query_id"], "true_sat_id": p["true_sat_id"], "scored": scored})

    return rank_metrics(pools_with_scores), pools_with_scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_city", type=int, default=100)
    ap.add_argument("--k_hard", type=int, default=9)
    ap.add_argument("--out", type=str, default="cache/pool_embedding_baseline_metrics.json")
    args = ap.parse_args()

    pools, query_lookup, sat_by_city = sample_queries(args.n_per_city, args.k_hard)
    print(f"Evaluating embedding baselines on {len(pools)} pools (pool size {1 + args.k_hard})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for method in ["bert", "mpnet", "tfidf"]:
        print(f"\n=== {method} (matched pools) ===")
        metrics, _ = evaluate_embedding_baseline(pools, query_lookup, sat_by_city, method=method, device=device)
        print(json.dumps(metrics, indent=2))
        results[method] = metrics

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")
