import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from geolingual.dataset_utils import load

OUT_DIR = "cache"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

GROUND_KEYS = ["road_topology_360", "sector_forward", "sector_backward",
               "spatial_layout", "road_markings", "orientation_cues",
               "distinctive_anchors", "environmental_context"]
SAT_KEYS = ["distinctive_anchors", "roof_fingerprints", "linear_sequence",
            "cardinal_orientation", "road_geometry", "shadow_analysis"]


def ground_text(fields):
    return " ".join(fields.get(k, "") for k in GROUND_KEYS)


def sat_text(fields):
    return " ".join(fields.get(k, "") for k in SAT_KEYS)


def build_unique_dbs(records):
    pano_db = {}
    sat_db = {}
    gt = {}
    city_of = {}
    for r in records:
        pano_id = f"{r['city']}_{r['pano_file']}"
        sat_id = f"{r['city']}_{r['sat_file']}"
        pano_db[pano_id] = ground_text(r["pano_fields"])
        if sat_id not in sat_db:
            sat_db[sat_id] = sat_text(r["sat_fields"])
        gt[pano_id] = sat_id
        city_of[pano_id] = r["city"]
    return pano_db, sat_db, gt, city_of


def embed_bert(texts, device, batch_size=32, max_length=256):
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embed"):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return normalize(np.concatenate(out, axis=0), norm="l2")


def embed_mpnet(texts, device, batch_size=64):
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    del model
    torch.cuda.empty_cache()
    return embs


def embed_tfidf(ref_texts, query_texts):
    vec = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.8,
                           ngram_range=(1, 2), stop_words="english",
                           lowercase=True, token_pattern=r"\b[a-zA-Z]{2,}\b")
    ref = normalize(vec.fit_transform(ref_texts).toarray(), norm="l2")
    query = normalize(vec.transform(query_texts).toarray(), norm="l2")
    return ref, query


def evaluate(query_ids, ref_ids, query_emb, ref_emb, gt, city_of, top_k=(1, 5, 10, 20)):
    max_k = max(top_k)
    recall = {k: 0 for k in top_k}
    per_city = defaultdict(lambda: defaultdict(int))
    per_city_n = defaultdict(int)
    detailed = {}

    sims = cosine_similarity(query_emb, ref_emb)
    for qi, qid in enumerate(query_ids):
        gt_id = gt[qid]
        order = np.argsort(sims[qi])[::-1][:max_k]
        top_ids = [ref_ids[i] for i in order]
        rank = top_ids.index(gt_id) + 1 if gt_id in top_ids else None
        city = city_of[qid]
        per_city_n[city] += 1
        for k in top_k:
            if rank is not None and rank <= k:
                recall[k] += 1
                per_city[city][k] += 1
        detailed[qid] = {"gt": gt_id, "rank": rank, "top5": top_ids[:5]}

    n = len(query_ids)
    metrics = {f"Recall@{k}": recall[k] / n * 100 for k in top_k}
    metrics["num_queries"] = n
    metrics["per_city"] = {
        city: {f"Recall@{k}": per_city[city][k] / per_city_n[city] * 100 for k in top_k} | {"n": per_city_n[city]}
        for city in per_city_n
    }
    return metrics, detailed


def main():
    records = load()
    pano_db, sat_db, gt, city_of = build_unique_dbs(records)
    pano_ids = list(pano_db.keys())
    sat_ids = list(sat_db.keys())
    pano_texts = [pano_db[i] for i in pano_ids]
    sat_texts = [sat_db[i] for i in sat_ids]
    print(f"Unique panorama queries: {len(pano_ids)}, unique satellite refs: {len(sat_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    all_metrics = {}

    print("\n=== BERT ===")
    bert_sat = embed_bert(sat_texts, device)
    bert_pano = embed_bert(pano_texts, device)
    metrics, detailed = evaluate(pano_ids, sat_ids, bert_pano, bert_sat, gt, city_of)
    all_metrics["bert"] = metrics
    json.dump(detailed, open(f"{OUT_DIR}/retrieval_bert_detailed.json", "w"))
    print(f"BERT  R@1={metrics['Recall@1']:.2f}  R@5={metrics['Recall@5']:.2f}  "
          f"R@10={metrics['Recall@10']:.2f}  R@20={metrics['Recall@20']:.2f}  (n={metrics['num_queries']})")

    sat_idx = {sid: i for i, sid in enumerate(sat_ids)}
    pano_idx = {pid: i for i, pid in enumerate(pano_ids)}
    pair_sims = defaultdict(list)
    for pid in pano_ids:
        city = city_of[pid]
        sim = float(np.dot(bert_pano[pano_idx[pid]], bert_sat[sat_idx[gt[pid]]]))
        pair_sims[city].append(sim)
    json.dump({c: v for c, v in pair_sims.items()}, open(f"{OUT_DIR}/bert_pair_cosine_by_city.json", "w"))
    overall = [s for v in pair_sims.values() for s in v]
    print(f"BERT true-pair cosine similarity: mean={np.mean(overall):.4f} std={np.std(overall):.4f}")

    print("\n=== all-mpnet-base-v2 ===")
    mpnet_sat = embed_mpnet(sat_texts, device)
    mpnet_pano = embed_mpnet(pano_texts, device)
    metrics, detailed = evaluate(pano_ids, sat_ids, mpnet_pano, mpnet_sat, gt, city_of)
    all_metrics["mpnet"] = metrics
    json.dump(detailed, open(f"{OUT_DIR}/retrieval_mpnet_detailed.json", "w"))
    print(f"MPNet R@1={metrics['Recall@1']:.2f}  R@5={metrics['Recall@5']:.2f}  "
          f"R@10={metrics['Recall@10']:.2f}  R@20={metrics['Recall@20']:.2f}")

    print("\n=== TF-IDF ===")
    tfidf_sat, tfidf_pano = embed_tfidf(sat_texts, pano_texts)
    metrics, detailed = evaluate(pano_ids, sat_ids, tfidf_pano, tfidf_sat, gt, city_of)
    all_metrics["tfidf"] = metrics
    json.dump(detailed, open(f"{OUT_DIR}/retrieval_tfidf_detailed.json", "w"))
    print(f"TFIDF R@1={metrics['Recall@1']:.2f}  R@5={metrics['Recall@5']:.2f}  "
          f"R@10={metrics['Recall@10']:.2f}  R@20={metrics['Recall@20']:.2f}")

    json.dump(all_metrics, open(f"{OUT_DIR}/retrieval_all_methods_metrics.json", "w"), indent=2)
    print(f"\nSaved -> {OUT_DIR}/retrieval_all_methods_metrics.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(8, 5))
    for city, sims in pair_sims.items():
        sims = np.array(sims)
        xs = np.linspace(sims.min() - 0.05, sims.max() + 0.05, 200)
        kde = gaussian_kde(sims)
        plt.plot(xs, kde(xs), label=city)
        plt.fill_between(xs, kde(xs), alpha=0.15)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(title="City")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kde_corrected.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, city in zip(axes.flat, pair_sims.keys()):
        city_pano_ids = [pid for pid in pano_ids if city_of[pid] == city][:50]
        city_sat_ids = [gt[pid] for pid in city_pano_ids]
        p_emb = np.stack([bert_pano[pano_idx[pid]] for pid in city_pano_ids])
        s_emb = np.stack([bert_sat[sat_idx[sid]] for sid in city_sat_ids])
        sim_mat = cosine_similarity(s_emb, p_emb)
        im = ax.imshow(sim_mat, cmap="viridis")
        ax.set_title(city)
        ax.set_xlabel("Panorama Embeddings")
        ax.set_ylabel("Satellite Embeddings")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_corrected.png", dpi=150)
    plt.close()
    print(f"Figures -> {FIG_DIR}/kde_corrected.png, {FIG_DIR}/heatmap_corrected.png")


if __name__ == "__main__":
    main()
