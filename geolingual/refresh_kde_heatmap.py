import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity

from geolingual.dataset_utils import load
from geolingual.similarity_and_retrieval import embed_bert, ground_text, sat_text

FIG_DIR = "figures"

pair_sims = json.load(open("cache/bert_pair_cosine_by_city.json"))

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
print(f"Saved -> {FIG_DIR}/kde_corrected.png")

records = load()
by_city = {}
for r in records:
    by_city.setdefault(r["city"], []).append(r)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, city in zip(axes.flat, ["Chicago", "NewYork", "SanFrancisco", "Seattle"]):
    recs = by_city[city][:50]
    pano_texts = [ground_text(r["pano_fields"]) for r in recs]
    sat_texts = [sat_text(r["sat_fields"]) for r in recs]
    p_emb = embed_bert(pano_texts, device, batch_size=50)
    s_emb = embed_bert(sat_texts, device, batch_size=50)
    sim_mat = cosine_similarity(s_emb, p_emb)
    im = ax.imshow(sim_mat, cmap="viridis")
    ax.set_title(city)
    ax.set_xlabel("Panorama Embeddings")
    ax.set_ylabel("Satellite Embeddings")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/heatmap_corrected.png", dpi=150)
plt.close()
print(f"Saved -> {FIG_DIR}/heatmap_corrected.png")
