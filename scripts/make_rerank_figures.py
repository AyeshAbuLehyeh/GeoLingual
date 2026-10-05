import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREFIX = "cache/vision_rerank_full591"
MODES = ["text", "image", "both"]
MODE_LABELS = {"text": "Text-only", "image": "Image-only", "both": "Image + Text"}
MODE_COLORS = {"text": "#4C72B0", "image": "#DD8452", "both": "#55A868"}
OUT_DIR = "figures"


def load_all(prefix):
    metrics, detail = {}, {}
    for mode in MODES:
        metrics[mode] = json.load(open(f"{prefix}_{mode}_hard_metrics.json"))
        detail[mode] = json.load(open(f"{prefix}_{mode}_hard_detail.json"))
    return metrics, detail


def fig_comparison_bar(metrics, out_path, n_pool=10):
    ks = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ks))
    width = 0.25
    for i, mode in enumerate(MODES):
        vals = [metrics[mode][f"R@{k}_in_pool"] for k in ks]
        ax.bar(x + (i - 1) * width, vals, width, label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    for i, k in enumerate(ks):
        ax.hlines(k / n_pool * 100, x[i] - 1.5 * width, x[i] + 1.5 * width,
                  colors="grey", linestyles="dashed", linewidth=1.2)
    ax.plot([], [], color="grey", linestyle="dashed", label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recovery rate on hard subset (%)")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def fig_rank_distribution(detail, out_path, n_pool=10):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bins = list(range(1, n_pool + 2))
    for ax, mode in zip(axes, MODES):
        ranks = [r["new_rank"] if r["new_rank"] is not None else n_pool + 1 for r in detail[mode]]
        counts = [ranks.count(b) for b in bins]
        labels = [str(b) if b <= n_pool else "N/A" for b in bins]
        colors = [MODE_COLORS[mode]] * n_pool + ["#999999"]
        ax.bar(labels, counts, color=colors)
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Rank of true match after reranking")
    axes[0].set_ylabel("Number of queries")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics, detail = load_all(PREFIX)
    fig_comparison_bar(metrics, f"{OUT_DIR}/rerank_comparison_bar.png")
    fig_rank_distribution(detail, f"{OUT_DIR}/rerank_rank_distribution.png")
