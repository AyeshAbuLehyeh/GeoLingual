import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("cache/lexical_distribution.json") as f:
    data = json.load(f)

cities = list(data.keys())
landmarks = list(data[cities[0]]["ground"].keys())

for view in ["ground", "satellite"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(landmarks))
    width = 0.2
    for i, city in enumerate(cities):
        ratios = [data[city][view][lm][0] for lm in landmarks]
        ax.bar(x + i * width, ratios, width, label=city)
    ax.set_xticks(x + width * (len(cities) - 1) / 2)
    ax.set_xticklabels(landmarks)
    ax.set_ylabel("Fraction of descriptions mentioning landmark")
    ax.legend(title="City")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = f"figures/{view}_bow_corrected.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")
