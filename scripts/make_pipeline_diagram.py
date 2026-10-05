import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.9)
ax.axis("off")


def box(x, y, w, h, text, fc="#EAF2FB", ec="#4C72B0", fontsize=11, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, side1="right", side2="left", color="black"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = {"right": (x1 + w1, y1 + h1 / 2), "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}[side1]
    p2 = {"left": (x2, y2 + h2 / 2), "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}[side2]
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    ax.add_patch(a)


b_pano = box(0.2, 5.6, 2.0, 1.0, "Ground panorama\n(query image)", fc="#FFF3E0", ec="#DD8452")
b_satdb = box(0.2, 1.0, 2.0, 1.0, "Satellite tile database\n(~9,368 candidates)", fc="#E8F5E9", ec="#55A868")

b_vision = box(2.7, 3.3, 2.4, 1.2, "Pretrained vision retriever\n(AuxGeo ConvNeXt-base,\nfrozen, contrastively trained)",
              fc="#F3E5F5", ec="#8E44AD")
arrow(b_pano, b_vision, side1="bottom", side2="top")
arrow(b_satdb, b_vision, side1="top", side2="bottom")

b_top10 = box(5.5, 3.4, 2.3, 1.0, "Top-10 candidates\n(94% R@1 already correct;\n591 queries wrong --\n\"hard subset\")",
             fc="#F3E5F5", ec="#8E44AD", fontsize=9.5)
arrow(b_vision, b_top10)

b_xml = box(0.2, 3.5, 2.0, 1.4, "VLM-generated structured\nXML descriptions\n(road topology, orientation,\ndistinctive anchors, ...)",
           fc="#E3F2FD", ec="#4C72B0", fontsize=9.5)

y0 = 0.3
b_text = box(8.3, y0 + 3.9, 2.3, 0.9, "TEXT-ONLY rerank\n(structured descriptions,\nlistwise)", fc="#E3F2FD", ec="#4C72B0", fontsize=9.5)
b_image = box(8.3, y0 + 2.5, 2.3, 0.9, "IMAGE-ONLY rerank\n(raw candidate images,\nlistwise)", fc="#FFF3E0", ec="#DD8452", fontsize=9.5)
b_both = box(8.3, y0 + 1.1, 2.3, 0.9, "IMAGE + TEXT rerank\n(both, concatenated)", fc="#E8F5E9", ec="#55A868", fontsize=9.5)

arrow(b_top10, b_text, side1="right", side2="left")
arrow(b_top10, b_image, side1="right", side2="left")
arrow(b_top10, b_both, side1="right", side2="left")
arrow(b_xml, b_text, side1="top", side2="left")
arrow(b_xml, b_both, side1="bottom", side2="left")

b_judge = box(11.0, y0 + 2.2, 1.7, 1.6, "Same VLM\n(Qwen3-VL-30B,\ntext-only or with\nimages, no fine-tuning)\nas LLM-as-judge",
             fc="#FCE4EC", ec="#C2185B", fontsize=9)
arrow(b_text, b_judge, side1="right", side2="left")
arrow(b_image, b_judge, side1="right", side2="left")
arrow(b_both, b_judge, side1="right", side2="left")

ax.text(9.45, 0.55, "Hard-subset recovery rate (n=591):\n"
                    "Text 10.5%  |  Image 25.5%  |  Both 10.5%\n"
                    "(random baseline = 10%)",
        ha="center", va="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#F9A825"))

plt.tight_layout()
plt.savefig("figures/pipeline_diagram.png", dpi=150)
plt.close()
print("Saved -> figures/pipeline_diagram.png")
