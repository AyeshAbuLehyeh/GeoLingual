import argparse
import json
import random
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from geolingual.dataset_utils import load

random.seed(0)

from geolingual.config import VIGOR_ROOT

JUDGE_PROMPT = """You are auditing your own earlier description of this image for hallucinated (fabricated) details.

Here are specific claims extracted from that description's "distinctive anchors" list:
{claims_block}

For EACH claim, judge strictly from what is visible in the image:
- "GROUNDED": clearly visible and accurately described
- "PARTIAL": something similar is visible but the description overstates/misdescribes it (e.g. wrong color, wrong count, wrong object type)
- "HALLUCINATED": not visible in the image at all

Respond with ONLY a JSON array, one object per claim, in order, no other text:
[{{"claim": <number>, "verdict": "GROUNDED"|"PARTIAL"|"HALLUCINATED"}}, ...]"""


def split_claims(anchors_text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r",\s*(?=[a-zA-Z])", anchors_text) if p.strip()]
    return [p for p in parts if len(p) > 3][:8]


def load_image(image_path, image_type):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image.resize((w // 2, h // 2), Image.Resampling.LANCZOS)


class Judge:
    def __init__(self, model_id="Qwen/Qwen3-VL-30B-A3B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()

    @torch.no_grad()
    def judge(self, image, claims: list[str], max_new_tokens=300) -> str:
        claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = JUDGE_PROMPT.format(claims_block=claims_block)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def parse_verdicts(raw: str, n_claims: int) -> list[str | None]:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    verdicts = [None] * n_claims
    if not m:
        return verdicts
    try:
        arr = json.loads(m.group(0))
        for item in arr:
            idx = int(item.get("claim", -1)) - 1
            v = item.get("verdict")
            if 0 <= idx < n_claims and v in ("GROUNDED", "PARTIAL", "HALLUCINATED"):
                verdicts[idx] = v
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass
    return verdicts


def image_path_for(city, modality, filename):
    return f"{VIGOR_ROOT}/{city}/{modality}/{filename}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_city", type=int, default=15)
    ap.add_argument("--modality", choices=["ground", "satellite"], default="ground")
    ap.add_argument("--out", type=str, default="cache/hallucination_check.json")
    args = ap.parse_args()

    records = load()
    by_city = {}
    for r in records:
        by_city.setdefault(r["city"], []).append(r)
    sampled = []
    for city, recs in by_city.items():
        sampled.extend(random.sample(recs, min(args.n_per_city, len(recs))))
    print(f"Sampled {len(sampled)} images for hallucination check ({args.modality})")

    judge = Judge()
    results = []
    for i, r in enumerate(sampled):
        fields = r["pano_fields"] if args.modality == "ground" else r["sat_fields"]
        anchors = fields.get("distinctive_anchors", "")
        claims = split_claims(anchors)
        if not claims:
            continue
        filename = r["pano_file"] if args.modality == "ground" else r["sat_file"]
        modality_dir = "panorama" if args.modality == "ground" else "satellite"
        img_path = image_path_for(r["city"], modality_dir, filename)
        try:
            image = load_image(img_path, args.modality)
        except (FileNotFoundError, OSError) as e:
            print(f"  skip {img_path}: {e}")
            continue

        raw = judge.judge(image, claims)
        verdicts = parse_verdicts(raw, len(claims))
        results.append({
            "city": r["city"], "file": filename, "claims": claims,
            "verdicts": verdicts, "raw": raw,
        })
        if i % 10 == 0:
            print(f"  {i+1}/{len(sampled)} done", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)

    all_verdicts = [v for r in results for v in r["verdicts"] if v is not None]
    n = len(all_verdicts)
    if n:
        summary = {v: round(all_verdicts.count(v) / n * 100, 1) for v in ("GROUNDED", "PARTIAL", "HALLUCINATED")}
        summary["n_claims_judged"] = n
        summary["n_images"] = len(results)
        print("\n=== Grounding summary ===")
        print(json.dumps(summary, indent=2))
        json.dump(summary, open(args.out.replace(".json", "_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
