import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

GROUND_KEYS = ["road_topology_360", "spatial_layout", "distinctive_anchors", "orientation_cues"]
SAT_KEYS = ["cardinal_orientation", "road_geometry", "linear_sequence", "distinctive_anchors"]

RANKING_INSTRUCTION = """TASK: Exactly one of these {n} candidates is the TRUE matching satellite location for the query; the others are nearby look-alike distractors. Rank ALL {n} candidates from MOST to LEAST likely to be the true match, using road topology/orientation consistency and landmark correspondence.

Respond with ONLY a JSON object on one line, no other text:
{{"ranking": [<candidate numbers 1-{n}, most likely first, all {n} numbers exactly once>], "reason": "<one short sentence>"}}"""


def sat_text_block(fields):
    return " | ".join(f"{k}: {fields.get(k, '(not described)')}" for k in SAT_KEYS)


def ground_text_block(fields):
    return " | ".join(f"{k}: {fields.get(k, '(not described)')}" for k in GROUND_KEYS)


def load_image(path, max_side=448):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


class RerankJudge:
    def __init__(self, model_id="Qwen/Qwen3-VL-30B-A3B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def build_content(self, mode, query_text, query_image_path, candidates):
        n = len(candidates)
        content = []
        content.append({"type": "text", "text": "QUERY (ground-level 360-degree panorama):"})
        if mode in ("image", "both"):
            content.append({"type": "image", "image": load_image(query_image_path)})
        if mode in ("text", "both"):
            content.append({"type": "text", "text": query_text})

        content.append({"type": "text", "text": f"\nCANDIDATES ({n} satellite locations):"})
        for i, cand in enumerate(candidates, 1):
            content.append({"type": "text", "text": f"\n[{i}]"})
            if mode in ("image", "both"):
                content.append({"type": "image", "image": load_image(cand["image_path"])})
            if mode in ("text", "both"):
                content.append({"type": "text", "text": cand["text"]})

        content.append({"type": "text", "text": "\n" + RANKING_INSTRUCTION.format(n=n)})
        return content

    @torch.no_grad()
    def rerank(self, mode, query_text, query_image_path, candidates, max_new_tokens=250):
        content = self.build_content(mode, query_text, query_image_path, candidates)
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_LIST_RE = re.compile(r"\[[\d,\s]+\]")


def parse_ranking(raw: str, n: int) -> list[int]:
    ranking = None
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            cand = obj.get("ranking")
            if isinstance(cand, list):
                ranking = cand
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    if ranking is None:
        m2 = _LIST_RE.search(raw)
        if m2:
            try:
                ranking = json.loads(m2.group(0))
            except (json.JSONDecodeError, ValueError):
                ranking = None

    valid = list(range(1, n + 1))
    if not ranking:
        return valid

    seen = set()
    repaired = []
    for x in ranking:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid and xi not in seen:
            repaired.append(xi)
            seen.add(xi)
    for x in valid:
        if x not in seen:
            repaired.append(x)
    return repaired
