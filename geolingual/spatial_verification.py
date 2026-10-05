import argparse
import json
import random
import re

from geolingual.dataset_utils import load
from geolingual.geo_negatives import build_pools

random.seed(0)

PROMPT_TEMPLATE = """You are verifying whether a ground-level street description and a satellite-view description depict the SAME physical location.

GROUND-LEVEL DESCRIPTION (egocentric, from a 360-degree panorama):
- Road topology: {road_topology_360}
- Spatial layout (left/right sequences): {spatial_layout}
- Distinctive anchors: {g_anchors}
- Orientation cues: {orientation_cues}

SATELLITE DESCRIPTION (allocentric, aerial view) -- CANDIDATE:
- Cardinal orientation: {cardinal_orientation}
- Road geometry: {road_geometry}
- Linear sequence along the road: {linear_sequence}
- Distinctive anchors: {s_anchors}

TASK: Judge spatial consistency. Consider whether road topology/orientation could align, whether the described anchors and sequences could plausibly correspond, and note any explicit contradictions (e.g. straight road vs. cul-de-sac, or incompatible orientations).

Respond with ONLY a JSON object on one line, no other text:
{{"score": <integer 0-10, 10 = very likely same location, 0 = clearly different>, "verdict": "MATCH" or "NO_MATCH", "reason": "<one short sentence>"}}"""

FIELD_ABLATIONS = {
    "all": ["road_topology_360", "spatial_layout", "g_anchors", "orientation_cues",
            "cardinal_orientation", "road_geometry", "linear_sequence", "s_anchors"],
    "topology_only": ["road_topology_360", "cardinal_orientation", "road_geometry"],
    "landmarks_only": ["g_anchors", "s_anchors", "linear_sequence"],
    "orientation_only": ["orientation_cues", "cardinal_orientation"],
}


def build_prompt(ground_fields: dict, sat_fields: dict, field_subset: list[str] = None) -> str:
    slot_values = {
        "road_topology_360": ground_fields.get("road_topology_360", "(not described)"),
        "spatial_layout": ground_fields.get("spatial_layout", "(not described)"),
        "g_anchors": ground_fields.get("distinctive_anchors", "(not described)"),
        "orientation_cues": ground_fields.get("orientation_cues", "(not described)"),
        "cardinal_orientation": sat_fields.get("cardinal_orientation", "(not described)"),
        "road_geometry": sat_fields.get("road_geometry", "(not described)"),
        "linear_sequence": sat_fields.get("linear_sequence", "(not described)"),
        "s_anchors": sat_fields.get("distinctive_anchors", "(not described)"),
    }
    if field_subset is not None:
        omit = set(slot_values) - set(field_subset)
        for k in omit:
            slot_values[k] = "(withheld for this ablation)"
    return PROMPT_TEMPLATE.format(**slot_values)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+)')
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(MATCH|NO_MATCH)"')


def parse_verdict(raw: str) -> dict:
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = int(obj.get("score", -1))
            verdict = obj.get("verdict", "")
            if 0 <= score <= 10 and verdict in ("MATCH", "NO_MATCH"):
                return {"score": score, "verdict": verdict, "reason": obj.get("reason", ""), "parsed": True}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    sm = _SCORE_RE.search(raw)
    vm = _VERDICT_RE.search(raw)
    if sm or vm:
        return {
            "score": int(sm.group(1)) if sm else (10 if (vm and vm.group(1) == "MATCH") else 0),
            "verdict": vm.group(1) if vm else None,
            "reason": raw[:120],
            "parsed": False,
        }
    return {"score": None, "verdict": None, "reason": raw[:120], "parsed": False}


def rank_metrics(pools_with_scores: list[dict]) -> dict:
    n = len(pools_with_scores)
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    for p in pools_with_scores:
        ranked = sorted(p["scored"], key=lambda x: x[1], reverse=True)
        ranked_ids = [sid for sid, _ in ranked]
        rank = ranked_ids.index(p["true_sat_id"]) + 1
        if rank == 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 5:
            hit5 += 1
        mrr += 1.0 / rank
    return {
        "n": n,
        "R@1_in_pool": hit1 / n * 100,
        "R@3_in_pool": hit3 / n * 100,
        "R@5_in_pool": hit5 / n * 100,
        "MRR": mrr / n,
    }


def sample_queries(n_per_city: int, k_hard: int, seed: int = 0):
    records = load()
    random.seed(seed)
    by_city = {}
    for r in records:
        by_city.setdefault(r["city"], []).append(r)
    sampled = []
    for city, recs in by_city.items():
        sampled.extend(random.sample(recs, min(n_per_city, len(recs))))
    pools, sat_by_city = build_pools(k_hard=k_hard, k_random=0, seed_records=records)
    pool_by_qid = {p["query_id"]: p for p in pools}
    query_lookup = {f"{r['city']}_{r['pano_file']}": r for r in sampled}
    sampled_pools = [pool_by_qid[qid] for qid in query_lookup if qid in pool_by_qid]
    return sampled_pools, query_lookup, sat_by_city


def run_verification(pools, query_lookup, sat_by_city, field_subset_name="all",
                      batch_size=8, max_new_tokens=120, model=None, log_every=50):
    from text_llm import TextModel
    if model is None:
        model = TextModel()

    field_subset = FIELD_ABLATIONS[field_subset_name]

    flat = []
    for p in pools:
        r = query_lookup[p["query_id"]]
        for cand in p["candidates"]:
            sat_fields = sat_by_city[p["city"]][cand]["fields"]
            prompt = build_prompt(r["pano_fields"], sat_fields, field_subset)
            flat.append((p["query_id"], p["true_sat_id"], cand, prompt))

    print(f"Total LLM calls: {len(flat)} ({len(pools)} pools x pool size {len(pools[0]['candidates'])})")

    raw_outputs = []
    for i in range(0, len(flat), batch_size):
        batch = flat[i:i + batch_size]
        prompts = [b[3] for b in batch]
        outs = model.generate_batch(prompts, max_new_tokens=max_new_tokens)
        raw_outputs.extend(outs)
        if (i // batch_size) % log_every == 0:
            print(f"  {i + len(batch)}/{len(flat)} calls done", flush=True)

    per_query = {}
    detail = []
    for (qid, true_sat, cand, prompt), raw in zip(flat, raw_outputs):
        v = parse_verdict(raw)
        score = v["score"] if v["score"] is not None else 0
        per_query.setdefault(qid, {"query_id": qid, "true_sat_id": true_sat, "scored": []})
        per_query[qid]["scored"].append((cand, score))
        detail.append({"query_id": qid, "candidate": cand, "is_true": cand == true_sat,
                        "raw": raw, **v})

    pools_with_scores = list(per_query.values())
    metrics = rank_metrics(pools_with_scores)
    return metrics, detail, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_city", type=int, default=3)
    ap.add_argument("--k_hard", type=int, default=9)
    ap.add_argument("--mode", choices=["smoke", "run", "ablation"], default="smoke")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out_prefix", type=str, default="cache/spatial_verification")
    args = ap.parse_args()

    pools, query_lookup, sat_by_city = sample_queries(args.n_per_city, args.k_hard)
    print(f"Sampled {len(pools)} query pools, pool size = 1+{args.k_hard}")

    if args.mode == "smoke":
        p = pools[0]
        r = query_lookup[p["query_id"]]
        cand_sat = p["candidates"][0]
        prompt = build_prompt(r["pano_fields"], sat_by_city[p["city"]][cand_sat]["fields"])
        print("\n--- Example prompt ---\n")
        print(prompt)
        print("\n--- parse_verdict smoke test ---")
        print(parse_verdict('blah blah {"score": 8, "verdict": "MATCH", "reason": "roads align"} trailing'))
        print(parse_verdict('not json at all, score: 3, NO_MATCH'))

    elif args.mode == "run":
        metrics, detail, model = run_verification(pools, query_lookup, sat_by_city,
                                                    field_subset_name="all", batch_size=args.batch_size, log_every=args.log_every)
        print("\n=== Structured verification ranking metrics ===")
        print(json.dumps(metrics, indent=2))
        json.dump(metrics, open(f"{args.out_prefix}_metrics.json", "w"), indent=2)
        json.dump(detail, open(f"{args.out_prefix}_detail.json", "w"))
        print(f"Saved -> {args.out_prefix}_metrics.json / _detail.json")

    elif args.mode == "ablation":
        model = None
        all_results = {}
        for name in FIELD_ABLATIONS:
            print(f"\n=== Ablation: {name} ===")
            metrics, detail, model = run_verification(
                pools, query_lookup, sat_by_city, field_subset_name=name,
                batch_size=args.batch_size, model=model,
            )
            print(json.dumps(metrics, indent=2))
            all_results[name] = metrics
            json.dump(detail, open(f"{args.out_prefix}_ablation_{name}_detail.json", "w"))
        json.dump(all_results, open(f"{args.out_prefix}_ablation_metrics.json", "w"), indent=2)
        print(f"Saved -> {args.out_prefix}_ablation_metrics.json")
