import re
import xml.etree.ElementTree as ET
from pathlib import Path

GROUND_SCHEMA = [
    "road_topology_360", "sector_forward", "sector_backward", "spatial_layout",
    "road_markings", "orientation_cues", "distinctive_anchors", "environmental_context",
]
SATELLITE_SCHEMA = [
    "distinctive_anchors", "roof_fingerprints", "linear_sequence",
    "cardinal_orientation", "road_geometry", "shadow_analysis",
]
ROOT_TAG = {"ground": "ground_scene", "satellite": "satellite_scene"}
SCHEMA = {"ground": GROUND_SCHEMA, "satellite": SATELLITE_SCHEMA}


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:xml)?\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text.strip())
    return text.strip()


def _extract_root_block(text: str, root_tag: str) -> str | None:
    start = text.find(f"<{root_tag}>")
    if start == -1:
        return None
    end = text.rfind(f"</{root_tag}>")
    if end == -1:
        return text[start:].rstrip() + f"\n</{root_tag}>"
    return text[start:end + len(f"</{root_tag}>")]


def _repair_mismatched_tags(block: str, schema: list[str]) -> str:
    tag_pattern = re.compile(
        r"<(/?)(" + "|".join(re.escape(t) for t in schema) + r")>"
    )
    tokens = list(tag_pattern.finditer(block))
    fields = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_close, name = tok.group(1), tok.group(2)
        if is_close:
            i += 1
            continue
        content_start = tok.end()
        if i + 1 < len(tokens):
            content_end = tokens[i + 1].start()
        else:
            content_end = len(block)
        content = block[content_start:content_end].strip()
        if name not in fields and content:
            fields[name] = content
        i += 1
    return fields


def parse_description(raw_text: str, modality: str) -> dict | None:
    schema = SCHEMA[modality]
    root_tag = ROOT_TAG[modality]

    text = _strip_code_fences(raw_text)
    block = _extract_root_block(text, root_tag)
    if block is None:
        block = text

    try:
        root = ET.fromstring(block)
        fields = {}
        for child in root:
            tag = child.tag
            if tag in schema and child.text and child.text.strip():
                fields[tag] = child.text.strip()
        if fields:
            return fields
    except ET.ParseError:
        pass

    fields = _repair_mismatched_tags(block, schema)
    return fields if fields else None


def load_and_parse(path: str, modality: str) -> dict | None:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not raw.strip():
        return None
    return parse_description(raw, modality)


def is_valid(fields: dict | None, modality: str, min_fields: int = 4) -> bool:
    if not fields:
        return False
    return len(fields) >= min_fields


if __name__ == "__main__":
    sample = Path("panorama/Qwen3-VL-30B-A3B-Instruct/Chicago").glob("panorama/*.txt")
    tested = 0
    ok = 0
    for p in list(sample)[:50]:
        fields = load_and_parse(str(p), "ground")
        tested += 1
        if is_valid(fields, "ground"):
            ok += 1
    print(f"self-test: {ok}/{tested} ground samples parsed valid")
