import json
import re
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from geolingual.dataset_utils import load

LEMM = WordNetLemmatizer()

PROTECTED_BIGRAMS = [
    "parking lot", "fire hydrant", "stop sign", "traffic light", "bus stop",
    "street light", "power line", "chain link fence", "brick building",
]

LANDMARKS = {
    "building": {"building", "house", "structure"},
    "tree": {"tree"},
    "intersection": {"intersection", "crossroad", "junction"},
    "lane": {"lane"},
    "crosswalk": {"crosswalk", "cross walk"},
}

FIELD_KEYS = {
    "ground": ["road_topology_360", "sector_forward", "sector_backward",
               "spatial_layout", "road_markings", "orientation_cues",
               "distinctive_anchors", "environmental_context"],
    "satellite": ["distinctive_anchors", "roof_fingerprints", "linear_sequence",
                  "cardinal_orientation", "road_geometry", "shadow_analysis"],
}

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}


def _protect_bigrams(text: str) -> str:
    for bg in PROTECTED_BIGRAMS:
        text = re.sub(re.escape(bg), bg.replace(" ", "_"), text, flags=re.IGNORECASE)
    return text


def lemmatized_nouns(text: str) -> list[str]:
    text = _protect_bigrams(text.lower())
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = []
    for word, tag in tagged:
        if tag in _NOUN_TAGS and word.isalpha() or "_" in word:
            lemma = LEMM.lemmatize(word.replace("_", " "), pos=wordnet.NOUN)
            nouns.append(lemma)
    return nouns


def full_text(fields: dict, modality: str) -> str:
    return " ".join(fields.get(k, "") for k in FIELD_KEYS[modality])


def contains_landmark(nouns: list[str], surface_forms: set[str]) -> bool:
    noun_set = set(nouns)
    for form in surface_forms:
        if " " in form:
            if form in " ".join(nouns):
                return True
        elif form in noun_set:
            return True
    return False


def main():
    records = load()
    by_city = defaultdict(list)
    for r in records:
        by_city[r["city"]].append(r)

    results = {}
    for city, recs in by_city.items():
        n = len(recs)
        counts = {"ground": defaultdict(int), "satellite": defaultdict(int)}
        for r in recs:
            g_nouns = lemmatized_nouns(full_text(r["pano_fields"], "ground"))
            s_nouns = lemmatized_nouns(full_text(r["sat_fields"], "satellite"))
            for lm, forms in LANDMARKS.items():
                if contains_landmark(g_nouns, forms):
                    counts["ground"][lm] += 1
                if contains_landmark(s_nouns, forms):
                    counts["satellite"][lm] += 1
        results[city] = {
            "valid_cases": n,
            "ground": {lm: [round(counts["ground"][lm] / n, 2), counts["ground"][lm]] for lm in LANDMARKS},
            "satellite": {lm: [round(counts["satellite"][lm] / n, 2), counts["satellite"][lm]] for lm in LANDMARKS},
        }
        print(f"\n{city} (n={n})")
        for lm in LANDMARKS:
            g = results[city]["ground"][lm]
            s = results[city]["satellite"][lm]
            print(f"  {lm:12s}  ground=({g[0]:.2f},{g[1]:4d})  satellite=({s[0]:.2f},{s[1]:4d})")

    with open("cache/lexical_distribution.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> cache/lexical_distribution.json")


if __name__ == "__main__":
    main()
