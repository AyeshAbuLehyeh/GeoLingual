import argparse
import os

import pandas as pd

from geolingual.config import CITIES, DATA_DIR, VIGOR_ROOT, VLM_ID
from geolingual.vlm import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", choices=CITIES)
    ap.add_argument("mode", choices=["satellite", "panorama"])
    ap.add_argument("--model_id", default=VLM_ID)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(DATA_DIR / f"{args.city}.csv")
    files = df[args.mode].tolist()
    if args.limit:
        files = files[:args.limit]

    folder = os.path.join(VIGOR_ROOT, args.city, args.mode)
    agent = Model(model_id=args.model_id)
    for name in files:
        agent.generate(image_path=os.path.join(folder, name), image_type=args.mode)


if __name__ == "__main__":
    main()
