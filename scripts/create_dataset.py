import os

import pandas as pd

from geolingual.config import VIGOR_ROOT, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

for city in ["NewYork", "SanFrancisco", "Chicago", "Seattle"]:

    satellite = []
    panorama = []

    with open(f"{VIGOR_ROOT}/splits/{city}/same_area_balanced_test.txt", "r") as f:
        for line in f:
            line = line.strip().split(" ")
            panorama.append(line[0])
            satellite.append(line[1])

    pd.DataFrame({"satellite": satellite[:2500], "panorama": panorama[:2500]}).to_csv(DATA_DIR / f"{city}.csv", index=False)
