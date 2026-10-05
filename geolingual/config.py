import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VIGOR_ROOT = Path(os.environ.get("VIGOR_ROOT", "/path/to/VIGOR")).expanduser()

DRESS_ROOT = Path(os.environ.get("DRESS_ROOT", "/path/to/DReSS")).expanduser()
AUXGEO_CHECKPOINT = DRESS_ROOT / "checkpoints/vigor_same/convnext_base/1216225004/weights_e40_80.4258.pth"

DESCRIPTIONS_DIR = Path(os.environ.get("GEOLINGUAL_DESCRIPTIONS", REPO_ROOT / "descriptions")).expanduser()

CACHE_DIR = Path(os.environ.get("GEOLINGUAL_CACHE", REPO_ROOT / "cache")).expanduser()
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data"
PROMPT_DIR = REPO_ROOT / "prompts"

VLM_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
VLM_DIR_NAME = VLM_ID.split("/")[1]

CITIES = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
