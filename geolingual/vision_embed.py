import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from geolingual.config import DRESS_ROOT, AUXGEO_CHECKPOINT

sys.path.insert(0, str(DRESS_ROOT))

from auxgeo.model_modified import make_model
from auxgeo.transforms import get_transforms_val

CHECKPOINT = AUXGEO_CHECKPOINT
IMG_SIZE = 384


class Opt:
    model = "convnext_base"


def load_model(device="cuda"):
    model = make_model(Opt())
    state = torch.load(CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load_state_dict] missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"  [load_state_dict] unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")
    model.eval().to(device)
    return model


def get_transforms():
    image_size_sat = (IMG_SIZE, IMG_SIZE)
    new_width = IMG_SIZE * 2
    new_height = int((1024 / 2048) * new_width)
    img_size_ground = (new_height, new_width)
    sat_t, ground_t = get_transforms_val(image_size_sat, img_size_ground)
    return sat_t, ground_t


def load_and_transform(path, transform):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(image=img)["image"]


@torch.no_grad()
def embed_images(paths, transform, model, device="cuda", batch_size=64, desc="embed"):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch_paths = paths[i:i + batch_size]
        tensors = torch.stack([load_and_transform(p, transform) for p in batch_paths]).to(device)
        gap_feature, _ = model(tensors)
        gap_feature = torch.nn.functional.normalize(gap_feature, dim=-1)
        embs.append(gap_feature.cpu().numpy())
    return np.concatenate(embs, axis=0)

