# -*- coding: utf-8 -*-
"""unetpp.ipynb
## Setup
"""

import os
import gc
import cv2
import math
import time
import json
import copy
import random
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# -------------------------
# Robust reproducibility
# -------------------------
def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

SEED = 42
seed_everything(SEED, deterministic=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("DEVICE:", DEVICE)
# print("torch:", torch.__version__)

"""## Get Data"""

# !pip install -q huggingface_hub

# from huggingface_hub import login
# login()

# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Heatwave114/organoids",
#     repo_type="dataset",
#     local_dir="/content/data",
#     local_dir_use_symlinks=False,
#     max_workers=8   # parallel download
# )

"""## Settings"""

COHORTS = [
    {
        "name": "annotate_b0",
        "images_dir": "./data/annotate_b0/images",
        "masks_dir": "./data/annotate_b0/organoid",
    },
    {
        "name": "annotate_b1",
        "images_dir": "./data/annotate_b1/images",
        "masks_dir": "./data/annotate_b1/organoid",
    },
    {
        "name": "annotate_b2",
        "images_dir": "./data/annotate_b2/images",
        "masks_dir": "./data/annotate_b2/organoid",
    },
    {
        "name": "data_paper",
        "images_dir": "./data/data_paper/imgs",
        "masks_dir": "./data/data_paper/labels",
    },
]

# -------- Evaluation settings --------
EVALUATION_MODE = False
MODEL_TYPE = "best" # best | final

# -------- split settings --------
USE_KFOLD = True
N_FOLDS = 5
VAL_SIZE_IF_NOT_KFOLD = 0.2

# -------- training settings --------
IMG_SIZE_UNET = 256
BATCH_SIZE_UNET = 32
NUM_WORKERS = 0

EPOCHS_UNET = 12
LR_UNET = 1e-3
WEIGHT_DECAY_UNET = 1e-4

# -------- augmentation / misc --------
USE_BASIC_AUG = False
PIN_MEMORY = torch.cuda.is_available()

# -------- output --------
RUN_ROOT = Path("./unetpp")
RESULTS_ROOT = Path(f"./unetpp/results/unetpp_{MODEL_TYPE}")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# -------- matching --------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# print("USE_KFOLD:", USE_KFOLD)
# print("N_FOLDS:", N_FOLDS)
# print("IMG_SIZE_UNET:", IMG_SIZE_UNET)

"""## Data Discovery"""

# def norm_stem(p: Path) -> str:
#     s = p.stem.strip()
#     s_low = s.lower()
#     if "b1" not in str(p):
#         suffix = "_organoid_mask"
#     else:
#         suffix = " _organoid_mask"
#     if s_low.endswith(suffix):
#         s = s[: -len(suffix)]
#     return s

# def list_files_with_exts(folder: Path, exts: set) -> List[Path]:
#     return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

# def read_image_size(path: Path) -> Tuple[int, int]:
#     with Image.open(path) as im:
#         w, h = im.size
#     return h, w

# def read_mask_np(path: Path) -> np.ndarray:
#     with Image.open(path) as im:
#         arr = np.array(im)
#     if arr.ndim == 3:
#         arr = arr[..., 0]
#     arr = (arr > 0).astype(np.uint8)
#     return arr

# def build_manifest(cohorts: List[Dict]) -> pd.DataFrame:
#     rows = []

#     for cohort in cohorts:
#         cohort_name = cohort["name"]
#         images_dir = Path(cohort["images_dir"])
#         masks_dir  = Path(cohort["masks_dir"])

#         if not images_dir.is_dir():
#             print(f"[WARN] images dir missing: {images_dir}")
#             continue
#         if not masks_dir.is_dir():
#             print(f"[WARN] masks dir missing: {masks_dir}")
#             continue

#         image_files = list_files_with_exts(images_dir, IMAGE_EXTS)
#         mask_files  = list_files_with_exts(masks_dir, MASK_EXTS)

#         image_map = {norm_stem(p): p for p in image_files}
#         mask_map  = {norm_stem(p): p for p in mask_files}

#         common = sorted(set(image_map.keys()) & set(mask_map.keys()))
#         only_images = sorted(set(image_map.keys()) - set(mask_map.keys()))
#         only_masks  = sorted(set(mask_map.keys()) - set(image_map.keys()))

#         print(f"\n[{cohort_name}]")
#         print(" matched:", len(common))
#         print(" images only:", len(only_images))
#         print(" masks only:", len(only_masks))

#         for stem in tqdm(common, desc=f"scan {cohort_name}"):
#             img_path = image_map[stem]
#             mask_path = mask_map[stem]

#             try:
#                 img_h, img_w = read_image_size(img_path)
#                 mask = read_mask_np(mask_path)
#                 mask_h, mask_w = mask.shape[:2]
#                 unique_vals = sorted(np.unique(mask).tolist())
#                 fg_pixels = int(mask.sum())
#                 ok_shape = (img_h == mask_h) and (img_w == mask_w)

#                 rows.append({
#                     "cohort": cohort_name,
#                     "stem": stem,
#                     "image_path": str(img_path),
#                     "mask_path": str(mask_path),
#                     "img_h": img_h,
#                     "img_w": img_w,
#                     "mask_h": mask_h,
#                     "mask_w": mask_w,
#                     "same_hw": ok_shape,
#                     "mask_unique": unique_vals,
#                     "fg_pixels": fg_pixels,
#                 })
#             except Exception as e:
#                 rows.append({
#                     "cohort": cohort_name,
#                     "stem": stem,
#                     "image_path": str(img_path),
#                     "mask_path": str(mask_path),
#                     "img_h": None,
#                     "img_w": None,
#                     "mask_h": None,
#                     "mask_w": None,
#                     "same_hw": False,
#                     "mask_unique": [f"ERROR: {e}"],
#                     "fg_pixels": None,
#                 })

#     df = pd.DataFrame(rows)
#     return df

# manifest = build_manifest(COHORTS)
# print("\nTotal matched pairs:", len(manifest))
# display(manifest.head())

# bad_shape = manifest[manifest["same_hw"] == False]
# print("Pairs with mismatched image/mask size:", len(bad_shape))

# bad_mask_values = manifest[
#     ~manifest["mask_unique"].apply(
#         lambda x: set(x).issubset({0, 1})
#         if isinstance(x, list) and all(isinstance(v, (int, np.integer)) for v in x)
#         else False
#     )
# ]
# print("Pairs with non-binary mask values:", len(bad_mask_values))

"""## Filtering"""

# df = manifest.copy()

# df = df[df["same_hw"] == True].reset_index(drop=True)
# df = df[df["mask_unique"].apply(lambda x: set(x).issubset({0, 1}))].reset_index(drop=True)

# print("Valid pairs kept:", len(df))
# display(df.groupby("cohort").size().reset_index(name="count"))

# optional: drop empty masks
# DROP_EMPTY_MASKS = False
# if DROP_EMPTY_MASKS:
#     before = len(df)
#     df = df[df["fg_pixels"] > 0].reset_index(drop=True)
#     print("Dropped empty masks:", before - len(df))
#     print("Remaining:", len(df))

"""## Visualization"""

# def load_rgb_image(path: str) -> np.ndarray:
#     with Image.open(path) as im:
#         im = im.convert("RGB")
#         arr = np.array(im)
#     return arr

# def load_bin_mask(path: str) -> np.ndarray:
#     with Image.open(path) as im:
#         arr = np.array(im)
#     if arr.ndim == 3:
#         arr = arr[..., 0]
#     arr = (arr > 0).astype(np.uint8)
#     return arr

# def show_samples_from_each_cohort(df: pd.DataFrame, n_per_cohort: int = 3, seed: int = 42):
#     rng = np.random.default_rng(seed)
#     cohorts = sorted(df["cohort"].unique().tolist())

#     for cohort in cohorts:
#         sub = df[df["cohort"] == cohort]
#         n = min(n_per_cohort, len(sub))
#         if n == 0:
#             continue

#         idxs = rng.choice(sub.index.to_numpy(), size=n, replace=False)
#         samples = sub.loc[idxs]

#         fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
#         if n == 1:
#             axes = np.expand_dims(axes, 0)

#         fig.suptitle(f"Cohort: {cohort}", fontsize=14)

#         for r, (_, row) in enumerate(samples.iterrows()):
#             img = load_rgb_image(row["image_path"])
#             msk = load_bin_mask(row["mask_path"])
#             msk_vis = (msk * 255).astype(np.uint8)

#             overlay = img.copy()
#             overlay[msk == 1] = (0.6 * overlay[msk == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

#             axes[r, 0].imshow(img)
#             axes[r, 0].set_title(f"Image\n{Path(row['image_path']).name}")
#             axes[r, 0].axis("off")

#             axes[r, 1].imshow(msk_vis, cmap="gray", vmin=0, vmax=255)
#             axes[r, 1].set_title(f"Mask visible\nunique={np.unique(msk).tolist()}")
#             axes[r, 1].axis("off")

#             axes[r, 2].imshow(overlay)
#             axes[r, 2].set_title("Overlay")
#             axes[r, 2].axis("off")

#         plt.tight_layout()
#         plt.show()

# show_samples_from_each_cohort(df, n_per_cohort=1, seed=SEED+4)

"""## Dataset and Transform"""

def resize_image_and_mask(img: np.ndarray, mask: np.ndarray, size: int):
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask_r = (mask_r > 0).astype(np.uint8)
    return img_r, mask_r

def maybe_augment(img: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        mask = np.ascontiguousarray(np.fliplr(mask))
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.flipud(img))
        mask = np.ascontiguousarray(np.flipud(mask))
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        img = np.ascontiguousarray(np.rot90(img, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
    return img, mask

class OrganoidSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 512, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = load_rgb_image(row["image_path"])
        mask = load_bin_mask(row["mask_path"])

        if self.train and USE_BASIC_AUG:
            img, mask = maybe_augment(img, mask)

        img, mask = resize_image_and_mask(img, mask, self.size)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return {
            "image": img,
            "mask": mask,
            "stem": row["stem"],
            "cohort": row["cohort"],
        }

"""## UNet++ Model"""

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        nb_filter = [base, base * 2, base * 4, base * 8, base * 16]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv0_0 = DoubleConv(in_ch, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            outputs = [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ]
            return outputs[-1]

        return self.final(x0_4)

"""## Losses and Dice"""

def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    dims = (1, 2, 3)
    inter = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * soft_dice_loss(logits, targets)

"""## Train/Eval Helpers"""

def make_loader(sub_df, train: bool, batch_size: int):
    ds = OrganoidSegDataset(sub_df, size=IMG_SIZE_UNET, train=train)
    g = torch.Generator()
    g.manual_seed(SEED)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=g,
    )

def train_one_epoch_unet(model, loader, optimizer, criterion, device, fold=None, epoch=None):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n = 0

    desc = f"fold {fold} train ep {epoch}" if fold is not None and epoch is not None else "train"

    for batch in tqdm(loader, desc=desc, leave=True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_dice += dice_score_from_logits(logits.detach(), y).item() * bs
        n += bs

    return running_loss / max(n, 1), running_dice / max(n, 1)


@torch.no_grad()
def eval_one_epoch_unet(model, loader, criterion, device, fold=None, epoch=None):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n = 0

    desc = f"fold {fold} val ep {epoch}" if fold is not None and epoch is not None else "val"

    for batch in tqdm(loader, desc=desc, leave=True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_dice += dice_score_from_logits(logits, y).item() * bs
        n += bs

    return running_loss / max(n, 1), running_dice / max(n, 1)

"""## Splits Logic"""

# def make_splits(df: pd.DataFrame):
#     idx = np.arange(len(df))

#     if USE_KFOLD:
#         kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
#         splits = []
#         for fold, (tr_idx, va_idx) in enumerate(kf.split(idx), start=1):
#             splits.append((fold, tr_idx, va_idx))
#         return splits
#     else:
#         tr_idx, va_idx = train_test_split(
#             idx,
#             test_size=VAL_SIZE_IF_NOT_KFOLD,
#             random_state=SEED,
#             shuffle=True,
#         )
#         return [(1, tr_idx, va_idx)]

# splits = make_splits(df)
# print("Num splits:", len(splits))
# for fold, tr_idx, va_idx in splits:
#     print(f"Fold {fold}: train={len(tr_idx)}, val={len(va_idx)}")

# """## Train"""

# if not EVALUATION_MODE:
#   criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
#   all_fold_results = []

#   for fold, tr_idx, va_idx in splits:
#       print(f"\n{'='*80}\nUNet++ Fold {fold}\n{'='*80}")

#       seed_everything(SEED + fold, deterministic=True)

#       tr_df = df.iloc[tr_idx].reset_index(drop=True)
#       va_df = df.iloc[va_idx].reset_index(drop=True)

#       train_loader = make_loader(tr_df, train=True, batch_size=BATCH_SIZE_UNET)
#       val_loader = make_loader(va_df, train=False, batch_size=BATCH_SIZE_UNET)

#       model = UNetPlusPlus(in_ch=3, out_ch=1, base=32).to(DEVICE)
#       optimizer = torch.optim.AdamW(model.parameters(), lr=LR_UNET, weight_decay=WEIGHT_DECAY_UNET)

#       best_val_dice = -1.0
#       best_state = None
#       history = []

#       fold_dir = RUN_ROOT / f"fold_{fold:02d}"
#       fold_dir.mkdir(parents=True, exist_ok=True)

#       for epoch in range(1, EPOCHS_UNET + 1):
#           t0 = time.time()

#           tr_loss, tr_dice = train_one_epoch_unet(model, train_loader, optimizer, criterion, DEVICE)
#           va_loss, va_dice = eval_one_epoch_unet(model, val_loader, criterion, DEVICE)

#           dt = time.time() - t0
#           history.append({
#               "fold": fold,
#               "epoch": epoch,
#               "train_loss": tr_loss,
#               "train_dice": tr_dice,
#               "val_loss": va_loss,
#               "val_dice": va_dice,
#               "time_sec": dt,
#           })

#           print(
#               f"Epoch {epoch:03d} | "
#               f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
#               f"val_loss={va_loss:.4f} val_dice={va_dice:.4f} | "
#               f"{dt:.1f}s"
#           )

#           if va_dice > best_val_dice:
#               best_val_dice = va_dice
#               best_state = copy.deepcopy(model.state_dict())
#               torch.save(best_state, fold_dir / "best_unetpp.pt")

#       # save final model (last epoch weights)
#       torch.save(model.state_dict(), fold_dir / "final_unetpp.pt")

#       hist_df = pd.DataFrame(history)
#       hist_df.to_csv(fold_dir / "history.csv", index=False)

#       all_fold_results.append({
#           "fold": fold,
#           "best_val_dice": best_val_dice,
#           "n_train": len(tr_df),
#           "n_val": len(va_df),
#       })

#       del model, optimizer, train_loader, val_loader, best_state
#       gc.collect()
#       if torch.cuda.is_available():
#           torch.cuda.empty_cache()

#   results_unet = pd.DataFrame(all_fold_results)
#   display(results_unet)

"""## Summary"""

# mean_dice = results_unet["best_val_dice"].mean()
# std_dice = results_unet["best_val_dice"].std(ddof=1) if len(results_unet) > 1 else 0.0

# print("UNet++ results")
# print(results_unet)
# print(f"\nMean best val Dice: {mean_dice:.4f}")
# print(f"Std best val Dice : {std_dice:.4f}")

# plt.figure(figsize=(6, 4))
# plt.plot(results_unet["fold"], results_unet["best_val_dice"], marker="o")
# plt.xlabel("Fold")
# plt.ylabel("Best validation Dice")
# plt.title("UNet++ fold-wise Dice")
# plt.grid(True, alpha=0.3)
# plt.show()

"""## Evaluation"""

import os
import re
import gc
import math
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# ============================================================
# SETTINGS
# ============================================================
EVAL_DIR = RESULTS_ROOT
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_BATCH_SIZE = 8

B0_DAY0_DATE = dt.date(2025, 7, 9)
B1_B2_DAY0_DATE = dt.date(2025, 10, 18)

COHORT_COLOR = {
    "annotate_b0": "tab:blue",
    "annotate_b1": "tab:orange",
    "annotate_b2": "tab:green",
    "data_paper": "tab:red",
}

COHORT_SHORT = {
    "annotate_b0": "b0",
    "annotate_b1": "b1",
    "annotate_b2": "b2",
    "data_paper": "labs_data",
}

LAB_COLOR = {
    "Lab A": "tab:purple",
    "Lab B": "tab:brown",
}

# unique fold colors, no reuse for the usual 5-fold case
FOLD_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

# ============================================================
# HELPERS
# ============================================================
def snap_to_annotate_bin(day: int):
    if day is None:
        return None
    day = int(day)
    if day <= 14:
        return max(0, day)
    later_bins = [30, 60, 90, 120, 150]
    return min(later_bins, key=lambda x: (abs(x - day), x))

def parse_date_from_filename(fname: str):
    stem = Path(fname).stem
    m = re.search(r'\b(20\d{2})[-_](\d{2})[-_](\d{2})\b', stem)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return dt.date(y, mo, d)
    except Exception:
        return None

def parse_explicit_day_string(fname: str):
    stem = Path(fname).stem
    patterns = [
        r'\bday[\s_\-]*0*(\d+)\b',
        r'\bd[\s_\-]*0*(\d+)\b',
    ]
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def parse_data_paper_day_lab(fname: str):
    stem = Path(fname).stem

    m_day = re.search(r'_d0*(\d+)(?:_|$)', stem, flags=re.IGNORECASE)
    day = int(m_day.group(1)) if m_day else None

    m_lab = re.search(r'_(LabA|LabB)(?:_|$)', stem, flags=re.IGNORECASE)
    lab = None
    if m_lab:
        lab_raw = m_lab.group(1).lower()
        if lab_raw == "laba":
            lab = "Lab A"
        elif lab_raw == "labb":
            lab = "Lab B"

    return {
        "raw_day": day,
        "day_bin": day,
        "lab": lab,
        "day_logic": "data_paper_filename",
    }

def parse_annotate_b0_day(fname: str):
    explicit_day = parse_explicit_day_string(fname)
    if explicit_day is not None:
        return {
            "raw_day": int(explicit_day),
            "day_bin": snap_to_annotate_bin(explicit_day),
            "lab": None,
            "day_logic": "b0_explicit_day",
        }

    file_date = parse_date_from_filename(fname)
    if file_date is not None:
        raw_day = (file_date - B0_DAY0_DATE).days
        if raw_day < 0:
            raw_day = 0
        return {
            "raw_day": int(raw_day),
            "day_bin": snap_to_annotate_bin(raw_day),
            "lab": None,
            "day_logic": "b0_date_fallback",
        }

    return {
        "raw_day": None,
        "day_bin": None,
        "lab": None,
        "day_logic": "b0_missing",
    }

def parse_annotate_b1_b2_day(fname: str, cohort_name: str):
    explicit_day = parse_explicit_day_string(fname)
    if explicit_day is not None:
        return {
            "raw_day": int(explicit_day),
            "day_bin": snap_to_annotate_bin(explicit_day),
            "lab": None,
            "day_logic": f"{COHORT_SHORT[cohort_name]}_explicit_day",
        }

    file_date = parse_date_from_filename(fname)
    if file_date is not None:
        raw_day = (file_date - B1_B2_DAY0_DATE).days
        if raw_day < 0:
            raw_day = 0
        return {
            "raw_day": int(raw_day),
            "day_bin": snap_to_annotate_bin(raw_day),
            "lab": None,
            "day_logic": f"{COHORT_SHORT[cohort_name]}_date_fallback",
        }

    return {
        "raw_day": None,
        "day_bin": None,
        "lab": None,
        "day_logic": f"{COHORT_SHORT[cohort_name]}_missing",
    }

def parse_metadata_for_row(row):
    fname = os.path.basename(str(row["image_path"]))
    cohort = str(row["cohort"])

    if cohort == "data_paper":
        out = parse_data_paper_day_lab(fname)
    elif cohort == "annotate_b0":
        out = parse_annotate_b0_day(fname)
    elif cohort in {"annotate_b1", "annotate_b2"}:
        out = parse_annotate_b1_b2_day(fname, cohort)
    else:
        out = {
            "raw_day": None,
            "day_bin": None,
            "lab": None,
            "day_logic": "unknown",
        }

    out["file_name"] = fname
    return out

@torch.no_grad()
def dice_per_sample_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    dims = (1, 2, 3)
    inter = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.detach().cpu().numpy()

def make_eval_loader(sub_df, batch_size):
    ds = OrganoidSegDataset(sub_df, size=IMG_SIZE_UNET, train=False)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=g,
    )

def save_df(df_obj, path):
    df_obj.to_csv(path, index=False)
    print(f"Saved CSV: {path}")

def ci95_from_std(std, n):
    std = 0.0 if pd.isna(std) else float(std)
    n = int(n)
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)

def make_day_summary(sub_df):
    if len(sub_df) == 0:
        return pd.DataFrame(columns=[
            "day_bin", "n", "dice_mean", "dice_std", "dice_ci95",
            "dice_median", "dice_min", "dice_max"
        ])

    out = (
        sub_df.groupby("day_bin", as_index=False)
        .agg(
            n=("dice", "size"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_median=("dice", "median"),
            dice_min=("dice", "min"),
            dice_max=("dice", "max"),
        )
        .sort_values("day_bin")
        .reset_index(drop=True)
    )
    out["dice_std"] = out["dice_std"].fillna(0.0)
    out["dice_ci95"] = [
        ci95_from_std(s, n) for s, n in zip(out["dice_std"], out["n"])
    ]
    return out

def make_fold_day_summary(sub_df):
    if len(sub_df) == 0:
        return pd.DataFrame(columns=[
            "fold", "day_bin", "n", "dice_mean", "dice_std", "dice_ci95"
        ])

    out = (
        sub_df.groupby(["fold", "day_bin"], as_index=False)
        .agg(
            n=("dice", "size"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
        )
        .sort_values(["fold", "day_bin"])
        .reset_index(drop=True)
    )
    out["dice_std"] = out["dice_std"].fillna(0.0)
    out["dice_ci95"] = [
        ci95_from_std(s, n) for s, n in zip(out["dice_std"], out["n"])
    ]
    return out

def style_axes(ax):
    ax.set_title("")
    ax.set_ylabel("Dice")
    ax.grid(True, axis="y", alpha=0.3)

def get_dynamic_ylim(y, err, min_top=1.02, margin_frac=0.06):
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)

    if y.size == 0:
        return 0.0, min_top

    ymax = float(np.max(y + err))
    ymin = float(np.min(y - err))

    ymin = min(0.0, ymin)
    top = max(min_top, ymax)

    span = top - ymin
    if span <= 0:
        span = 0.1

    top = top + margin_frac * span
    return ymin, top

def ci95_yerr(ci95):
    ci95 = np.asarray(ci95, dtype=float)
    return np.vstack([ci95, ci95])

def plot_fold_bar(fold_df, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = fold_df["fold"].astype(str).tolist()
    y = fold_df["mean_dice"].to_numpy()
    ci95 = fold_df["dice_ci95"].to_numpy()

    ax.bar(x, y, yerr=ci95_yerr(ci95), capsize=4, color="tab:blue")
    ax.set_xlabel("Fold")
    ymin, ymax = get_dynamic_ylim(y, ci95)
    ax.set_ylim(ymin, ymax)
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

def plot_single_cohort_day_bar(day_summary_df, out_path, cohort_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(day_summary_df) > 0:
        x = day_summary_df["day_bin"].astype(int).astype(str).tolist()
        y = day_summary_df["dice_mean"].to_numpy()
        ci95 = day_summary_df["dice_ci95"].to_numpy()

        ax.bar(
            x, y,
            yerr=ci95_yerr(ci95),
            capsize=4,
            color=COHORT_COLOR[cohort_name]
        )

        ymin, ymax = get_dynamic_ylim(y, ci95)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.02)

    ax.set_xlabel("Day")
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

def plot_single_cohort_fold_day_bar(fold_day_df, out_path, cohort_name):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_y = []
    all_ci = []

    if len(fold_day_df) > 0:
        pivot_mean = fold_day_df.pivot(index="day_bin", columns="fold", values="dice_mean").sort_index()
        pivot_ci = fold_day_df.pivot(index="day_bin", columns="fold", values="dice_ci95").sort_index()

        x = np.arange(len(pivot_mean.index))
        n_folds = len(pivot_mean.columns)
        width = 0.8 / max(n_folds, 1)

        for i, fold_id in enumerate(pivot_mean.columns):
            y = pivot_mean[fold_id].to_numpy()
            ci95 = pivot_ci[fold_id].to_numpy()

            all_y.extend(y.tolist())
            all_ci.extend(ci95.tolist())

            color = FOLD_COLORS[i % len(FOLD_COLORS)]

            ax.bar(
                x + i * width - (n_folds - 1) * width / 2,
                y,
                width=width,
                yerr=ci95_yerr(ci95),
                capsize=3,
                alpha=0.9,
                color=color,
                label=f"Fold {fold_id}"
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in pivot_mean.index])

        ymin, ymax = get_dynamic_ylim(np.array(all_y), np.array(all_ci))
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.02)

    ax.set_xlabel("Day")
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

def plot_joint_cohort_day_bar(joint_day_summary, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    all_y = []
    all_ci = []

    if len(joint_day_summary) > 0:
        pivot_mean = joint_day_summary.pivot(index="day_bin", columns="cohort", values="dice_mean").sort_index()
        pivot_ci = joint_day_summary.pivot(index="day_bin", columns="cohort", values="dice_ci95").sort_index()

        x = np.arange(len(pivot_mean.index))
        cohorts = list(pivot_mean.columns)
        width = 0.8 / max(len(cohorts), 1)

        for i, cohort_name in enumerate(cohorts):
            y = pivot_mean[cohort_name].to_numpy()
            ci95 = pivot_ci[cohort_name].to_numpy()

            all_y.extend(y.tolist())
            all_ci.extend(ci95.tolist())

            ax.bar(
                x + i * width - (len(cohorts) - 1) * width / 2,
                y,
                width=width,
                yerr=ci95_yerr(ci95),
                capsize=3,
                color=COHORT_COLOR.get(cohort_name, "tab:gray")
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in pivot_mean.index])

        ymin, ymax = get_dynamic_ylim(np.array(all_y), np.array(all_ci))
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.02)

    ax.set_xlabel("Day")
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

def plot_data_paper_lab_joint(day_lab_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_y = []
    all_ci = []

    if len(day_lab_df) > 0:
        pivot_mean = day_lab_df.pivot(index="day_bin", columns="lab", values="dice_mean").sort_index()
        pivot_ci = day_lab_df.pivot(index="day_bin", columns="lab", values="dice_ci95").sort_index()

        x = np.arange(len(pivot_mean.index))
        labs = list(pivot_mean.columns)
        width = 0.8 / max(len(labs), 1)

        for i, lab_name in enumerate(labs):
            y = pivot_mean[lab_name].to_numpy()
            ci95 = pivot_ci[lab_name].to_numpy()

            all_y.extend(y.tolist())
            all_ci.extend(ci95.tolist())

            ax.bar(
                x + i * width - (len(labs) - 1) * width / 2,
                y,
                width=width,
                yerr=ci95_yerr(ci95),
                capsize=3,
                color=LAB_COLOR.get(lab_name, "tab:gray"),
                label=lab_name
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in pivot_mean.index])

        ymin, ymax = get_dynamic_ylim(np.array(all_y), np.array(all_ci))
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="lower left")
    else:
        ax.set_ylim(0, 1.02)

    ax.set_xlabel("Day")
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

def plot_data_paper_lab_separate(day_lab_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    global_y = []
    global_ci = []

    for lab_name in ["Lab A", "Lab B"]:
        sub = day_lab_df[day_lab_df["lab"] == lab_name].sort_values("day_bin")
        if len(sub) > 0:
            global_y.extend(sub["dice_mean"].tolist())
            global_ci.extend(sub["dice_ci95"].tolist())

    if len(global_y) > 0:
        ymin_global, ymax_global = get_dynamic_ylim(np.array(global_y), np.array(global_ci))
    else:
        ymin_global, ymax_global = (0.0, 1.02)

    for ax, lab_name in zip(axes, ["Lab A", "Lab B"]):
        sub = day_lab_df[day_lab_df["lab"] == lab_name].sort_values("day_bin")
        if len(sub) > 0:
            x = sub["day_bin"].astype(int).astype(str).tolist()
            y = sub["dice_mean"].to_numpy()
            ci95 = sub["dice_ci95"].to_numpy()

            ax.bar(
                x, y,
                yerr=ci95_yerr(ci95),
                capsize=4,
                color=LAB_COLOR[lab_name],
                label=lab_name
            )
            ax.legend(loc="lower left")

        ax.set_xlabel("Day")
        ax.set_ylim(ymin_global, ymax_global)
        style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

# # ============================================================
# # EVALUATION
# # ============================================================
# all_rows = []
# fold_rows = []

# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# for fold, tr_idx, va_idx in splits:
#     fold_dir = RUN_ROOT / f"fold_{fold:02d}"
#     ckpt_path = fold_dir / f"{MODEL_TYPE}_unetpp.pt"

#     if not ckpt_path.is_file():
#         print(f"[WARN] Missing checkpoint for fold {fold}: {ckpt_path}")
#         continue

#     va_df = df.iloc[va_idx].reset_index(drop=True)

#     print(f"\nFold {fold} validation counts by cohort:")
#     vc = va_df["cohort"].value_counts().sort_index()
#     for k, v in vc.items():
#         print(f"  {k}: {v}")

#     loader = make_eval_loader(va_df, batch_size=EVAL_BATCH_SIZE)

#     model = UNetPlusPlus(in_ch=3, out_ch=1, base=32).to(DEVICE)
#     state = torch.load(ckpt_path, map_location=DEVICE)
#     model.load_state_dict(state)
#     model.eval()

#     cursor = 0
#     fold_dices = []

#     for batch in tqdm(loader, desc=f"eval fold {fold}", leave=True):
#         x = batch["image"].to(DEVICE, non_blocking=True)
#         y = batch["mask"].to(DEVICE, non_blocking=True)

#         logits = model(x)
#         batch_dices = dice_per_sample_from_logits(logits, y)

#         bs = x.size(0)
#         batch_rows = va_df.iloc[cursor:cursor + bs].reset_index(drop=True)
#         cursor += bs

#         for i in range(bs):
#             row = {
#                 "fold": fold,
#                 "cohort": str(batch_rows.loc[i, "cohort"]),
#                 "stem": str(batch_rows.loc[i, "stem"]),
#                 "image_path": str(batch_rows.loc[i, "image_path"]),
#                 "mask_path": str(batch_rows.loc[i, "mask_path"]),
#                 "dice": float(batch_dices[i]),
#             }
#             row.update(parse_metadata_for_row(row))
#             all_rows.append(row)
#             fold_dices.append(float(batch_dices[i]))

#     fold_mean = float(np.mean(fold_dices)) if len(fold_dices) else np.nan
#     fold_std = float(np.std(fold_dices, ddof=1)) if len(fold_dices) > 1 else 0.0
#     fold_n = len(fold_dices)
#     fold_ci95 = ci95_from_std(fold_std, fold_n)

#     fold_rows.append({
#         "fold": fold,
#         "n_val": len(va_df),
#         "mean_dice": fold_mean,
#         "std_dice": fold_std,
#         "dice_ci95": fold_ci95,
#     })

#     del model, state, loader
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# eval_df = pd.DataFrame(all_rows)
# fold_df = pd.DataFrame(fold_rows)

# # ============================================================
# # SAVE JOINT
# # ============================================================
# save_df(eval_df, EVAL_DIR / "unetpp_val_per_image.csv")
# save_df(fold_df, EVAL_DIR / "unetpp_val_fold_summary.csv")
# plot_fold_bar(fold_df, EVAL_DIR / "unetpp_val_dice_by_fold.png")

# overall_mean_df = pd.DataFrame([{
#     "mean_dice": float(eval_df["dice"].mean()) if len(eval_df) else np.nan,
#     "n_images": int(len(eval_df)),
#     "n_folds": int(fold_df["fold"].nunique()) if len(fold_df) else 0,
# }])
# save_df(overall_mean_df, EVAL_DIR / "unetpp_val_overall_mean.csv")

# eval_day_df = eval_df[eval_df["day_bin"].notna()].copy()
# if len(eval_day_df) > 0:
#     eval_day_df["day_bin"] = eval_day_df["day_bin"].astype(int)

# joint_day_summary = (
#     eval_day_df.groupby(["cohort", "day_bin"], as_index=False)
#     .agg(
#         n=("dice", "size"),
#         dice_mean=("dice", "mean"),
#         dice_std=("dice", "std"),
#         dice_median=("dice", "median"),
#         dice_min=("dice", "min"),
#         dice_max=("dice", "max"),
#     )
#     .sort_values(["cohort", "day_bin"])
#     .reset_index(drop=True)
# ) if len(eval_day_df) > 0 else pd.DataFrame(columns=[
#     "cohort", "day_bin", "n", "dice_mean", "dice_std", "dice_ci95", "dice_median", "dice_min", "dice_max"
# ])

# if len(joint_day_summary) > 0:
#     joint_day_summary["dice_std"] = joint_day_summary["dice_std"].fillna(0.0)
#     joint_day_summary["dice_ci95"] = [
#         ci95_from_std(s, n) for s, n in zip(joint_day_summary["dice_std"], joint_day_summary["n"])
#     ]
# else:
#     joint_day_summary["dice_ci95"] = []

# save_df(joint_day_summary, EVAL_DIR / "unetpp_val_joint_day_summary.csv")
# plot_joint_cohort_day_bar(joint_day_summary, EVAL_DIR / "unetpp_val_dice_by_day_per_cohort.png")

# # ============================================================
# # PER-COHORT
# # ============================================================
# cohort_order = ["annotate_b0", "annotate_b1", "annotate_b2", "data_paper"]

# for cohort_name in cohort_order:
#     cohort_dir = EVAL_DIR / cohort_name
#     cohort_dir.mkdir(parents=True, exist_ok=True)

#     sub = eval_df[(eval_df["cohort"] == cohort_name) & (eval_df["day_bin"].notna())].copy()
#     if len(sub) > 0:
#         sub["day_bin"] = sub["day_bin"].astype(int)

#     day_summary = make_day_summary(sub)
#     fold_day_summary = make_fold_day_summary(sub)

#     save_df(sub, cohort_dir / f"{cohort_name}_per_image_with_day.csv")
#     save_df(day_summary, cohort_dir / f"{cohort_name}_day_summary.csv")
#     save_df(fold_day_summary, cohort_dir / f"{cohort_name}_fold_day_summary.csv")

#     cohort_overall_mean = pd.DataFrame([{
#         "label": COHORT_SHORT[cohort_name],
#         "mean_dice": float(sub["dice"].mean()) if len(sub) else np.nan,
#         "n_images": int(len(sub)),
#     }])
#     save_df(cohort_overall_mean, cohort_dir / f"{cohort_name}_overall_mean.csv")

#     plot_single_cohort_day_bar(
#         day_summary,
#         cohort_dir / f"{cohort_name}_day_bar.png",
#         cohort_name
#     )

#     plot_single_cohort_fold_day_bar(
#         fold_day_summary,
#         cohort_dir / f"{cohort_name}_fold_day_bar.png",
#         cohort_name
#     )

#     if cohort_name == "data_paper":
#         dp_lab_summary = (
#             sub[sub["lab"].notna()]
#             .groupby(["lab", "day_bin"], as_index=False)
#             .agg(
#                 n=("dice", "size"),
#                 dice_mean=("dice", "mean"),
#                 dice_std=("dice", "std"),
#                 dice_median=("dice", "median"),
#                 dice_min=("dice", "min"),
#                 dice_max=("dice", "max"),
#             )
#             .sort_values(["lab", "day_bin"])
#             .reset_index(drop=True)
#         ) if len(sub) > 0 else pd.DataFrame(columns=[
#             "lab", "day_bin", "n", "dice_mean", "dice_std", "dice_ci95", "dice_median", "dice_min", "dice_max"
#         ])

#         if len(dp_lab_summary) > 0:
#             dp_lab_summary["dice_std"] = dp_lab_summary["dice_std"].fillna(0.0)
#             dp_lab_summary["dice_ci95"] = [
#                 ci95_from_std(s, n) for s, n in zip(dp_lab_summary["dice_std"], dp_lab_summary["n"])
#             ]
#         else:
#             dp_lab_summary["dice_ci95"] = []

#         dp_lab_fold_summary = (
#             sub[sub["lab"].notna()]
#             .groupby(["fold", "lab", "day_bin"], as_index=False)
#             .agg(
#                 n=("dice", "size"),
#                 dice_mean=("dice", "mean"),
#                 dice_std=("dice", "std"),
#             )
#             .sort_values(["fold", "lab", "day_bin"])
#             .reset_index(drop=True)
#         ) if len(sub) > 0 else pd.DataFrame(columns=[
#             "fold", "lab", "day_bin", "n", "dice_mean", "dice_std", "dice_ci95"
#         ])

#         if len(dp_lab_fold_summary) > 0:
#             dp_lab_fold_summary["dice_std"] = dp_lab_fold_summary["dice_std"].fillna(0.0)
#             dp_lab_fold_summary["dice_ci95"] = [
#                 ci95_from_std(s, n) for s, n in zip(dp_lab_fold_summary["dice_std"], dp_lab_fold_summary["n"])
#             ]
#         else:
#             dp_lab_fold_summary["dice_ci95"] = []

#         save_df(dp_lab_summary, cohort_dir / "data_paper_lab_day_summary.csv")
#         save_df(dp_lab_fold_summary, cohort_dir / "data_paper_lab_fold_day_summary.csv")

#         dp_lab_overall_mean = (
#             sub[sub["lab"].notna()]
#             .groupby("lab", as_index=False)
#             .agg(
#                 mean_dice=("dice", "mean"),
#                 n_images=("dice", "size"),
#             )
#         )
#         save_df(dp_lab_overall_mean, cohort_dir / "data_paper_lab_overall_mean.csv")

#         plot_data_paper_lab_joint(
#             dp_lab_summary,
#             cohort_dir / "data_paper_lab_day_bar.png"
#         )

#         plot_data_paper_lab_separate(
#             dp_lab_summary,
#             cohort_dir / "data_paper_lab_day_bar_separate.png"
#         )

# print("\nPreview:")
# display(eval_df.head(20))
# display(fold_df.head())
# display(joint_day_summary.head(20))

# ============================================================
# Inference Visualization: Original | Predicted Mask | Overlay
# Loads the best fold UNet++ checkpoint automatically
# and runs inference on a list of image paths.
#
# MODES
# - BACK_PROJECT_MODE = True:
#       resize to 256x256 -> predict -> back-project mask to original size
# - BACK_PROJECT_MODE = False:
#       run on original image with pad-to-multiple-of-16
#
# Assumes:
# - UNetPlusPlus class is already defined
# - RUN_ROOT, DEVICE exist
# ============================================================

# import os
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image

# import cv2
# import torch

# # ------------------------------------------------------------
# # USER INPUT
# # ------------------------------------------------------------
# IMAGE_PATHS = [
#     # "/content/data/data_paper/imgs/org01_wt2D_d02_LabB.png",
#     "/content/2025-10-18 19-27-01 B2 TC ORG  44_Overlay.png",
#     # "/content/2025-10-18 19-39-29 B2 PDMS 5_1 ORG  6_Overlay.png",
#     "/content/2025-10-20 08-59-51 B2 TC ORG 25_Overlay.png",
#     "/content/G22 Org 15 D90 PDMS 10_1 .png",
#     "/content/org01_wt2D_d02_LabB.png",
#     "/content/org01_wt2D_d05_LabB.png",
# ]

# BACK_PROJECT_MODE = True
# IMG_SIZE_UNET = 256      # must match training resize
# THRESHOLD = 0.5
# BASE_CHANNELS = 32       # must match training
# ALPHA = 0.40

# # ------------------------------------------------------------
# # Helpers
# # ------------------------------------------------------------
# def load_rgb_image_pil(path):
#     with Image.open(path) as im:
#         return np.array(im.convert("RGB"))

# def resize_image_for_model(img_rgb, size=256):
#     img_r = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
#     return img_r

# def back_project_mask_to_original(mask_256, orig_h, orig_w):
#     mask_orig = cv2.resize(
#         mask_256.astype(np.uint8),
#         (orig_w, orig_h),
#         interpolation=cv2.INTER_NEAREST
#     )
#     return (mask_orig > 0).astype(np.uint8)

# def pad_to_multiple_of_16(img):
#     h, w = img.shape[:2]
#     new_h = int(math.ceil(h / 16) * 16)
#     new_w = int(math.ceil(w / 16) * 16)

#     pad_h = new_h - h
#     pad_w = new_w - w

#     top = pad_h // 2
#     bottom = pad_h - top
#     left = pad_w // 2
#     right = pad_w - left

#     img_pad = cv2.copyMakeBorder(
#         img, top, bottom, left, right,
#         borderType=cv2.BORDER_CONSTANT,
#         value=0
#     )

#     pad_info = {
#         "top": top,
#         "bottom": bottom,
#         "left": left,
#         "right": right,
#         "orig_h": h,
#         "orig_w": w,
#         "new_h": new_h,
#         "new_w": new_w,
#     }
#     return img_pad, pad_info

# def unpad_mask(mask, pad_info):
#     t = pad_info["top"]
#     b = pad_info["bottom"]
#     l = pad_info["left"]
#     r = pad_info["right"]

#     if b == 0:
#         h_slice = slice(t, None)
#     else:
#         h_slice = slice(t, -b)

#     if r == 0:
#         w_slice = slice(l, None)
#     else:
#         w_slice = slice(l, -r)

#     return mask[h_slice, w_slice]

# def preprocess_original_mode(img_rgb):
#     img_pad, pad_info = pad_to_multiple_of_16(img_rgb)
#     x = img_pad.astype(np.float32) / 255.0
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
#     return x, pad_info

# def preprocess_backproject_mode(img_rgb, size=256):
#     orig_h, orig_w = img_rgb.shape[:2]
#     img_r = resize_image_for_model(img_rgb, size=size)
#     x = img_r.astype(np.float32) / 255.0
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
#     meta = {
#         "orig_h": orig_h,
#         "orig_w": orig_w,
#         "size": size,
#     }
#     return x, meta

# def predict_mask_original_mode(model, img_rgb, threshold=0.5):
#     x, pad_info = preprocess_original_mode(img_rgb)
#     x = x.to(DEVICE)

#     with torch.no_grad():
#         logits = model(x)
#         probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

#     pred_pad = (probs > threshold).astype(np.uint8)
#     pred = unpad_mask(pred_pad, pad_info)
#     return pred

# def predict_mask_backproject_mode(model, img_rgb, threshold=0.5, size=256):
#     x, meta = preprocess_backproject_mode(img_rgb, size=size)
#     x = x.to(DEVICE)

#     with torch.no_grad():
#         logits = model(x)
#         probs_256 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

#     pred_256 = (probs_256 > threshold).astype(np.uint8)
#     pred_orig = back_project_mask_to_original(
#         pred_256,
#         orig_h=meta["orig_h"],
#         orig_w=meta["orig_w"]
#     )
#     return pred_orig

# def predict_mask(model, img_rgb, threshold=0.5, back_project_mode=True, size=256):
#     if back_project_mode:
#         return predict_mask_backproject_mode(
#             model, img_rgb, threshold=threshold, size=size
#         )
#     else:
#         return predict_mask_original_mode(
#             model, img_rgb, threshold=threshold
#         )

# def make_overlay(img_rgb, mask_bin, alpha=0.4):
#     overlay = img_rgb.copy()
#     red = np.array([255, 0, 0], dtype=np.uint8)

#     idx = mask_bin.astype(bool)
#     overlay[idx] = (
#         (1 - alpha) * overlay[idx].astype(np.float32) +
#         alpha * red.astype(np.float32)
#     ).astype(np.uint8)

#     return overlay

# def find_best_fold_checkpoint(run_root):
#     run_root = Path(run_root)
#     fold_dirs = sorted(run_root.glob("fold_*"))

#     best_fold = None
#     best_score = -float("inf")
#     best_ckpt = None

#     for fold_dir in fold_dirs:
#         hist_path = fold_dir / "history.csv"
#         ckpt_path = fold_dir / "best_unetpp.pt"

#         if not hist_path.is_file() or not ckpt_path.is_file():
#             continue

#         try:
#             hist = pd.read_csv(hist_path)
#             if "val_dice" not in hist.columns or len(hist) == 0:
#                 continue
#             fold_best = float(hist["val_dice"].max())
#         except Exception:
#             continue

#         if fold_best > best_score:
#             best_score = fold_best
#             best_fold = fold_dir.name
#             best_ckpt = ckpt_path

#     if best_ckpt is None:
#         raise FileNotFoundError(
#             f"Could not find any valid best_unetpp.pt + history.csv inside {run_root}"
#         )

#     return best_fold, best_score, best_ckpt

# # ------------------------------------------------------------
# # Load best fold model
# # ------------------------------------------------------------
# best_fold_name, best_fold_score, best_ckpt_path = find_best_fold_checkpoint(RUN_ROOT)

# print(f"Using checkpoint from {best_fold_name}")
# print(f"Best validation Dice found: {best_fold_score:.4f}")
# print(f"Checkpoint: {best_ckpt_path}")
# print(f"BACK_PROJECT_MODE: {BACK_PROJECT_MODE}")

# model = UNetPlusPlus(in_ch=3, out_ch=1, base=BASE_CHANNELS).to(DEVICE)
# state = torch.load(best_ckpt_path, map_location=DEVICE)
# model.load_state_dict(state)
# model.eval()

# # ------------------------------------------------------------
# # Run inference and visualize
# # ------------------------------------------------------------
# if len(IMAGE_PATHS) == 0:
#     print("Please fill IMAGE_PATHS with one or more image paths.")
# else:
#     n = len(IMAGE_PATHS)
#     fig, axes = plt.subplots(n, 3, figsize=(14, 4.5 * n))

#     if n == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for i, img_path in enumerate(IMAGE_PATHS):
#         img_path = str(img_path)
#         img_rgb = load_rgb_image_pil(img_path)

#         pred_mask = predict_mask(
#             model,
#             img_rgb,
#             threshold=THRESHOLD,
#             back_project_mode=BACK_PROJECT_MODE,
#             size=IMG_SIZE_UNET
#         )

#         overlay = make_overlay(img_rgb, pred_mask, alpha=ALPHA)
#         pred_vis = (pred_mask * 255).astype(np.uint8)

#         axes[i, 0].imshow(img_rgb)
#         axes[i, 0].set_title(f"Original\n{Path(img_path).name}")
#         axes[i, 0].axis("off")

#         axes[i, 1].imshow(pred_vis, cmap="gray", vmin=0, vmax=255)
#         mode_name = "Predicted Mask\n(256→orig)" if BACK_PROJECT_MODE else "Predicted Mask\n(orig padded)"
#         axes[i, 1].set_title(mode_name)
#         axes[i, 1].axis("off")

#         axes[i, 2].imshow(overlay)
#         axes[i, 2].set_title("Overlay")
#         axes[i, 2].axis("off")

#     plt.tight_layout()
#     plt.show()

"""## Excel Generation"""

# # ============================================================
# # UNet++ -> Back-projected predictions -> Morphology Excel
# # FINAL ROBUST VERSION
# # ============================================================

# import os
# import re
# import gc
# import math
# import json
# import cv2
# import numpy as np
# import pandas as pd
# import datetime as dt
# from pathlib import Path
# from PIL import Image
# from tqdm.auto import tqdm
# import torch

# # ------------------------------------------------------------
# # USER SETTINGS
# # ------------------------------------------------------------
# REPORT_UNIT = "mm"   # "mm" or "um"
# REFERENCE_JSON_PATH = "/mnt/data/reference_by_filename.json"   # change if needed

# IMG_SIZE_UNET_INFER = 256
# THRESHOLD = 0.5
# BASE_CHANNELS = 32
# BACK_PROJECT_MODE = True

# MIN_COMPONENT_SIZE_PX = 50
# KEEP_ONLY_LARGEST_COMPONENT = True

# LAB_DEFAULT_SUBSTRATE = {
#     "Lab A": "TC",
#     "Lab B": "TC",
# }

# LAB_UM_PER_PX = {
#     "Lab A": 500.0 / 158.0,
#     "Lab B": 500.0 / 167.0,
# }

# UM_PER_PX_BY_DAY_BATCHES = {
#     "day30": 3.066,
#     "day60": 3.064,
#     "day90": 3.064,
#     "day120": 3.064,
#     "day123": 3.064,
#     "day150": 3.064,
#     "day151": 3.064,
#     "default": 1.527,
# }

# OUTPUT_XLSX = RESULTS_ROOT / f"unetpp_predicted_organoid_measurements_{REPORT_UNIT}.xlsx"

# # ------------------------------------------------------------
# # DATES / BATCH MAP
# # ------------------------------------------------------------
# B0_DAY0_DATE = dt.date(2025, 7, 9)
# B1_B2_DAY0_DATE = dt.date(2025, 10, 18)

# COHORT_TO_BATCH = {
#     "annotate_b0": "0",
#     "annotate_b1": "1",
#     "annotate_b2": "2",
# }

# # ------------------------------------------------------------
# # UNIT HELPERS
# # ------------------------------------------------------------
# if REPORT_UNIT not in {"um", "mm"}:
#     raise ValueError("REPORT_UNIT must be 'um' or 'mm'")

# if REPORT_UNIT == "um":
#     LEN_UNIT_LABEL = "µm"
#     AREA_UNIT_LABEL = "µm²"
# else:
#     LEN_UNIT_LABEL = "mm"
#     AREA_UNIT_LABEL = "mm²"

# AREA_COL = f"Area ({AREA_UNIT_LABEL})"
# PERIM_COL = f"Perimeter ({LEN_UNIT_LABEL})"
# DIAM_COL = f"Diameter ({LEN_UNIT_LABEL})"

# # ------------------------------------------------------------
# # LOAD REFERENCE JSON
# # ------------------------------------------------------------
# if os.path.isfile(REFERENCE_JSON_PATH):
#     with open(REFERENCE_JSON_PATH, "r", encoding="utf-8") as f:
#         REFERENCE_BY_FILENAME = json.load(f)
#     print(f"Loaded reference JSON: {REFERENCE_JSON_PATH} | entries={len(REFERENCE_BY_FILENAME)}")
# else:
#     REFERENCE_BY_FILENAME = {}
#     print(f"[WARN] Reference JSON not found: {REFERENCE_JSON_PATH}")

# # ------------------------------------------------------------
# # HELPERS
# # ------------------------------------------------------------
# def clean_filename_variants(fname: str):
#     variants = []

#     def add(x):
#         if x and x not in variants:
#             variants.append(x)

#     base = os.path.basename(str(fname))
#     add(base)
#     add(base.strip())
#     add(re.sub(r"\s+", " ", base).strip())
#     add(re.sub(r"\s+\.(\w+)$", r".\1", base))

#     tmp = re.sub(r"\s+", " ", base).strip()
#     tmp = re.sub(r"\s+\.(\w+)$", r".\1", tmp)
#     add(tmp)

#     return variants

# def get_reference_meta_for_file(fname: str):
#     for cand in clean_filename_variants(fname):
#         if cand in REFERENCE_BY_FILENAME:
#             return REFERENCE_BY_FILENAME[cand], cand
#     return None, None

# def standardize_substrate(text):
#     if text is None:
#         return None

#     s = str(text).strip()
#     if s == "":
#         return None

#     s_up = s.upper().replace("_", ":").replace("-", ":")
#     s_up = re.sub(r"\s+", " ", s_up)

#     if re.search(r"\bTC\b", s_up):
#         return "TC"

#     # with PDMS explicitly
#     m = re.search(r"PDMS\s*(5|10|20)\s*:\s*1\b", s_up)
#     if m:
#         return f"PDMS {m.group(1)}:1"

#     m = re.search(r"PDMS\s*(5|10|20)\s+1\b", s_up)
#     if m:
#         return f"PDMS {m.group(1)}:1"

#     # without PDMS explicitly
#     m = re.search(r"\b(5|10|20)\s*:\s*1\b", s_up)
#     if m:
#         return f"PDMS {m.group(1)}:1"

#     return None

# # ------------------------------------------------------------
# # PARSERS
# # ------------------------------------------------------------
# DATE_RE = re.compile(r"\b(20\d{2})[-_](\d{2})[-_](\d{2})\b")

# def parse_date_from_filename(fname: str):
#     stem = Path(fname).stem
#     m = DATE_RE.search(stem)
#     if not m:
#         return None
#     y, mo, d = map(int, m.groups())
#     try:
#         return dt.date(y, mo, d)
#     except Exception:
#         return None

# def parse_day_from_filename(fname: str):
#     """
#     Robust day parser.
#     Priority patterns:
#       D4, d4, DAY4, day4, DAY 4, day 4, D 4
#     """
#     stem = Path(fname).stem

#     patterns = [
#         r"\bDAY[\s_:-]*0*(\d+)\b",
#         r"\bD[\s_:-]*0*(\d+)\b",
#     ]

#     for pat in patterns:
#         m = re.search(pat, stem, flags=re.IGNORECASE)
#         if m:
#             try:
#                 return int(m.group(1))
#             except Exception:
#                 pass
#     return None

# def parse_organoid_id_from_filename(fname: str):
#     """
#     Robust organoid parser.
#     Handles:
#       org01, ORG01
#       org 2, ORG 2
#       org#2, ORG#2
#       org 2_, ORG 2_
#     """
#     stem = Path(fname).stem

#     patterns = [
#         r"\bORG(?:ANOID)?\b[\s_:#-]*0*(\d+)\b",
#         r"\borg(?:anoid)?[\s_:#-]*0*(\d+)\b",
#     ]

#     for pat in patterns:
#         m = re.search(pat, stem, flags=re.IGNORECASE)
#         if m:
#             try:
#                 return int(m.group(1))
#             except Exception:
#                 return m.group(1)
#     return None

# def parse_sample_type_from_filename(fname: str, cohort: str):
#     if cohort == "data_paper":
#         return None

#     s = Path(fname).stem.upper()

#     if "G22" in s:
#         return "G22"

#     if "WTC" in s or "WT2D" in s or "WTC2D" in s or "W TC" in s:
#         return "WTC"

#     # id-based fallback
#     oid = parse_organoid_id_from_filename(fname)
#     if oid is not None:
#         try:
#             return "G22" if int(oid) < 25 else "WTC"
#         except Exception:
#             pass

#     return None

# def parse_substrate_from_filename(fname: str):
#     stem = Path(fname).stem

#     if re.search(r"\bTC\b", stem, flags=re.IGNORECASE):
#         return "TC"

#     # explicit PDMS
#     for pat in [
#         r"\bPDMS\s*5[_:\-\s]?1\b",
#         r"\bPDMS\s*10[_:\-\s]?1\b",
#         r"\bPDMS\s*20[_:\-\s]?1\b",
#     ]:
#         m = re.search(pat, stem, flags=re.IGNORECASE)
#         if m:
#             return standardize_substrate(m.group(0))

#     # shorthand without PDMS
#     for pat in [
#         r"\b5[_:\-]1\b",
#         r"\b10[_:\-]1\b",
#         r"\b20[_:\-]1\b",
#     ]:
#         m = re.search(pat, stem, flags=re.IGNORECASE)
#         if m:
#             return standardize_substrate(m.group(0))

#     return None

# def parse_lab_from_filename(fname: str):
#     stem = Path(fname).stem

#     m = re.search(r"_(LabA|LabB)(?:_|$)", stem, flags=re.IGNORECASE)
#     if m:
#         raw = m.group(1).lower()
#         if raw == "laba":
#             return "Lab A"
#         if raw == "labb":
#             return "Lab B"

#     return None

# def snap_to_annotate_bin(day: int):
#     if day is None:
#         return None
#     day = int(day)
#     if day <= 14:
#         return max(0, day)
#     later_bins = [30, 60, 90, 120, 150]
#     return min(later_bins, key=lambda x: (abs(x - day), x))

# # ------------------------------------------------------------
# # METADATA COMBINER
# # filename first -> reference -> date fallback
# # ------------------------------------------------------------
# def build_export_metadata(image_path: str, cohort: str):
#     fname = os.path.basename(str(image_path))

#     # filename first
#     file_day = parse_day_from_filename(fname)
#     file_organoid_id = parse_organoid_id_from_filename(fname)
#     file_sample_type = parse_sample_type_from_filename(fname, cohort)
#     file_substrate = parse_substrate_from_filename(fname)
#     file_lab = parse_lab_from_filename(fname)
#     file_date = parse_date_from_filename(fname)

#     # reference second
#     ref_meta, matched_ref_key = get_reference_meta_for_file(fname)

#     ref_day = None
#     ref_organoid_id = None
#     ref_sample_type = None
#     ref_substrate = None

#     if ref_meta is not None:
#         ref_day = ref_meta.get("Day", None)
#         ref_organoid_id = ref_meta.get("Organoid ID", None)
#         ref_sample_type = ref_meta.get("Sample Type", None)
#         ref_substrate = ref_meta.get("Substrate", None)

#         if ref_day == "":
#             ref_day = None
#         if ref_organoid_id == "":
#             ref_organoid_id = None
#         if ref_sample_type == "":
#             ref_sample_type = None
#         if ref_substrate == "":
#             ref_substrate = None

#     # combine with filename priority
#     day = file_day if file_day not in [None, ""] else ref_day
#     organoid_id = file_organoid_id if file_organoid_id not in [None, ""] else ref_organoid_id
#     sample_type = file_sample_type if file_sample_type not in [None, ""] else ref_sample_type
#     substrate = file_substrate if file_substrate not in [None, ""] else ref_substrate

#     substrate = standardize_substrate(substrate)

#     # date fallback only if still missing
#     if day in [None, ""]:
#         if file_date is None and ref_meta is not None:
#             date_str = ref_meta.get("Date", None)
#             if date_str not in [None, ""]:
#                 try:
#                     file_date = dt.datetime.strptime(str(date_str), "%Y-%m-%d").date()
#                 except Exception:
#                     file_date = None

#         if file_date is not None:
#             if cohort == "annotate_b0":
#                 raw_day = max(0, (file_date - B0_DAY0_DATE).days)
#                 day = snap_to_annotate_bin(raw_day)
#             elif cohort in {"annotate_b1", "annotate_b2"}:
#                 raw_day = max(0, (file_date - B1_B2_DAY0_DATE).days)
#                 day = snap_to_annotate_bin(raw_day)

#     # cohort-specific
#     if cohort == "data_paper":
#         lab = file_lab

#         # filename parser for labs should fill these
#         if organoid_id in [None, ""]:
#             organoid_id = file_organoid_id
#         if day in [None, ""]:
#             day = file_day

#         if lab == "Lab A":
#             batch = "LabA"
#         elif lab == "Lab B":
#             batch = "LabB"
#         else:
#             batch = ""

#         if substrate is None:
#             substrate = LAB_DEFAULT_SUBSTRATE.get(lab, "TC")

#         sample_type = None
#     else:
#         lab = None
#         batch = COHORT_TO_BATCH.get(cohort, "")

#     # clean final types
#     if organoid_id not in [None, ""]:
#         try:
#             organoid_id = int(organoid_id)
#         except Exception:
#             pass

#     if day not in [None, ""]:
#         try:
#             day = int(day)
#         except Exception:
#             pass

#     sample_type = sample_type if sample_type in {"G22", "WTC"} else None
#     substrate = standardize_substrate(substrate)

#     return {
#         "Organoid ID": organoid_id,
#         "Sample Type": sample_type,
#         "Substrate": substrate,
#         "Day": day,
#         "Batch": batch,
#         "File": fname,
#         "_lab": lab,
#         "_matched_ref_key": matched_ref_key,
#     }

# # ------------------------------------------------------------
# # SCALE HELPERS
# # ------------------------------------------------------------
# def um_per_px_for_day(day: int, dct: dict) -> float:
#     return float(dct.get(f"day{int(day)}", dct["default"]))

# def get_um_per_px(batch, lab, day):
#     if batch in {"LabA", "LabB"}:
#         if lab == "Lab A":
#             return float(LAB_UM_PER_PX["Lab A"])
#         if lab == "Lab B":
#             return float(LAB_UM_PER_PX["Lab B"])
#         return float(LAB_UM_PER_PX["Lab A"])

#     if day is not None and str(day) != "":
#         try:
#             return um_per_px_for_day(int(day), UM_PER_PX_BY_DAY_BATCHES)
#         except Exception:
#             pass

#     return float(UM_PER_PX_BY_DAY_BATCHES["default"])

# # ------------------------------------------------------------
# # IMAGE / MODEL HELPERS
# # ------------------------------------------------------------
# def load_rgb_image_pil(path):
#     with Image.open(path) as im:
#         return np.array(im.convert("RGB"))

# def resize_image_for_model(img_rgb, size=256):
#     return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

# def back_project_mask_to_original(mask_256, orig_h, orig_w):
#     mask_orig = cv2.resize(mask_256.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return (mask_orig > 0).astype(np.uint8)

# def preprocess_backproject_mode(img_rgb, size=256):
#     orig_h, orig_w = img_rgb.shape[:2]
#     img_r = resize_image_for_model(img_rgb, size=size)
#     x = img_r.astype(np.float32) / 255.0
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
#     return x, {"orig_h": orig_h, "orig_w": orig_w}

# def predict_mask(model, img_rgb, threshold=0.5, size=256):
#     x, meta = preprocess_backproject_mode(img_rgb, size=size)
#     x = x.to(DEVICE)

#     with torch.no_grad():
#         logits = model(x)
#         probs_256 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

#     pred_256 = (probs_256 > threshold).astype(np.uint8)
#     return back_project_mask_to_original(pred_256, meta["orig_h"], meta["orig_w"])

# def find_best_fold_checkpoint(run_root):
#     run_root = Path(run_root)
#     fold_dirs = sorted(run_root.glob("fold_*"))

#     best_fold = None
#     best_score = -float("inf")
#     best_ckpt = None

#     for fold_dir in fold_dirs:
#         hist_path = fold_dir / "history.csv"
#         ckpt_path = fold_dir / "best_unetpp.pt"

#         if not hist_path.is_file() or not ckpt_path.is_file():
#             continue

#         try:
#             hist = pd.read_csv(hist_path)
#             if "val_dice" not in hist.columns or len(hist) == 0:
#                 continue
#             fold_best = float(hist["val_dice"].max())
#         except Exception:
#             continue

#         if fold_best > best_score:
#             best_score = fold_best
#             best_fold = fold_dir.name
#             best_ckpt = ckpt_path

#     if best_ckpt is None:
#         raise FileNotFoundError(f"Could not find any valid best_unetpp.pt + history.csv inside {run_root}")

#     return best_fold, best_score, best_ckpt

# # ------------------------------------------------------------
# # MORPHOLOGY HELPERS
# # ------------------------------------------------------------
# def clean_small_components(mask01, min_size_px=50, keep_only_largest=True):
#     mask01 = (mask01 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)

#     if num_labels <= 1:
#         return mask01

#     kept = np.zeros_like(mask01)
#     components = []

#     for lab in range(1, num_labels):
#         area = stats[lab, cv2.CC_STAT_AREA]
#         if area >= min_size_px:
#             components.append((lab, area))

#     if not components:
#         return np.zeros_like(mask01)

#     if keep_only_largest:
#         lab_keep = max(components, key=lambda x: x[1])[0]
#         kept[labels == lab_keep] = 1
#     else:
#         for lab_keep, _ in components:
#             kept[labels == lab_keep] = 1

#     return kept.astype(np.uint8)

# def measure_metrics_from_mask(mask01, um_per_px):
#     mask01 = (mask01 > 0).astype(np.uint8)
#     area_px = float(mask01.sum())

#     if area_px <= 0:
#         return {
#             "area_px": 0.0,
#             "perimeter_px": 0.0,
#             "diameter_px": 0.0,
#             "circularity": np.nan,
#             "area_report": 0.0,
#             "perimeter_report": 0.0,
#             "diameter_report": 0.0,
#         }

#     contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     perimeter_px = float(sum(cv2.arcLength(cnt, closed=True) for cnt in contours)) if len(contours) else 0.0

#     diameter_px = float(math.sqrt(4.0 * area_px / math.pi))
#     circularity = float(4.0 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else np.nan

#     area_um2 = area_px * (um_per_px ** 2)
#     perimeter_um = perimeter_px * um_per_px
#     diameter_um = diameter_px * um_per_px

#     if REPORT_UNIT == "um":
#         area_report = area_um2
#         perimeter_report = perimeter_um
#         diameter_report = diameter_um
#     else:
#         area_report = area_um2 / 1_000_000.0
#         perimeter_report = perimeter_um / 1000.0
#         diameter_report = diameter_um / 1000.0

#     return {
#         "area_px": area_px,
#         "perimeter_px": perimeter_px,
#         "diameter_px": diameter_px,
#         "circularity": circularity,
#         "area_report": area_report,
#         "perimeter_report": perimeter_report,
#         "diameter_report": diameter_report,
#     }

# # ------------------------------------------------------------
# # LOAD MODEL
# # ------------------------------------------------------------
# best_fold_name, best_fold_score, best_ckpt_path = find_best_fold_checkpoint(RUN_ROOT)

# print(f"Using checkpoint from {best_fold_name}")
# print(f"Best validation Dice found: {best_fold_score:.4f}")
# print(f"Checkpoint: {best_ckpt_path}")

# model = UNetPlusPlus(in_ch=3, out_ch=1, base=BASE_CHANNELS).to(DEVICE)
# state = torch.load(best_ckpt_path, map_location=DEVICE)
# model.load_state_dict(state)
# model.eval()

# # ------------------------------------------------------------
# # RUN OVER DATAFRAME
# # ------------------------------------------------------------
# rows = []
# work_df = df.copy().reset_index(drop=True)

# for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Predict + export"):
#     image_path = str(row["image_path"])
#     cohort = str(row["cohort"])

#     try:
#         meta = build_export_metadata(image_path, cohort)

#         img_rgb = load_rgb_image_pil(image_path)
#         pred_mask = predict_mask(
#             model,
#             img_rgb,
#             threshold=THRESHOLD,
#             size=IMG_SIZE_UNET_INFER
#         )

#         pred_mask = clean_small_components(
#             pred_mask,
#             min_size_px=MIN_COMPONENT_SIZE_PX,
#             keep_only_largest=KEEP_ONLY_LARGEST_COMPONENT
#         )

#         um_per_px = get_um_per_px(
#             batch=meta["Batch"],
#             lab=meta["_lab"],
#             day=meta["Day"]
#         )

#         meas = measure_metrics_from_mask(pred_mask, um_per_px)

#         rows.append({
#             "Organoid ID": meta["Organoid ID"],
#             "Sample Type": meta["Sample Type"],
#             "Substrate": meta["Substrate"],
#             "Day": meta["Day"],
#             "Batch": meta["Batch"],
#             "File": meta["File"],
#             "Label": False,
#             "Area (px²)": meas["area_px"],
#             AREA_COL: meas["area_report"],
#             "Perimeter (px)": meas["perimeter_px"],
#             PERIM_COL: meas["perimeter_report"],
#             "Diameter (px)": meas["diameter_px"],
#             DIAM_COL: meas["diameter_report"],
#             "Circularity": meas["circularity"],
#             "Scale µm/px": um_per_px,
#         })

#     except Exception as e:
#         print(f"[WARN] Failed on {image_path}: {repr(e)}")
#         rows.append({
#             "Organoid ID": None,
#             "Sample Type": None,
#             "Substrate": None,
#             "Day": None,
#             "Batch": "",
#             "File": os.path.basename(image_path),
#             "Label": False,
#             "Area (px²)": None,
#             AREA_COL: None,
#             "Perimeter (px)": None,
#             PERIM_COL: None,
#             "Diameter (px)": None,
#             DIAM_COL: None,
#             "Circularity": None,
#             "Scale µm/px": None,
#         })

# results_df = pd.DataFrame(rows)

# results_df["Substrate"] = results_df["Substrate"].apply(standardize_substrate)

# batch_order = {"0": 0, "1": 1, "2": 2, "LabA": 3, "LabB": 4}
# results_df["_batch_sort"] = results_df["Batch"].map(batch_order).fillna(999)
# results_df["_day_sort"] = pd.to_numeric(results_df["Day"], errors="coerce").fillna(999999)
# results_df["Substrate"] = results_df["Substrate"].fillna("")
# results_df["File"] = results_df["File"].fillna("")

# results_df = (
#     results_df
#     .sort_values(by=["_batch_sort", "_day_sort", "Substrate", "File"], na_position="last")
#     .drop(columns=["_batch_sort", "_day_sort"])
#     .reset_index(drop=True)
# )

# results_df = results_df[
#     [
#         "Organoid ID",
#         "Sample Type",
#         "Substrate",
#         "Day",
#         "Batch",
#         "File",
#         "Label",
#         # "Area (px²)",
#         AREA_COL,
#         # "Perimeter (px)",
#         PERIM_COL,
#         # "Diameter (px)",
#         DIAM_COL,
#         "Circularity",
#         "Scale µm/px",
#     ]
# ]

# OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
# with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
#     results_df.to_excel(writer, sheet_name="Results", index=False)

# print(f"Saved Excel: {OUTPUT_XLSX}")
# display(results_df.head(30))

"""## Plots from Excel"""

# # ============================================================
# # Plot grid from Excel
# #
# # Rows:
# #   0, 1, 2, LabA, LabB
# #
# # Columns:
# #   Area, Perimeter, Diameter, Circularity
# #
# # Rules:
# # - batch 1 uses BAR plots
# # - x-limit per row = highest day + 10
# # - for batch 2, keep early ticks only as 0 and 14
# # ============================================================

# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# XLSX_PATH = OUTPUT_XLSX

# BATCH_ORDER = ["0", "1", "2", "LabA", "LabB"]
# SUBSTRATE_ORDER = ["TC", "PDMS 5:1", "PDMS 10:1", "PDMS 20:1"]

# if REPORT_UNIT == "um":
#     AREA_COL = "Area (µm²)"
#     PERIM_COL = "Perimeter (µm)"
#     DIAM_COL = "Diameter (µm)"
# else:
#     AREA_COL = "Area (mm²)"
#     PERIM_COL = "Perimeter (mm)"
#     DIAM_COL = "Diameter (mm)"

# CIRC_COL = "Circularity"

# def standardize_substrate(text):
#     if text is None:
#         return None

#     s = str(text).strip()
#     if s == "":
#         return None

#     s_up = s.upper().replace("_", ":").replace("-", ":")
#     s_up = re.sub(r"\s+", " ", s_up)

#     if re.search(r"\bTC\b", s_up):
#         return "TC"

#     m = re.search(r"PDMS\s*(5|10|20)\s*:\s*1\b", s_up)
#     if m:
#         return f"PDMS {m.group(1)}:1"

#     m = re.search(r"\b(5|10|20)\s*:\s*1\b", s_up)
#     if m:
#         return f"PDMS {m.group(1)}:1"

#     return None

# def sem(x: pd.Series) -> float:
#     x = x.dropna()
#     n = len(x)
#     if n <= 1:
#         return np.nan
#     return float(x.std(ddof=1) / np.sqrt(n))

# # ------------------------------------------------------------
# # LOAD + CLEAN
# # ------------------------------------------------------------
# dfp = pd.read_excel(XLSX_PATH, sheet_name="Results")

# dfp["Day_num"] = pd.to_numeric(dfp["Day"], errors="coerce")
# for c in [AREA_COL, PERIM_COL, DIAM_COL, CIRC_COL]:
#     dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

# dfp["Substrate"] = dfp["Substrate"].apply(standardize_substrate)
# dfp["Batch"] = dfp["Batch"].astype(str).str.strip()

# dfp = dfp.dropna(subset=["Day_num"])
# dfp = dfp[dfp["Day_num"] >= 0]
# dfp = dfp[dfp["Batch"].isin(BATCH_ORDER)]
# dfp = dfp[dfp["Substrate"].isin(SUBSTRATE_ORDER)]

# # ------------------------------------------------------------
# # AGGREGATE
# # ------------------------------------------------------------
# agg = (
#     dfp.groupby(["Batch", "Day_num", "Substrate"], as_index=False)
#        .agg(
#            Area_mean=(AREA_COL, "mean"), Area_sem=(AREA_COL, sem),
#            Per_mean=(PERIM_COL, "mean"), Per_sem=(PERIM_COL, sem),
#            Dia_mean=(DIAM_COL, "mean"), Dia_sem=(DIAM_COL, sem),
#            Cir_mean=(CIRC_COL, "mean"), Cir_sem=(CIRC_COL, sem),
#        )
# )

# # ------------------------------------------------------------
# # TICK HELPER
# # ------------------------------------------------------------
# def build_xticks_for_batch(raw_batch, batch_name):
#     xticks = sorted(raw_batch["Day_num"].dropna().astype(int).unique().tolist())
#     if len(xticks) == 0:
#         return [0]

#     if batch_name == "2":
#         # keep only 0 and 14 in early range, then later bins
#         out = []
#         if any(x == 0 for x in xticks):
#             out.append(0)
#         if any(x == 14 for x in xticks):
#             out.append(14)
#         later = [x for x in xticks if x > 14]
#         out.extend(later)
#         return out if len(out) else [0]

#     return xticks

# # ------------------------------------------------------------
# # PLOT
# # ------------------------------------------------------------
# n_rows = len(BATCH_ORDER)
# n_cols = 4

# fig, axes = plt.subplots(
#     n_rows, n_cols,
#     figsize=(22, 4.3 * n_rows),
#     constrained_layout=True,
#     squeeze=False
# )

# plot_specs = [
#     ("Area",        "Area_mean", "Area_sem", AREA_COL),
#     ("Perimeter",   "Per_mean",  "Per_sem",  PERIM_COL),
#     ("Diameter",    "Dia_mean",  "Dia_sem",  DIAM_COL),
#     ("Circularity", "Cir_mean",  "Cir_sem",  CIRC_COL),
# ]

# for r, batch_name in enumerate(BATCH_ORDER):
#     df_batch = agg[agg["Batch"] == batch_name].copy()
#     raw_batch = dfp[dfp["Batch"] == batch_name].copy()

#     if raw_batch.empty:
#         highest_day = 0
#         max_day = 10
#         xticks_present = [0]
#     else:
#         highest_day = int(np.nanmax(raw_batch["Day_num"].to_numpy()))
#         max_day = highest_day + 10
#         xticks_present = build_xticks_for_batch(raw_batch, batch_name)

#     for c, (title, ymean, ysem, ylabel) in enumerate(plot_specs):
#         ax = axes[r, c]

#         # batch 1 as bar plot
#         if batch_name == "1":
#             day_vals = sorted(df_batch["Day_num"].dropna().unique().tolist())
#             x = np.arange(len(day_vals), dtype=float)

#             if len(day_vals) > 0:
#                 present_subs = [s for s in SUBSTRATE_ORDER if s in df_batch["Substrate"].unique()]
#                 if len(present_subs) == 0:
#                     present_subs = SUBSTRATE_ORDER

#                 width = 0.8 / max(len(present_subs), 1)

#                 for i, sub in enumerate(present_subs):
#                     dsub = df_batch[df_batch["Substrate"] == sub].sort_values("Day_num")
#                     dsub = dsub.set_index("Day_num").reindex(day_vals)

#                     ax.bar(
#                         x + i * width - (len(present_subs) - 1) * width / 2,
#                         dsub[ymean].to_numpy(),
#                         width=width,
#                         yerr=dsub[ysem].to_numpy(),
#                         capsize=3,
#                         alpha=0.9,
#                         label=sub,
#                     )

#                 ax.set_xticks(x)
#                 # for batch 1, show only actual present days
#                 ax.set_xticklabels([str(int(v)) for v in day_vals])

#         else:
#             for sub in SUBSTRATE_ORDER:
#                 dsub = df_batch[df_batch["Substrate"] == sub].sort_values("Day_num")
#                 if dsub.empty:
#                     continue

#                 ax.errorbar(
#                     dsub["Day_num"],
#                     dsub[ymean],
#                     yerr=dsub[ysem],
#                     marker="o",
#                     capsize=3,
#                     linewidth=2,
#                     label=sub,
#                 )

#             ax.set_xlim(0, max_day)
#             ax.set_xticks(xticks_present)

#         if c == 0:
#             ax.set_ylabel(f"Batch {batch_name}\n{ylabel}")
#         else:
#             ax.set_ylabel(ylabel)

#         if r == 0:
#             ax.set_title(title)

#         ax.set_xlabel("Day")
#         ax.grid(True, alpha=0.25)

#         handles, labels = ax.get_legend_handles_labels()
#         if handles:
#             ax.legend(title="Substrate", loc="best")

# plt.show()

# ============================================================
# LABS DATA ONLY
# UNet++ -> Back-projected predictions -> Morphology Excel
#
# Uses only data_paper / LabA / LabB files.
# Parses:
#   - organoid id from org01, org02, ...
#   - day from d02, d05, d30, ...
#   - batch from LabA / LabB
#
# Final columns:
#   Organoid ID | Sample Type | Substrate | Day | Batch | File | Label |
#   Area (px²) | Area (...) | Perimeter (px) | Perimeter (...) |
#   Diameter (px) | Diameter (...) | Circularity | Scale µm/px
# ============================================================

# import os
# import re
# import gc
# import math
# import cv2
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# from tqdm.auto import tqdm
# import torch

# # ------------------------------------------------------------
# # USER SETTINGS
# # ------------------------------------------------------------
# REPORT_UNIT = "mm"   # "mm" or "um"

# IMG_SIZE_UNET_INFER = 256
# THRESHOLD = 0.5
# BASE_CHANNELS = 32
# BACK_PROJECT_MODE = True

# MIN_COMPONENT_SIZE_PX = 50
# KEEP_ONLY_LARGEST_COMPONENT = True

# LAB_DEFAULT_SUBSTRATE = {
#     "Lab A": "TC",
#     "Lab B": "TC",
# }

# LAB_UM_PER_PX = {
#     "Lab A": 500.0 / 158.0,
#     "Lab B": 500.0 / 167.0,
# }

# OUTPUT_XLSX = RESULTS_ROOT / f"unetpp_labs_only_measurements_{REPORT_UNIT}.xlsx"

# # ------------------------------------------------------------
# # UNIT HELPERS
# # ------------------------------------------------------------
# if REPORT_UNIT not in {"um", "mm"}:
#     raise ValueError("REPORT_UNIT must be 'um' or 'mm'")

# if REPORT_UNIT == "um":
#     LEN_UNIT_LABEL = "µm"
#     AREA_UNIT_LABEL = "µm²"
# else:
#     LEN_UNIT_LABEL = "mm"
#     AREA_UNIT_LABEL = "mm²"

# AREA_COL = f"Area ({AREA_UNIT_LABEL})"
# PERIM_COL = f"Perimeter ({LEN_UNIT_LABEL})"
# DIAM_COL = f"Diameter ({LEN_UNIT_LABEL})"

# # ------------------------------------------------------------
# # FILTER TO LABS DATA ONLY
# # ------------------------------------------------------------
# labs_df = df[df["cohort"] == "data_paper"].copy().reset_index(drop=True)

# if len(labs_df) == 0:
#     raise ValueError("No data_paper rows found in df.")

# print(f"Using only labs data rows: {len(labs_df)}")

# # ------------------------------------------------------------
# # PARSERS
# # ------------------------------------------------------------
# def parse_lab_from_filename(fname: str):
#     stem = Path(fname).stem
#     m = re.search(r"_(LabA|LabB)(?:_|$)", stem, flags=re.IGNORECASE)
#     if not m:
#         return None
#     raw = m.group(1).lower()
#     if raw == "laba":
#         return "Lab A"
#     if raw == "labb":
#         return "Lab B"
#     return None

# def parse_day_from_labs_filename(fname: str):
#     stem = Path(fname).stem

#     # examples: _d02_, _d5_, _d30_
#     m = re.search(r"_d0*(\d+)(?:_|$)", stem, flags=re.IGNORECASE)
#     if m:
#         try:
#             return int(m.group(1))
#         except Exception:
#             pass
#     return None

# def parse_organoid_id_from_labs_filename(fname: str):
#     stem = Path(fname).stem

#     # examples: org01, org02, org10
#     m = re.search(r"\borg0*(\d+)\b", stem, flags=re.IGNORECASE)
#     if m:
#         try:
#             return int(m.group(1))
#         except Exception:
#             pass

#     # fallback a bit looser
#     m = re.search(r"\borg0*(\d+)", stem, flags=re.IGNORECASE)
#     if m:
#         try:
#             return int(m.group(1))
#         except Exception:
#             pass

#     return None

# def build_labs_metadata(image_path: str):
#     fname = os.path.basename(str(image_path))

#     lab = parse_lab_from_filename(fname)
#     day = parse_day_from_labs_filename(fname)
#     organoid_id = parse_organoid_id_from_labs_filename(fname)

#     if lab == "Lab A":
#         batch = "LabA"
#     elif lab == "Lab B":
#         batch = "LabB"
#     else:
#         batch = ""

#     substrate = LAB_DEFAULT_SUBSTRATE.get(lab, "TC")

#     return {
#         "Organoid ID": organoid_id,
#         "Sample Type": None,
#         "Substrate": substrate,
#         "Day": day,
#         "Batch": batch,
#         "File": fname,
#         "_lab": lab,
#     }

# # ------------------------------------------------------------
# # SCALE HELPERS
# # ------------------------------------------------------------
# def get_um_per_px_for_lab(lab: str):
#     if lab == "Lab A":
#         return float(LAB_UM_PER_PX["Lab A"])
#     if lab == "Lab B":
#         return float(LAB_UM_PER_PX["Lab B"])
#     return float(LAB_UM_PER_PX["Lab A"])

# # ------------------------------------------------------------
# # IMAGE / MODEL HELPERS
# # ------------------------------------------------------------
# def load_rgb_image_pil(path):
#     with Image.open(path) as im:
#         return np.array(im.convert("RGB"))

# def resize_image_for_model(img_rgb, size=256):
#     return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

# def back_project_mask_to_original(mask_256, orig_h, orig_w):
#     mask_orig = cv2.resize(
#         mask_256.astype(np.uint8),
#         (orig_w, orig_h),
#         interpolation=cv2.INTER_NEAREST
#     )
#     return (mask_orig > 0).astype(np.uint8)

# def preprocess_backproject_mode(img_rgb, size=256):
#     orig_h, orig_w = img_rgb.shape[:2]
#     img_r = resize_image_for_model(img_rgb, size=size)
#     x = img_r.astype(np.float32) / 255.0
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
#     return x, {"orig_h": orig_h, "orig_w": orig_w}

# def predict_mask(model, img_rgb, threshold=0.5, size=256):
#     x, meta = preprocess_backproject_mode(img_rgb, size=size)
#     x = x.to(DEVICE)

#     with torch.no_grad():
#         logits = model(x)
#         probs_256 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

#     pred_256 = (probs_256 > threshold).astype(np.uint8)
#     return back_project_mask_to_original(pred_256, meta["orig_h"], meta["orig_w"])

# def find_best_fold_checkpoint(run_root):
#     run_root = Path(run_root)
#     fold_dirs = sorted(run_root.glob("fold_*"))

#     best_fold = None
#     best_score = -float("inf")
#     best_ckpt = None

#     for fold_dir in fold_dirs:
#         hist_path = fold_dir / "history.csv"
#         ckpt_path = fold_dir / "best_unetpp.pt"

#         if not hist_path.is_file() or not ckpt_path.is_file():
#             continue

#         try:
#             hist = pd.read_csv(hist_path)
#             if "val_dice" not in hist.columns or len(hist) == 0:
#                 continue
#             fold_best = float(hist["val_dice"].max())
#         except Exception:
#             continue

#         if fold_best > best_score:
#             best_score = fold_best
#             best_fold = fold_dir.name
#             best_ckpt = ckpt_path

#     if best_ckpt is None:
#         raise FileNotFoundError(f"Could not find any valid best_unetpp.pt + history.csv inside {run_root}")

#     return best_fold, best_score, best_ckpt

# # ------------------------------------------------------------
# # MORPHOLOGY HELPERS
# # ------------------------------------------------------------
# def clean_small_components(mask01, min_size_px=50, keep_only_largest=True):
#     mask01 = (mask01 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)

#     if num_labels <= 1:
#         return mask01

#     kept = np.zeros_like(mask01)
#     components = []

#     for lab in range(1, num_labels):
#         area = stats[lab, cv2.CC_STAT_AREA]
#         if area >= min_size_px:
#             components.append((lab, area))

#     if not components:
#         return np.zeros_like(mask01)

#     if keep_only_largest:
#         lab_keep = max(components, key=lambda x: x[1])[0]
#         kept[labels == lab_keep] = 1
#     else:
#         for lab_keep, _ in components:
#             kept[labels == lab_keep] = 1

#     return kept.astype(np.uint8)

# def measure_metrics_from_mask(mask01, um_per_px):
#     mask01 = (mask01 > 0).astype(np.uint8)
#     area_px = float(mask01.sum())

#     if area_px <= 0:
#         return {
#             "area_px": 0.0,
#             "perimeter_px": 0.0,
#             "diameter_px": 0.0,
#             "circularity": np.nan,
#             "area_report": 0.0,
#             "perimeter_report": 0.0,
#             "diameter_report": 0.0,
#         }

#     contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     perimeter_px = float(sum(cv2.arcLength(cnt, closed=True) for cnt in contours)) if len(contours) else 0.0

#     diameter_px = float(math.sqrt(4.0 * area_px / math.pi))
#     circularity = float(4.0 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else np.nan

#     area_um2 = area_px * (um_per_px ** 2)
#     perimeter_um = perimeter_px * um_per_px
#     diameter_um = diameter_px * um_per_px

#     if REPORT_UNIT == "um":
#         area_report = area_um2
#         perimeter_report = perimeter_um
#         diameter_report = diameter_um
#     else:
#         area_report = area_um2 / 1_000_000.0
#         perimeter_report = perimeter_um / 1000.0
#         diameter_report = diameter_um / 1000.0

#     return {
#         "area_px": area_px,
#         "perimeter_px": perimeter_px,
#         "diameter_px": diameter_px,
#         "circularity": circularity,
#         "area_report": area_report,
#         "perimeter_report": perimeter_report,
#         "diameter_report": diameter_report,
#     }

# # ------------------------------------------------------------
# # LOAD MODEL
# # ------------------------------------------------------------
# best_fold_name, best_fold_score, best_ckpt_path = find_best_fold_checkpoint(RUN_ROOT)

# print(f"Using checkpoint from {best_fold_name}")
# print(f"Best validation Dice found: {best_fold_score:.4f}")
# print(f"Checkpoint: {best_ckpt_path}")

# model = UNetPlusPlus(in_ch=3, out_ch=1, base=BASE_CHANNELS).to(DEVICE)
# state = torch.load(best_ckpt_path, map_location=DEVICE)
# model.load_state_dict(state)
# model.eval()

# # ------------------------------------------------------------
# # RUN OVER LABS DATA
# # ------------------------------------------------------------
# rows = []

# for _, row in tqdm(labs_df.iterrows(), total=len(labs_df), desc="Predict + export labs"):
#     image_path = str(row["image_path"])

#     try:
#         meta = build_labs_metadata(image_path)

#         img_rgb = load_rgb_image_pil(image_path)
#         pred_mask = predict_mask(
#             model,
#             img_rgb,
#             threshold=THRESHOLD,
#             size=IMG_SIZE_UNET_INFER
#         )

#         pred_mask = clean_small_components(
#             pred_mask,
#             min_size_px=MIN_COMPONENT_SIZE_PX,
#             keep_only_largest=KEEP_ONLY_LARGEST_COMPONENT
#         )

#         um_per_px = get_um_per_px_for_lab(meta["_lab"])
#         meas = measure_metrics_from_mask(pred_mask, um_per_px)

#         rows.append({
#             "Organoid ID": meta["Organoid ID"],
#             "Sample Type": meta["Sample Type"],
#             "Substrate": meta["Substrate"],
#             "Day": meta["Day"],
#             "Batch": meta["Batch"],
#             "File": meta["File"],
#             "Label": False,
#             "Area (px²)": meas["area_px"],
#             AREA_COL: meas["area_report"],
#             "Perimeter (px)": meas["perimeter_px"],
#             PERIM_COL: meas["perimeter_report"],
#             "Diameter (px)": meas["diameter_px"],
#             DIAM_COL: meas["diameter_report"],
#             "Circularity": meas["circularity"],
#             "Scale µm/px": um_per_px,
#         })

#     except Exception as e:
#         print(f"[WARN] Failed on {image_path}: {repr(e)}")
#         rows.append({
#             "Organoid ID": None,
#             "Sample Type": None,
#             "Substrate": "TC",
#             "Day": None,
#             "Batch": "",
#             "File": os.path.basename(image_path),
#             "Label": False,
#             "Area (px²)": None,
#             AREA_COL: None,
#             "Perimeter (px)": None,
#             PERIM_COL: None,
#             "Diameter (px)": None,
#             DIAM_COL: None,
#             "Circularity": None,
#             "Scale µm/px": None,
#         })

# results_df = pd.DataFrame(rows)

# batch_order = {"LabA": 0, "LabB": 1}
# results_df["_batch_sort"] = results_df["Batch"].map(batch_order).fillna(999)
# results_df["_day_sort"] = pd.to_numeric(results_df["Day"], errors="coerce").fillna(999999)
# results_df["File"] = results_df["File"].fillna("")

# results_df = (
#     results_df
#     .sort_values(by=["_batch_sort", "_day_sort", "File"], na_position="last")
#     .drop(columns=["_batch_sort", "_day_sort"])
#     .reset_index(drop=True)
# )

# results_df = results_df[
#     [
#         "Organoid ID",
#         # "Sample Type",
#         # "Substrate",
#         "Day",
#         "Batch",
#         "File",
#         "Label",
#         # "Area (px²)",
#         AREA_COL,
#         # "Perimeter (px)",
#         PERIM_COL,
#         # "Diameter (px)",
#         DIAM_COL,
#         "Circularity",
#         "Scale µm/px",
#     ]
# ]

# OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
# with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
#     results_df.to_excel(writer, sheet_name="Results", index=False)

# print(f"Saved Excel: {OUTPUT_XLSX}")
# display(results_df.head(30))

# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# # ============================================================
# # Plot labs-only Excel (NO SUBSTRATE, CUSTOM COLORS)
# #
# # Rows:
# #   LabA, LabB
# #
# # Columns:
# #   Area, Perimeter, Diameter, Circularity
# #
# # Edit colors in BATCH_COLORS below.
# # ============================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# XLSX_PATH = OUTPUT_XLSX

# BATCH_ORDER = ["LabA", "LabB"]

# # ------------------------------------------------------------
# # GLOBAL FONT SETTINGS (EDIT HERE)
# # ------------------------------------------------------------
# plt.rcParams.update({
#     "font.size": 14,          # base size (ticks, legend, etc.)
#     "font.weight": "bold",    # global weight

#     "axes.titlesize": 16,
#     "axes.titleweight": "bold",

#     "axes.labelsize": 15,
#     "axes.labelweight": "bold",

#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,

#     "legend.fontsize": 12,
# })

# # ------------------------------------------------------------
# # COLORS (EDIT HERE)
# # ------------------------------------------------------------
# BATCH_COLORS = {
#     "LabA": "tab:green",
#     "LabB": "tab:red",
# }

# if REPORT_UNIT == "um":
#     AREA_COL = "Area (µm²)"
#     PERIM_COL = "Perimeter (µm)"
#     DIAM_COL = "Diameter (µm)"
# else:
#     AREA_COL = "Area (mm²)"
#     PERIM_COL = "Perimeter (mm)"
#     DIAM_COL = "Diameter (mm)"

# CIRC_COL = "Circularity"

# def sem(x: pd.Series) -> float:
#     x = x.dropna()
#     n = len(x)
#     if n <= 1:
#         return np.nan
#     return float(x.std(ddof=1) / np.sqrt(n))

# # ------------------------------------------------------------
# # LOAD + CLEAN
# # ------------------------------------------------------------
# dfp = pd.read_excel(XLSX_PATH, sheet_name="Results")

# dfp["Day_num"] = pd.to_numeric(dfp["Day"], errors="coerce")

# for c in [AREA_COL, PERIM_COL, DIAM_COL, CIRC_COL]:
#     dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

# dfp["Batch"] = dfp["Batch"].astype(str).str.strip()

# dfp = dfp.dropna(subset=["Day_num"])
# dfp = dfp[dfp["Day_num"] >= 0]
# dfp = dfp[dfp["Batch"].isin(BATCH_ORDER)]

# # ------------------------------------------------------------
# # AGGREGATE
# # ------------------------------------------------------------
# agg = (
#     dfp.groupby(["Batch", "Day_num"], as_index=False)
#        .agg(
#            Area_mean=(AREA_COL, "mean"), Area_sem=(AREA_COL, sem),
#            Per_mean=(PERIM_COL, "mean"), Per_sem=(PERIM_COL, sem),
#            Dia_mean=(DIAM_COL, "mean"), Dia_sem=(DIAM_COL, sem),
#            Cir_mean=(CIRC_COL, "mean"), Cir_sem=(CIRC_COL, sem),
#        )
# )

# # ------------------------------------------------------------
# # PLOT
# # ------------------------------------------------------------
# fig, axes = plt.subplots(
#     2, 4,
#     figsize=(20, 8.5),
#     constrained_layout=True,
#     squeeze=False
# )

# plot_specs = [
#     ("Area",        "Area_mean", "Area_sem", AREA_COL),
#     ("Perimeter",   "Per_mean",  "Per_sem",  PERIM_COL),
#     ("Diameter",    "Dia_mean",  "Dia_sem",  DIAM_COL),
#     ("Circularity", "Cir_mean",  "Cir_sem",  CIRC_COL),
# ]

# for r, batch_name in enumerate(BATCH_ORDER):
#     df_batch = agg[agg["Batch"] == batch_name].copy()
#     raw_batch = dfp[dfp["Batch"] == batch_name].copy()

#     if raw_batch.empty:
#         highest_day = 0
#         max_day = 10
#         xticks_present = [0]
#     else:
#         highest_day = int(np.nanmax(raw_batch["Day_num"].to_numpy()))
#         max_day = highest_day + 2
#         xticks_present = sorted(raw_batch["Day_num"].dropna().astype(int).unique().tolist())

#     for c, (title, ymean, ysem, ylabel) in enumerate(plot_specs):
#         ax = axes[r, c]

#         dsub = df_batch.sort_values("Day_num")

#         ax.errorbar(
#             dsub["Day_num"],
#             dsub[ymean],
#             yerr=dsub[ysem],
#             marker="o",
#             capsize=3,
#             linewidth=2,
#             color=BATCH_COLORS.get(batch_name, None),
#             label=batch_name,
#         )

#         ax.set_xlim(0, max_day)
#         ax.set_xticks(xticks_present if len(xticks_present) else [0])

#         if c == 0:
#             ax.set_ylabel(f"{batch_name}\n{ylabel}")
#         else:
#             ax.set_ylabel(ylabel)

#         if r == 0:
#             ax.set_title(title)

#         ax.set_xlabel("Day")
#         ax.grid(True, alpha=0.25)
#         # ax.legend(loc="best")

# plt.show()