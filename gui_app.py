"""Brain organoid brightfield image analysis GUI.

This application provides batch image loading, segmentation, calibrated
morphometric measurement, visualization, manual refinement, optional SAM
refinement, and structured result export.

Segmentation methods:
    - Handcrafted physics-based image processing
    - FastUNet++ inference using a trained checkpoint

Model files are expected in the ``models`` directory when running from
source. PyInstaller-bundled resources are also supported.
"""

import os
import math
import threading
import queue
import re
import sqlite3
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import time
import sys

from unetpp import UNetPlusPlus

# ============================================================
# OPTIONAL TORCH / U-NET++
# ============================================================
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as e:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# U-Net++ CONFIG
# ------------------------------------------------------------
UNETPP_MODULE_NAME = "unetpp"
UNETPP_CLASS_NAME = "UNetPlusPlus"

def resource_path(relative_path: str) -> Path:
    """
    Get the correct path for files bundled with PyInstaller.

    Normal Python:
        Uses the folder containing this .py file.

    PyInstaller --onefile:
        Uses the temporary extraction directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent

    return base_path / relative_path


BEST_MODEL_PATH = str(
    resource_path("models/best_unetpp_512x384.pt")
)

TARGET_WIDTH = 512
ASPECT_RATIO = 4/3  # width / height. Examples: 1, 4/3, 16/9
PAD_TO_MULTIPLE = 16

UNETPP_THRESHOLD = 0.5
UNETPP_BASE_CHANNELS = 32
MIN_COMPONENT_SIZE_PX = 50
KEEP_ONLY_LARGEST_COMPONENT = True

# Controls whether U-Net++ results are displayed at model resolution or back-projected.
RENDER_RESIZED = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

UNIT_TO_UM = {"um": 1.0, "mm": 1000.0, "cm": 10000.0, "m": 1_000_000.0}

_OCR_UNIT_ALIASES = {
    "µm": "um",
    "μm": "um",
    "um": "um",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
}


def compute_unet_geometry(
    target_width: int = TARGET_WIDTH,
    aspect_ratio: float = ASPECT_RATIO,
    pad_to_multiple: int = PAD_TO_MULTIPLE,
) -> Dict[str, int]:
    target_w = int(round(target_width))
    target_h = int(round(target_w / float(aspect_ratio)))

    padded_w = int(math.ceil(target_w / pad_to_multiple) * pad_to_multiple)
    padded_h = int(math.ceil(target_h / pad_to_multiple) * pad_to_multiple)

    return {
        "target_w": target_w,
        "target_h": target_h,
        "padded_w": padded_w,
        "padded_h": padded_h,
    }


UNET_GEOM = compute_unet_geometry()


def unit_labels(report_unit: str) -> Dict[str, str]:
    u = report_unit.strip()
    return {"area": f"{u}²", "len": f"{u}"}


def convert_um_per_px_to_unit_per_px(um_per_px: float, report_unit: str) -> float:
    u = report_unit.lower().strip()
    if u not in UNIT_TO_UM:
        raise ValueError(f"Unit must be one of {list(UNIT_TO_UM.keys())}")
    return float(um_per_px) / UNIT_TO_UM[u]


def read_image_any(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3].copy()
    return img[:, :, :3].copy()


def bgr_to_rgb3(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_bgr3(img), cv2.COLOR_BGR2RGB)


# ============================================================
# FOV DETECTION + FOCUS MASK
# ============================================================
def detect_fov_circle(gray_or_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int]], str]:
    gray = to_gray(gray_or_bgr)
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g -= np.min(g)
        mx = np.max(g)
        if mx > 0:
            g = (g / mx) * 255.0
        gray = g.astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "FOV auto: no contours"

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 1000:
        return None, "FOV auto: contour too small"

    (cx, cy), r = cv2.minEnclosingCircle(c)
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))

    h, w = gray.shape[:2]
    if r < int(min(h, w) * 0.25):
        return None, f"FOV auto: radius too small ({r}px)"
    if not (0 <= cx < w and 0 <= cy < h):
        return None, "FOV auto: center out of bounds"

    return (cx, cy, r), f"FOV auto: (cx={cx}, cy={cy}, r={r}px)"


def make_fov_focus_mask(h: int, w: int, fov: Tuple[int, int, int], inner_margin_px: int) -> np.ndarray:
    cx, cy, r = fov
    rr = max(1, int(r - max(0, inner_margin_px)))
    keep = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(keep, (int(cx), int(cy)), int(rr), 1, thickness=-1)
    return keep


# ============================================================
# SEGMENTATION + METRICS
# ============================================================
def largest_contour_from_mask(mask01: np.ndarray) -> Optional[np.ndarray]:
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def feret_max_diameter_from_contour(cnt: np.ndarray) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    if cnt is None or len(cnt) < 2:
        return 0.0, (0, 0), (0, 0)
    hull = cv2.convexHull(cnt, returnPoints=True)
    pts = hull.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 2:
        p = tuple(pts[0].astype(int))
        return 0.0, p, p
    diffs = pts[None, :, :] - pts[:, None, :]
    d2 = np.sum(diffs * diffs, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    p1 = tuple(pts[i].astype(int))
    p2 = tuple(pts[j].astype(int))
    return float(np.sqrt(d2[i, j])), p1, p2


def measure_metrics_from_mask(mask01: np.ndarray, um_per_px: float, report_unit: str) -> Dict[str, Any]:
    cnt = largest_contour_from_mask(mask01)
    if cnt is None:
        return {
            "Area": np.nan,
            "Perimeter": np.nan,
            "Diameter": np.nan,
            "Circularity": np.nan,
            "contour": None,
            "feret_p1": (0, 0),
            "feret_p2": (0, 0),
            "area_px2": np.nan,
            "perim_px": np.nan,
            "feret_px": np.nan,
        }

    area_px2 = float(cv2.contourArea(cnt))
    perim_px = float(cv2.arcLength(cnt, closed=True))
    feret_px, p1, p2 = feret_max_diameter_from_contour(cnt)

    unit_per_px = convert_um_per_px_to_unit_per_px(um_per_px, report_unit)
    area = area_px2 * (unit_per_px ** 2)
    perimeter = perim_px * unit_per_px
    diameter = feret_px * unit_per_px

    if perim_px > 0:
        circ = 4.0 * math.pi * area_px2 / (perim_px ** 2)
    else:
        circ = np.nan
    if np.isfinite(circ):
        circ = float(np.clip(circ, 0.0, 1.0))

    return {
        "Area": float(area),
        "Perimeter": float(perimeter),
        "Diameter": float(diameter),
        "Circularity": float(circ) if np.isfinite(circ) else np.nan,
        "contour": cnt,
        "feret_p1": p1,
        "feret_p2": p2,
        "area_px2": area_px2,
        "perim_px": perim_px,
        "feret_px": feret_px,
    }


def visual_geometry_from_mask(mask01: Optional[np.ndarray]) -> Dict[str, Any]:
    if mask01 is None:
        return {"contour": None, "feret_p1": (0, 0), "feret_p2": (0, 0)}
    cnt = largest_contour_from_mask(mask01)
    if cnt is None:
        return {"contour": None, "feret_p1": (0, 0), "feret_p2": (0, 0)}
    _, p1, p2 = feret_max_diameter_from_contour(cnt)
    return {"contour": cnt, "feret_p1": p1, "feret_p2": p2}


def segment_organoid_mask_physics(
    img_bgr_or_gray: np.ndarray,
    min_area_px2: int,
    max_area_px2: Optional[int],
    use_fov: bool,
    fov_circle: Optional[Tuple[int, int, int]],
    fov_inner_margin_px: int,
) -> Tuple[Optional[np.ndarray], str]:
    gray = to_gray(img_bgr_or_gray)
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g -= np.min(g)
        mx = np.max(g)
        if mx > 0:
            g = (g / mx) * 255.0
        gray = g.astype(np.uint8)

    h, w = gray.shape[:2]

    focus = None
    cx = cy = r_focus = None
    if use_fov and fov_circle is not None:
        cx, cy, r = fov_circle
        r_focus = int(max(1, r - max(0, int(fov_inner_margin_px))))
        focus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(focus, (int(cx), int(cy)), int(r_focus), 1, thickness=-1)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(thr > 0)) > 0.6:
        thr = cv2.bitwise_not(thr)

    if focus is not None:
        thr = ((thr > 0).astype(np.uint8) & focus) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)

    mask01 = (thr > 0).astype(np.uint8)
    if int(mask01.sum()) < 10:
        return None, "No foreground after thresholding"

    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num <= 1:
        return None, "No components found"

    candidates: List[Tuple[int, int]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_px2):
            continue
        if max_area_px2 is not None and area > int(max_area_px2):
            continue

        if focus is not None and cx is not None and cy is not None and r_focus is not None:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])

            corners = np.array(
                [
                    [x, y],
                    [x + ww, y],
                    [x, y + hh],
                    [x + ww, y + hh],
                ],
                dtype=np.float32,
            )
            dx = corners[:, 0] - float(cx)
            dy = corners[:, 1] - float(cy)
            d = np.sqrt(dx * dx + dy * dy)

            if np.any(d >= float(r_focus - 4)):
                continue

        candidates.append((area, i))

    if not candidates:
        return None, "No component passed filters"

    candidates.sort(reverse=True, key=lambda x: x[0])
    chosen_area, chosen_label = candidates[0]

    out = np.zeros_like(mask01, dtype=np.uint8)
    out[lab == chosen_label] = 1

    if len(candidates) > 1:
        return out, f"Multiple components ({len(candidates)}). Using largest valid (area_px2={chosen_area})"
    return out, "OK"


def polygon_to_mask01(h: int, w: int, pts: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    if pts is None or len(pts) < 3:
        return None
    poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def overlay_visual(
    img: np.ndarray,
    mask01: Optional[np.ndarray],
    cnt: Optional[np.ndarray],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    poly_pts: Optional[List[Tuple[int, int]]] = None,
    fov_circle: Optional[Tuple[int, int, int]] = None,
    fov_inner_margin_px: int = 0,
) -> np.ndarray:
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = img[:, :, :3].copy()
        else:
            bgr = img[:, :, :3].copy()
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    h, w = bgr.shape[:2]
    out = bgr.astype(np.float32)

    if mask01 is not None:
        if mask01.shape[:2] != (h, w):
            mask01 = cv2.resize(mask01, (w, h), interpolation=cv2.INTER_NEAREST)
        m = mask01.astype(bool)
        shade = np.zeros_like(out)
        shade[:, :, 2] = 255.0
        alpha = 0.25
        out[m] = (1 - alpha) * out[m] + alpha * shade[m]

    out_bgr = np.clip(out, 0, 255).astype(np.uint8)

    if cnt is not None:
        cv2.drawContours(out_bgr, [cnt], -1, (0, 255, 0), 2)
        cv2.line(out_bgr, p1, p2, (255, 255, 0), 2)
        cv2.circle(out_bgr, p1, 4, (255, 255, 0), -1)
        cv2.circle(out_bgr, p2, 4, (255, 255, 0), -1)

    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if poly_pts is not None and len(poly_pts) >= 2:
        for i in range(1, len(poly_pts)):
            cv2.line(out_bgr, poly_pts[i - 1], poly_pts[i], (255, 0, 255), 2)
        for p in poly_pts:
            cv2.circle(out_bgr, p, 3, (255, 0, 255), -1)

    if fov_circle is not None:
        cx, cy, r = fov_circle
        rr = max(1, int(r - max(0, fov_inner_margin_px)))
        cv2.circle(out_bgr, (int(cx), int(cy)), int(rr), (255, 255, 255), 2)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# AUTO SCALE DETECTION + OPTIONAL OCR
# ============================================================
def _safe_import_pytesseract():
    try:
        import pytesseract
        return pytesseract, None
    except Exception as e:
        return None, str(e)


PYTESSERACT, PYTESSERACT_IMPORT_ERROR = _safe_import_pytesseract()


def try_detect_scale_bar_px_and_roi(gray_or_bgr: np.ndarray) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]], str]:
    gray = to_gray(gray_or_bgr)
    h, w = gray.shape[:2]

    crop_y0 = int(h * 0.60)
    crop_x0 = int(w * 0.45)
    crop = gray[crop_y0:, crop_x0:]

    _, mask = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, "Auto scale: no candidates"

    best_width = None
    best_rect = None
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh <= 0:
            continue
        aspect = ww / hh
        if aspect < 5.0:
            continue
        if ww < int(0.05 * w):
            continue
        if best_width is None or ww > best_width:
            best_width = ww
            best_rect = (x, y, ww, hh)

    if best_width is None or best_rect is None:
        return None, None, "Auto scale: no bar-like rectangle found"

    length_px = int(best_width)

    bx, by, bww, bhh = best_rect
    full_x1 = crop_x0 + bx
    full_y1 = crop_y0 + by
    full_x2 = crop_x0 + bx + bww
    full_y2 = crop_y0 + by + bhh

    pad_x = int(0.08 * w)
    pad_y_up = int(0.12 * h)
    pad_y_down = int(0.08 * h)

    x1 = max(0, full_x1 - pad_x)
    x2 = min(w, full_x2 + pad_x)
    y1 = max(0, full_y1 - pad_y_up)
    y2 = min(h, full_y2 + pad_y_down)

    ocr_roi = (x1, y1, x2, y2)
    return length_px, ocr_roi, f"Auto scale: detected ~{length_px}px"


def _normalize_ocr_text(s: str) -> str:
    s = s.replace("\n", " ").replace("\t", " ")
    s = s.replace("μ", "µ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def try_ocr_scale_value_and_unit(gray_or_bgr: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> Tuple[Optional[float], Optional[str], str]:
    if PYTESSERACT is None:
        return None, None, f"OCR unavailable: {PYTESSERACT_IMPORT_ERROR}"

    gray = to_gray(gray_or_bgr)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = roi_xyxy
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(1, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(1, min(h, y2)))

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None, None, "OCR ROI too small"

    crop = gray[y1:y2, x1:x2].copy()
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

    scale = 2.5
    crop = cv2.resize(
        crop,
        (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

    _, thr = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r'--psm 6 -c tessedit_char_whitelist="0123456789.umµμMmCc "'
    try:
        text = PYTESSERACT.image_to_string(thr, config=config)
    except Exception as e:
        return None, None, f"OCR error: {e}"

    text = _normalize_ocr_text(text)

    m = re.search(r"(\d+(?:\.\d+)?)\s*([µμ]m|um|mm|cm|m)\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)([µμ]m|um|mm|cm|m)\b", text, flags=re.IGNORECASE)

    if not m:
        return None, None, f"OCR did not find '<number><unit>' in: {text}"

    val = float(m.group(1))
    unit_raw = m.group(2)
    unit_raw = unit_raw.replace("μ", "µ").lower()
    unit_key = _OCR_UNIT_ALIASES.get(unit_raw)
    if unit_key is None:
        return None, None, f"OCR unit not recognized: {unit_raw}"

    return val, unit_key, f"OCR OK: {val} {unit_raw}"


# ============================================================
# OPTIONAL SAM
# ============================================================
def _try_import_sam():
    try:
        from ultralytics import SAM
        return SAM, None
    except Exception as e:
        return None, str(e)


SAM_CLASS, SAM_IMPORT_ERROR = _try_import_sam()


def sam_predict_mask01_from_bbox(sam_model, image_bgr: np.ndarray, bbox_xyxy: List[int]) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    if sam_model is None:
        return None, None, "SAM model not loaded"

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = image_bgr[:, :, :3]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = sam_model(rgb, bboxes=[bbox_xyxy])
        if not results or results[0].masks is None:
            return None, None, "SAM returned no masks"

        masks = results[0].masks.data
        if masks is None:
            return None, None, "SAM masks missing"

        masks_np = masks.cpu().numpy()
        if masks_np.ndim != 3 or masks_np.shape[0] == 0:
            return None, None, "SAM produced empty mask set"

        cand = [(m > 0.5).astype(np.uint8) for m in masks_np]
        areas = [int(m.sum()) for m in cand]
        k = int(np.argmax(areas))
        mask01 = cand[k]

        score = None
        try:
            conf = results[0].boxes.conf.cpu().numpy()
            score = float(conf[k]) if k < len(conf) else float(conf[0])
        except Exception:
            score = None

        return mask01, score, "OK"
    except Exception as e:
        return None, None, f"SAM error: {e}"


# ============================================================
# U-NET++ HELPERS
# ============================================================
def safe_torch_device():
    if not TORCH_AVAILABLE:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_unetpp_class():
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")

    module = importlib.import_module(UNETPP_MODULE_NAME)
    if not hasattr(module, UNETPP_CLASS_NAME):
        raise AttributeError(f"Module '{UNETPP_MODULE_NAME}' has no class '{UNETPP_CLASS_NAME}'")
    return getattr(module, UNETPP_CLASS_NAME)


def resize_image_for_model(img_rgb_or_bgr: np.ndarray) -> np.ndarray:
    target_w = UNET_GEOM["target_w"]
    target_h = UNET_GEOM["target_h"]
    return cv2.resize(img_rgb_or_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def pad_to_multiple(img: np.ndarray, multiple: int = PAD_TO_MULTIPLE):
    h, w = img.shape[:2]
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)

    pad_h = new_h - h
    pad_w = new_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_pad = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    pad_info = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
    }
    return img_pad, pad_info


def unpad_mask(mask: np.ndarray, pad_info: Dict[str, int]) -> np.ndarray:
    t = pad_info["top"]
    b = pad_info["bottom"]
    l = pad_info["left"]
    r = pad_info["right"]

    if b == 0:
        h_slice = slice(t, None)
    else:
        h_slice = slice(t, -b)

    if r == 0:
        w_slice = slice(l, None)
    else:
        w_slice = slice(l, -r)

    return mask[h_slice, w_slice]


def back_project_mask_to_original(mask_resized: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    mask_orig = cv2.resize(
        mask_resized.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (mask_orig > 0).astype(np.uint8)


def preprocess_backproject_mode(img_rgb: np.ndarray):
    orig_h, orig_w = img_rgb.shape[:2]

    img_r = resize_image_for_model(img_rgb)
    resized_h, resized_w = img_r.shape[:2]

    img_pad, pad_info = pad_to_multiple(img_r, multiple=PAD_TO_MULTIPLE)

    x = img_pad.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_info": pad_info,
    }
    return x, meta


def build_unetpp_render_meta_from_original(img: np.ndarray) -> Dict[str, Any]:
    orig_h, orig_w = img.shape[:2]
    render_w = UNET_GEOM["target_w"]
    render_h = UNET_GEOM["target_h"]
    sx = float(render_w) / max(float(orig_w), 1.0)
    sy = float(render_h) / max(float(orig_h), 1.0)
    return {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "render_h": render_h,
        "render_w": render_w,
        "sx": sx,
        "sy": sy,
    }


def predict_mask_unetpp(
    model,
    device,
    img_rgb: np.ndarray,
    threshold: float = UNETPP_THRESHOLD,
) -> Dict[str, np.ndarray]:
    x, meta = preprocess_backproject_mode(img_rgb)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs_pad = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred_pad = (probs_pad > threshold).astype(np.uint8)
    pred_resized = unpad_mask(pred_pad, meta["pad_info"])
    pred_orig = back_project_mask_to_original(
        pred_resized,
        meta["orig_h"],
        meta["orig_w"],
    )
    return {
        "pred_orig": pred_orig,
        "pred_resized": pred_resized,
    }


def clean_small_components(mask01: np.ndarray, min_size_px: int = 50, keep_only_largest: bool = True) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)

    if num_labels <= 1:
        return mask01

    kept = np.zeros_like(mask01)
    components = []

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_size_px:
            components.append((lab, area))

    if not components:
        return np.zeros_like(mask01)

    if keep_only_largest:
        lab_keep = max(components, key=lambda x: x[1])[0]
        kept[labels == lab_keep] = 1
    else:
        for lab_keep, _ in components:
            kept[labels == lab_keep] = 1

    return kept.astype(np.uint8)


def apply_focus_mask_if_needed(mask01: np.ndarray, focus01: Optional[np.ndarray]) -> np.ndarray:
    if focus01 is None:
        return (mask01 > 0).astype(np.uint8)
    if focus01.shape[:2] != mask01.shape[:2]:
        focus01 = cv2.resize(
            focus01.astype(np.uint8),
            (mask01.shape[1], mask01.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return ((mask01 > 0).astype(np.uint8) & (focus01 > 0).astype(np.uint8)).astype(np.uint8)


def filter_mask_by_area(mask01: np.ndarray, min_area_px2: int, max_area_px2: Optional[int]) -> Tuple[Optional[np.ndarray], str]:
    area = int((mask01 > 0).sum())
    if area < int(min_area_px2):
        return None, "Predicted mask below min area filter"
    if max_area_px2 is not None and area > int(max_area_px2):
        return None, "Predicted mask above max area filter"
    if area <= 0:
        return None, "Predicted mask empty"
    return (mask01 > 0).astype(np.uint8), "OK"


# ============================================================
# EXPORT FRAMEWORK
# ============================================================
class ExportError(Exception):
    pass


class Exporter:
    key: str = "base"
    label: str = "Base"
    default_ext: str = ".dat"
    filetypes: List[Tuple[str, str]] = [("All files", "*.*")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        raise NotImplementedError


class ExcelExporter(Exporter):
    key = "excel"
    label = "Excel (.xlsx)"
    default_ext = ".xlsx"
    filetypes = [("Excel", "*.xlsx")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")
                meta_df = pd.DataFrame([meta])
                meta_df.to_excel(writer, index=False, sheet_name="Meta")
        except Exception as e:
            raise ExportError(f"Excel export failed: {e}") from e


class CSVExporter(Exporter):
    key = "csv"
    label = "CSV (.csv)"
    default_ext = ".csv"
    filetypes = [("CSV", "*.csv")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            df.to_csv(out_path, index=False)
            meta_path = os.path.splitext(out_path)[0] + "_meta.json"
            pd.DataFrame([meta]).to_json(meta_path, orient="records", indent=2)
        except Exception as e:
            raise ExportError(f"CSV export failed: {e}") from e


class JSONExporter(Exporter):
    key = "json"
    label = "JSON (.json)"
    default_ext = ".json"
    filetypes = [("JSON", "*.json")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            payload = {"meta": meta, "rows": df.to_dict(orient="records")}
            with open(out_path, "w", encoding="utf-8") as f:
                import json
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ExportError(f"JSON export failed: {e}") from e


class NumpyExporter(Exporter):
    key = "numpy"
    label = "NumPy (.npz)"
    default_ext = ".npz"
    filetypes = [("NumPy NPZ", "*.npz")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            rec = df.to_records(index=False)
            cols = np.array(df.columns.tolist(), dtype=object)
            import json
            meta_json = json.dumps(meta).encode("utf-8")
            np.savez_compressed(out_path, records=rec, columns=cols, meta=meta_json)
        except Exception as e:
            raise ExportError(f"NumPy export failed: {e}") from e


class SQLiteExporter(Exporter):
    key = "sqlite"
    label = "SQLite (.sqlite)"
    default_ext = ".sqlite"
    filetypes = [("SQLite DB", "*.sqlite;*.db"), ("SQLite DB", "*.db")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            conn = sqlite3.connect(out_path)
            try:
                df.to_sql("results", conn, if_exists="replace", index=False)
                meta_df = pd.DataFrame([meta])
                meta_df.to_sql("meta", conn, if_exists="replace", index=False)
            finally:
                conn.close()
        except Exception as e:
            raise ExportError(f"SQLite export failed: {e}") from e


class HDF5Exporter(Exporter):
    key = "hdf5"
    label = "HDF5 (.h5)"
    default_ext = ".h5"
    filetypes = [("HDF5", "*.h5;*.hdf5"), ("HDF5", "*.hdf5")]

    def export(self, out_path: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        try:
            import h5py
            import json
            with h5py.File(out_path, "w") as f:
                rec = df.to_records(index=False)
                f.create_dataset("results", data=rec, compression="gzip", compression_opts=4)
                f.create_dataset("columns", data=np.array(df.columns.tolist(), dtype="S"))
                f.attrs["meta_json"] = json.dumps(meta)
        except Exception as e:
            raise ExportError(f"HDF5 export failed: {e}") from e


EXPORTERS: Dict[str, Exporter] = {
    ExcelExporter.key: ExcelExporter(),
    CSVExporter.key: CSVExporter(),
    JSONExporter.key: JSONExporter(),
    NumpyExporter.key: NumpyExporter(),
    SQLiteExporter.key: SQLiteExporter(),
    HDF5Exporter.key: HDF5Exporter(),
}


def build_results_dataframe(image_paths: List[str], results_by_path: Dict[str, "AnalysisResult"]) -> pd.DataFrame:
    rows = []
    for p in image_paths:
        r = results_by_path.get(p)
        if r is None:
            continue
        rows.append(
            {
                "File": r.file,
                "Path": r.path,
                "OK": bool(r.ok),
                "Message": r.message,
                "Method": r.method,
                f"Area ({r.report_unit})²": r.area,
                f"Perimeter ({r.report_unit})": r.perimeter,
                f"Diameter ({r.report_unit})": r.diameter,
                "Circularity": r.circularity,
                "Scale_um_per_px": r.um_per_px,
                "Refined": bool(r.refined),
                "RefinedMode": r.refined_mode,
                "MinAreaUnit2": r.min_area_unit2,
                "MaxAreaUnit2": r.max_area_unit2,
            }
        )
    return pd.DataFrame(rows)


def build_export_meta(app: "OrganoidBatchGUI") -> Dict[str, Any]:
    return {
        "report_unit": app.report_unit.get(),
        "um_per_px": float(app.um_per_px.get()),
        "scale_known_len": float(app.scale_known_len.get()),
        "scale_unit": app.scale_unit.get(),
        "use_fov": bool(app.use_fov.get()),
        "fov_inner_margin_px": int(app.fov_inner_margin_px.get()),
        "analysis_method": app.analysis_method.get(),
        "best_model_path": BEST_MODEL_PATH,
        "unetpp_target_width": TARGET_WIDTH,
        "unetpp_aspect_ratio": ASPECT_RATIO,
        "unetpp_pad_to_multiple": PAD_TO_MULTIPLE,
        "unetpp_target_height_before_pad": UNET_GEOM["target_h"],
        "unetpp_padded_width": UNET_GEOM["padded_w"],
        "unetpp_padded_height": UNET_GEOM["padded_h"],
        "render_resized": bool(RENDER_RESIZED),
        "min_area_unit2": float(app.min_area_unit2.get()),
        "max_area_unit2": app.max_area_unit2.get(),
        "image_count": int(len(app.image_paths)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def ask_save_path_for_export(master, exporter: Exporter) -> Optional[str]:
    return filedialog.asksaveasfilename(
        defaultextension=exporter.default_ext,
        filetypes=exporter.filetypes,
    )


# ============================================================
# GUI STATE
# ============================================================
@dataclass
class AnalysisResult:
    file: str
    path: str
    ok: bool
    message: str
    method: str
    um_per_px: float
    report_unit: str
    min_area_unit2: float
    max_area_unit2: Optional[float]
    area: Optional[float]
    perimeter: Optional[float]
    diameter: Optional[float]
    circularity: Optional[float]
    refined: bool = False
    refined_mode: str = "auto"


# ============================================================
# SCROLLABLE FRAME
# ============================================================
class ScrollFrame(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.inner.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)

    def _on_configure(self, _evt=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, evt):
        self.canvas.itemconfigure(self.window_id, width=evt.width)

    def _bind_wheel(self, _evt=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_wheel(self, _evt=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, evt):
        delta = evt.delta
        if delta == 0:
            return
        self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    def _on_mousewheel_linux(self, evt):
        if evt.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif evt.num == 5:
            self.canvas.yview_scroll(1, "units")


# ============================================================
# MAIN GUI
# ============================================================
class OrganoidBatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyzer")
        self.geometry("1500x860")

        self.dir_path = tk.StringVar(value="")
        self.report_unit = tk.StringVar(value="mm")

        self.um_per_px = tk.DoubleVar(value=1.0)
        self.scale_known_len = tk.DoubleVar(value=1060.0)
        self.scale_unit = tk.StringVar(value="um")
        self.auto_scale_px = tk.StringVar(value="")

        self.export_format = tk.StringVar(value="excel")

        self.min_area_unit2 = tk.DoubleVar(value=0.0)
        self.max_area_unit2 = tk.StringVar(value="")

        self.analysis_method = tk.StringVar(value="physics")   # physics | unetpp

        self.use_fov = tk.BooleanVar(value=False)
        self.fov_inner_margin_px = tk.IntVar(value=60)
        self.fov_status = tk.StringVar(value="FOV: not set")
        self._fov_circle: Optional[Tuple[int, int, int]] = None
        self._fov_manual_click_mode = False
        self._fov_pts: List[Tuple[int, int]] = []

        self.image_paths: List[str] = []
        self.results_by_path: Dict[str, AnalysisResult] = {}
        self.masks_by_path: Dict[str, np.ndarray] = {}
        self.metrics_by_path: Dict[str, Dict[str, Any]] = {}

        # Cached U-Net++ render data for the optional resized display mode.
        self.render_images_by_path: Dict[str, np.ndarray] = {}
        self.render_masks_by_path: Dict[str, np.ndarray] = {}
        self.render_meta_by_path: Dict[str, Dict[str, Any]] = {}

        self._current_img_path: Optional[str] = None
        self._current_img_np: Optional[np.ndarray] = None
        self._display_scale = 1.0
        self._canvas_img_origin = (0, 0)

        # Base image dimensions before fitting the image to the canvas.
        self._current_display_base_w = 1
        self._current_display_base_h = 1

        self._calib_pts: List[Tuple[int, int]] = []
        self._calib_click_mode = False

        self._roi_dragging = False
        self._roi_start_canvas: Optional[Tuple[int, int]] = None
        self._roi_end_canvas: Optional[Tuple[int, int]] = None
        self._roi_xyxy_by_path: Dict[str, Tuple[int, int, int, int]] = {}

        self._draw_mode = tk.BooleanVar(value=False)
        self._poly_pts_by_path: Dict[str, List[Tuple[int, int]]] = {}

        self._COLOR_RED = "#c00000"
        self._COLOR_GREEN = "#0a8f0a"
        self._COLOR_BLUE = "#1f6feb"

        self._q = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._analysis_has_run = False
        self._problems_buffer: List[str] = []

        self.sam_models = self._build_sam_models_dict()
        self.sam_choice = tk.StringVar(value="Pick a SAM Model")
        self.sam_model = None

        self.unetpp_model = None
        self.unetpp_device = safe_torch_device()
        self.unetpp_status = tk.StringVar(value="U-Net++: not loaded")

        self._build_ui()
        self.after(100, self._poll_queue)

    # --------------------------------------------------------
    # SAM MODELS DISCOVERY
    # --------------------------------------------------------
    def _build_sam_models_dict(self) -> dict:
        candidates = {
            "SAM 2 tiny": "sam2_t.pt",
            "SAM 2 small": "sam2_s.pt",
            "SAM 2 base": "sam2_b.pt",
            "SAM 2 large": "sam2_l.pt",
            "SAM 2.1 tiny": "sam2.1_t.pt",
            "SAM 2.1 small": "sam2.1_s.pt",
            "SAM 2.1 base": "sam2.1_b.pt",
            "SAM 2.1 large": "sam2.1_l.pt",
        }

        out = {}

        for display, fname in candidates.items():
            candidate_paths = [
                resource_path(f"models/{fname}"),
                resource_path(fname),
            ]
            for model_path in candidate_paths:
                if model_path.is_file():
                    out[display] = str(model_path)
                    break

        return out

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(self, padding=6)
        right = ttk.Frame(self, padding=6)
        left_outer.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        sf = ScrollFrame(left_outer)
        sf.grid(row=0, column=0, sticky="nsew")
        left = sf.inner
        left.columnconfigure(0, weight=1)

        box = ttk.LabelFrame(left, text="Input", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.dir_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Browse", command=self._browse_dir).grid(row=0, column=2, sticky="ew")
        ttk.Button(box, text="Load", command=self._load_images).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        method_box = ttk.LabelFrame(left, text="Method", padding=8)
        method_box.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        r0 = ttk.Frame(method_box)
        r0.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(r0, text="Handcrafted Physics-Based", variable=self.analysis_method, value="physics").pack(side="left")
        ttk.Radiobutton(r0, text="U-Net++", variable=self.analysis_method, value="unetpp").pack(side="left", padx=10)

        r1 = ttk.Frame(method_box)
        r1.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(r1, text="Load U-Net++", command=self._load_unetpp_model).pack(side="left", fill="x", expand=True)
        ttk.Label(method_box, textvariable=self.unetpp_status, foreground="#444").grid(row=2, column=0, sticky="w", pady=(4, 0))

        scale = ttk.LabelFrame(left, text="Scale", padding=8)
        scale.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        scale.columnconfigure(1, weight=1)

        r0 = ttk.Frame(scale)
        r0.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(r0, text="Report").pack(side="left")
        ttk.Combobox(r0, textvariable=self.report_unit, values=["um", "mm", "cm", "m"], width=6, state="readonly").pack(side="left", padx=6)
        ttk.Label(r0, text="µm/px").pack(side="left", padx=(10, 0))
        ttk.Entry(r0, textvariable=self.um_per_px, width=12).pack(side="left", padx=6)

        r1 = ttk.Frame(scale)
        r1.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(r1, text="Known").pack(side="left")
        ttk.Entry(r1, textvariable=self.scale_known_len, width=10).pack(side="left", padx=6)
        ttk.Combobox(r1, textvariable=self.scale_unit, values=["um", "mm", "cm", "m"], width=6, state="readonly").pack(side="left")

        r2 = ttk.Frame(scale)
        r2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(r2, text="Click-calib", command=self._apply_click_calibration).pack(side="left", fill="x", expand=True)
        ttk.Button(r2, text="Auto scale-bar", command=self._auto_detect_scale).pack(side="left", fill="x", expand=True, padx=6)

        ttk.Label(scale, textvariable=self.auto_scale_px, foreground="#444").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ocr_hint = "OCR: available" if PYTESSERACT is not None else "OCR: unavailable"
        ttk.Label(scale, text=ocr_hint, foreground="#444").grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        fov = ttk.LabelFrame(left, text="FOV focus", padding=8)
        fov.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        fov.columnconfigure(1, weight=1)

        r0 = ttk.Frame(fov)
        r0.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(r0, text="Apply FOV focus", variable=self.use_fov).pack(side="left")

        r1 = ttk.Frame(fov)
        r1.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(r1, text="Inner margin px").pack(side="left")
        ttk.Entry(r1, textvariable=self.fov_inner_margin_px, width=8).pack(side="left", padx=6)

        r2 = ttk.Frame(fov)
        r2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(r2, text="Auto-detect", command=self._auto_detect_fov_current).pack(side="left", fill="x", expand=True)
        ttk.Button(r2, text="Manual set", command=self._start_manual_fov).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Clear", command=self._clear_fov).pack(side="left", fill="x", expand=True)

        ttk.Label(fov, textvariable=self.fov_status, foreground="#444").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        filt = ttk.LabelFrame(left, text="Area filters (report unit²)", padding=8)
        filt.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        filt.columnconfigure(1, weight=1)

        ttk.Label(filt, text="Min (keep ≥)").grid(row=0, column=0, sticky="w")
        ttk.Entry(filt, textvariable=self.min_area_unit2).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(filt, text="Max (discard >)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(filt, textvariable=self.max_area_unit2).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))

        tools = ttk.LabelFrame(left, text="SAM Refinement", padding=8)
        tools.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        tools.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            tools,
            text="Manual draw mode",
            variable=self._draw_mode,
            command=self._on_toggle_draw_mode,
        ).grid(row=0, column=0, sticky="w")

        r = ttk.Frame(tools)
        r.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(r, text="Clear poly", command=self._clear_manual_poly).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="Apply poly", command=self._apply_manual_poly_mask).pack(side="left", fill="x", expand=True, padx=6)

        ttk.Separator(tools).grid(row=2, column=0, sticky="ew", pady=8)

        if SAM_CLASS is None:
            ttk.Label(tools, text=f"SAM unavailable: {SAM_IMPORT_ERROR}", foreground="#aa0000").grid(row=3, column=0, sticky="w")
        else:
            ttk.Label(tools, text="SAM model").grid(row=3, column=0, sticky="w")
            model_values = ["Pick a SAM Model"] + list(self.sam_models.keys())
            ttk.Combobox(tools, textvariable=self.sam_choice, values=model_values, state="readonly").grid(row=4, column=0, sticky="ew", pady=(4, 0))

            r2 = ttk.Frame(tools)
            r2.grid(row=5, column=0, sticky="ew", pady=(6, 0))
            ttk.Button(r2, text="Load SAM", command=self._load_sam_model).pack(side="left", fill="x", expand=True)
            ttk.Button(r2, text="Refine", command=self._refine_selected_with_sam).pack(side="left", fill="x", expand=True, padx=6)

        act = ttk.LabelFrame(left, text="Actions", padding=8)
        act.grid(row=6, column=0, sticky="ew", pady=(0, 6))
        act.columnconfigure(0, weight=1)

        r = ttk.Frame(act)
        r.grid(row=0, column=0, sticky="ew")
        ttk.Button(r, text="Analyze", command=self._start_analysis).pack(side="left", fill="x", expand=True)

        export_frame = ttk.Frame(act)
        export_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        export_frame.columnconfigure(1, weight=1)

        ttk.Label(export_frame, text="Export").grid(row=0, column=0, sticky="w")
        exporter_keys = list(EXPORTERS.keys())
        exporter_labels = [EXPORTERS[k].label for k in exporter_keys]
        self._export_key_by_label = {EXPORTERS[k].label: k for k in exporter_keys}
        self._export_label_by_key = {k: EXPORTERS[k].label for k in exporter_keys}

        self.export_label = tk.StringVar(value=self._export_label_by_key.get(self.export_format.get(), EXPORTERS["excel"].label))
        ttk.Combobox(export_frame, textvariable=self.export_label, values=exporter_labels, state="readonly").grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(export_frame, text="Export", command=self._export_any).grid(row=0, column=2, sticky="ew")

        ttk.Button(r, text="Export Excel", command=lambda: self._export_any(force_key="excel")).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r, text="Stop", command=self._stop_analysis).pack(side="left", fill="x", expand=True)

        prob = ttk.LabelFrame(left, text="Problems", padding=8)
        prob.grid(row=7, column=0, sticky="ew", pady=(0, 6))
        prob.columnconfigure(0, weight=1)
        self.prob_text = tk.Text(prob, height=6, wrap="word")
        self.prob_text.grid(row=0, column=0, sticky="ew")

        lst = ttk.LabelFrame(left, text="Images (red=not analyzed, green=ok, blue=refined)", padding=8)
        lst.grid(row=8, column=0, sticky="ew")
        lst.columnconfigure(0, weight=1)
        lst.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(lst)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, height=12)
        self.listbox.grid(row=0, column=0, sticky="nsew")

        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=list_scroll.set)

        self.listbox.bind("<<ListboxSelect>>", self._on_select_image)
        self.listbox.bind("<Button-3>", self._on_list_right_click)
        self.listbox.bind("<MouseWheel>", lambda e: (self.listbox.yview_scroll(int(-e.delta / 120), "units"), "break"))

        self._list_menu = tk.Menu(self, tearoff=0)
        self._list_menu.add_command(label="Discard analysis", command=self._discard_selected_analysis)

        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        view_box = ttk.LabelFrame(right, text="Viewer", padding=8)
        view_box.grid(row=0, column=0, sticky="nsew")
        view_box.rowconfigure(0, weight=1)
        view_box.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(view_box, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<Button-1>", self._on_canvas_left_click)
        self.canvas.bind("<ButtonPress-3>", self._roi_press)
        self.canvas.bind("<B3-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-3>", self._roi_release)

        self.bind("<Return>", self._on_enter_refine)

        info = ttk.LabelFrame(right, text="Metrics", padding=8)
        info.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        info.columnconfigure(0, weight=1)
        self.metrics_var = tk.StringVar(value="No image selected.")
        ttk.Label(info, textvariable=self.metrics_var, justify="left").grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=6)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------
    def _parse_max_area_unit2(self) -> Optional[float]:
        s = self.max_area_unit2.get().strip()
        if not s:
            return None
        try:
            v = float(s)
            if v <= 0:
                return None
            return float(v)
        except Exception:
            return None

    def _area_filters_px2(self, um_per_px: float, report_unit: str) -> Tuple[int, Optional[int], float, Optional[float]]:
        min_u2 = float(self.min_area_unit2.get())
        max_u2 = self._parse_max_area_unit2()

        unit_per_px = convert_um_per_px_to_unit_per_px(um_per_px, report_unit)
        area_per_px2 = unit_per_px ** 2
        if area_per_px2 <= 0:
            area_per_px2 = 1e-12

        min_px2 = int(max(1, round(min_u2 / area_per_px2)))
        max_px2 = int(round(max_u2 / area_per_px2)) if max_u2 is not None else None
        return min_px2, max_px2, min_u2, max_u2

    def _should_render_resized_for_path(self, img_path: Optional[str]) -> bool:
        if not RENDER_RESIZED:
            return False
        if img_path is None:
            return False
        r = self.results_by_path.get(img_path)
        if r is not None:
            return r.method == "unetpp"
        return self.analysis_method.get() == "unetpp"

    def _get_render_meta_for_path(self, img_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if img_path is None:
            return None
        meta = self.render_meta_by_path.get(img_path)
        if meta is not None:
            return meta
        if self._current_img_path == img_path and self._current_img_np is not None and self._should_render_resized_for_path(img_path):
            return build_unetpp_render_meta_from_original(self._current_img_np)
        return None

    def _orig_point_to_render_point(self, img_path: str, pt: Tuple[int, int]) -> Tuple[int, int]:
        meta = self._get_render_meta_for_path(img_path)
        if meta is None:
            return pt
        x, y = pt
        rx = int(round(x * meta["sx"]))
        ry = int(round(y * meta["sy"]))
        rx = max(0, min(meta["render_w"] - 1, rx))
        ry = max(0, min(meta["render_h"] - 1, ry))
        return (rx, ry)

    def _orig_roi_to_render_roi(self, img_path: str, roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        (x1, y1) = self._orig_point_to_render_point(img_path, (roi[0], roi[1]))
        (x2, y2) = self._orig_point_to_render_point(img_path, (roi[2], roi[3]))
        return (x1, y1, x2, y2)

    def _orig_poly_to_render_poly(self, img_path: str, pts: Optional[List[Tuple[int, int]]]) -> Optional[List[Tuple[int, int]]]:
        if pts is None:
            return None
        return [self._orig_point_to_render_point(img_path, p) for p in pts]

    def _orig_fov_to_render_fov(self, img_path: str, fov_circle: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        if fov_circle is None:
            return None
        meta = self._get_render_meta_for_path(img_path)
        if meta is None:
            return fov_circle
        cx, cy, r = fov_circle
        rcx, rcy = self._orig_point_to_render_point(img_path, (cx, cy))
        avg_scale = 0.5 * (meta["sx"] + meta["sy"])
        rr = int(round(r * avg_scale))
        rr = max(1, rr)
        return (rcx, rcy, rr)

    def _canvas_point_to_base_point(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        x0, y0 = self._canvas_img_origin
        x = cx - x0
        y = cy - y0
        if x < 0 or y < 0:
            return None
        ix = int(x / max(1e-9, self._display_scale))
        iy = int(y / max(1e-9, self._display_scale))
        if not (0 <= ix < self._current_display_base_w and 0 <= iy < self._current_display_base_h):
            return None
        return (ix, iy)

    def _base_point_to_original_point(self, bx: int, by: int) -> Tuple[int, int]:
        img_path = self._current_img_path
        if img_path is not None and self._should_render_resized_for_path(img_path):
            meta = self._get_render_meta_for_path(img_path)
            if meta is not None:
                ox = int(round(bx * meta["orig_w"] / max(meta["render_w"], 1)))
                oy = int(round(by * meta["orig_h"] / max(meta["render_h"], 1)))
                ox = max(0, min(meta["orig_w"] - 1, ox))
                oy = max(0, min(meta["orig_h"] - 1, oy))
                return (ox, oy)
        h, w = self._current_img_np.shape[:2]
        ox = max(0, min(w - 1, bx))
        oy = max(0, min(h - 1, by))
        return (ox, oy)

    def _original_xyxy_to_canvas_xyxy(self, roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        img_path = self._current_img_path
        if img_path is not None and self._should_render_resized_for_path(img_path):
            roi = self._orig_roi_to_render_roi(img_path, roi)
        x1, y1, x2, y2 = roi
        x0, y0 = self._canvas_img_origin
        s = max(1e-9, self._display_scale)
        cx1 = int(round(x0 + x1 * s))
        cy1 = int(round(y0 + y1 * s))
        cx2 = int(round(x0 + x2 * s))
        cy2 = int(round(y0 + y2 * s))
        return cx1, cy1, cx2, cy2

    def _current_focus_mask(self, h: int, w: int) -> Optional[np.ndarray]:
        if not self.use_fov.get():
            return None
        if self._fov_circle is None:
            return None
        return make_fov_focus_mask(h, w, self._fov_circle, int(self.fov_inner_margin_px.get()))

    # --------------------------------------------------------
    # U-Net++ loading
    # --------------------------------------------------------
    def _load_unetpp_model(self):
        try:
            if not TORCH_AVAILABLE:
                raise RuntimeError(f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")

            ckpt_path = Path(BEST_MODEL_PATH)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {BEST_MODEL_PATH}")

            UNetPlusPlus = import_unetpp_class()
            device = self.unetpp_device
            if device is None:
                raise RuntimeError("Could not resolve torch device")

            model = UNetPlusPlus(in_ch=3, out_ch=1, base=UNETPP_BASE_CHANNELS).to(device)
            state = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(state)
            model.eval()

            self.unetpp_model = model
            self.unetpp_status.set(
                f"U-Net++ loaded: {ckpt_path} | "
                f"target={UNET_GEOM['target_h']}x{UNET_GEOM['target_w']} | "
                f"padded={UNET_GEOM['padded_h']}x{UNET_GEOM['padded_w']} | "
                f"render={'resized' if RENDER_RESIZED else 'original'}"
            )
            self.status_var.set("U-Net++ model loaded.")
        except Exception as e:
            self.unetpp_model = None
            self.unetpp_status.set(f"U-Net++ load failed: {e}")
            messagebox.showerror("U-Net++", str(e))

    # --------------------------------------------------------
    # Directory / loading
    # --------------------------------------------------------
    def _browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.dir_path.set(d)

    def _load_images(self):
        root = self.dir_path.get().strip()
        if not root:
            messagebox.showerror("Error", "Select a directory.")
            return
        p = Path(root)
        if not p.exists():
            messagebox.showerror("Error", "Directory does not exist.")
            return

        paths = [str(fp) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
        paths.sort(key=lambda x: os.path.basename(x).lower())

        self.image_paths = paths
        self.listbox.delete(0, tk.END)
        for i, s in enumerate(self.image_paths):
            self.listbox.insert(tk.END, os.path.basename(s))
            self.listbox.itemconfig(i, fg=self._COLOR_RED)

        self.results_by_path.clear()
        self.masks_by_path.clear()
        self.metrics_by_path.clear()
        self.render_images_by_path.clear()
        self.render_masks_by_path.clear()
        self.render_meta_by_path.clear()
        self._roi_xyxy_by_path.clear()
        self._poly_pts_by_path.clear()

        self._analysis_has_run = False
        self._problems_buffer = []
        self.prob_text.delete("1.0", tk.END)

        self.status_var.set(f"Loaded {len(self.image_paths)} images.")
        if self.image_paths:
            self._select_index(0)

    def _select_index(self, idx: int):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.listbox.see(idx)
        self._show_image(self.image_paths[idx])

    def _on_select_image(self, _evt=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.image_paths):
            self._show_image(self.image_paths[idx])

    # --------------------------------------------------------
    # Context menu
    # --------------------------------------------------------
    def _on_list_right_click(self, evt):
        if not self.image_paths:
            return
        idx = self.listbox.nearest(evt.y)
        if idx < 0 or idx >= len(self.image_paths):
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self._list_menu.tk_popup(evt.x_root, evt.y_root)

    def _discard_selected_analysis(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        p = self.image_paths[idx]

        self.results_by_path.pop(p, None)
        self.masks_by_path.pop(p, None)
        self.metrics_by_path.pop(p, None)
        self.render_images_by_path.pop(p, None)
        self.render_masks_by_path.pop(p, None)
        self.render_meta_by_path.pop(p, None)
        self._roi_xyxy_by_path.pop(p, None)
        self._poly_pts_by_path.pop(p, None)

        self.listbox.itemconfig(idx, fg=self._COLOR_RED)
        self.status_var.set(f"Discarded: {os.path.basename(p)}")
        if self._current_img_path == p:
            self._show_image(p)

    def _refresh_list_colors(self):
        for i, p in enumerate(self.image_paths):
            r = self.results_by_path.get(p)
            if r is None:
                self.listbox.itemconfig(i, fg=self._COLOR_RED)
            else:
                self.listbox.itemconfig(i, fg=self._COLOR_BLUE if r.refined else self._COLOR_GREEN)

    # --------------------------------------------------------
    # Viewer display
    # --------------------------------------------------------
    def _show_image(self, img_path: str):
        self._current_img_path = img_path
        self._calib_pts = []

        try:
            img = read_image_any(img_path)
            self._current_img_np = img
        except Exception as e:
            self.status_var.set(f"Read error: {e}")
            return

        roi_orig = self._roi_xyxy_by_path.get(img_path)
        poly_orig = self._poly_pts_by_path.get(img_path)
        inner_margin = int(self.fov_inner_margin_px.get()) if self.use_fov.get() else 0
        fov_circle_orig = self._fov_circle if self.use_fov.get() else None

        use_resized_render = self._should_render_resized_for_path(img_path)

        if use_resized_render:
            render_img = self.render_images_by_path.get(img_path)
            render_mask = self.render_masks_by_path.get(img_path)
            render_meta = self.render_meta_by_path.get(img_path)

            if render_img is None:
                render_img = resize_image_for_model(to_bgr3(img))
            if render_meta is None:
                render_meta = build_unetpp_render_meta_from_original(img)

            self.render_meta_by_path[img_path] = render_meta

            roi_disp = self._orig_roi_to_render_roi(img_path, roi_orig) if roi_orig is not None else None
            poly_disp = self._orig_poly_to_render_poly(img_path, poly_orig)
            fov_disp = self._orig_fov_to_render_fov(img_path, fov_circle_orig)

            if render_mask is not None:
                geom = visual_geometry_from_mask(render_mask)
                rgb = overlay_visual(
                    render_img,
                    render_mask,
                    geom.get("contour"),
                    geom.get("feret_p1", (0, 0)),
                    geom.get("feret_p2", (0, 0)),
                    roi_xyxy=roi_disp,
                    poly_pts=poly_disp,
                    fov_circle=fov_disp,
                    fov_inner_margin_px=inner_margin,
                )
            else:
                rgb = overlay_visual(
                    render_img,
                    None,
                    None,
                    (0, 0),
                    (0, 0),
                    roi_xyxy=roi_disp,
                    poly_pts=poly_disp,
                    fov_circle=fov_disp,
                    fov_inner_margin_px=inner_margin,
                )
            disp = Image.fromarray(rgb)

        else:
            if img_path in self.metrics_by_path and img_path in self.masks_by_path:
                mask01 = self.masks_by_path[img_path]
                m = self.metrics_by_path[img_path]
                rgb = overlay_visual(
                    img,
                    mask01,
                    m.get("contour"),
                    m.get("feret_p1", (0, 0)),
                    m.get("feret_p2", (0, 0)),
                    roi_xyxy=roi_orig,
                    poly_pts=poly_orig,
                    fov_circle=fov_circle_orig,
                    fov_inner_margin_px=inner_margin,
                )
                disp = Image.fromarray(rgb)
            else:
                rgb = overlay_visual(
                    img,
                    None,
                    None,
                    (0, 0),
                    (0, 0),
                    roi_xyxy=roi_orig,
                    poly_pts=poly_orig,
                    fov_circle=fov_circle_orig,
                    fov_inner_margin_px=inner_margin,
                )
                disp = Image.fromarray(rgb)

        self._draw_on_canvas(disp)
        self._update_metrics_text(img_path)

    def _draw_on_canvas(self, pil_img: Image.Image):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 1100, 700

        w, h = pil_img.size
        self._current_display_base_w = w
        self._current_display_base_h = h

        scale = min(cw / w, ch / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        self._display_scale = float(scale)

        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x0 = (cw - new_w) // 2
        y0 = (ch - new_h) // 2
        self.canvas.create_image(x0, y0, anchor="nw", image=self._tk_img)
        self._canvas_img_origin = (x0, y0)

        if self._current_img_path is not None:
            roi = self._roi_xyxy_by_path.get(self._current_img_path)
            if roi is not None:
                cx1, cy1, cx2, cy2 = self._original_xyxy_to_canvas_xyxy(roi)
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="yellow", width=2, tag="roi_rect")

        if self._roi_dragging and self._roi_start_canvas and self._roi_end_canvas:
            self._draw_roi_rect_on_canvas()

    def _update_metrics_text(self, img_path: str):
        base = os.path.basename(img_path)
        labels = unit_labels(self.report_unit.get())

        max_u2 = self._parse_max_area_unit2()
        max_str = f"{max_u2:.4f} {labels['area']}" if max_u2 is not None else "—"

        fov_str = "OFF"
        if self.use_fov.get():
            if self._fov_circle is None:
                fov_str = "ON (not set)"
            else:
                cx, cy, rr = self._fov_circle
                fov_str = f"ON (cx={cx},cy={cy},r={rr}px,inner={int(self.fov_inner_margin_px.get())}px)"

        render_mode_str = "resized" if self._should_render_resized_for_path(img_path) else "original"

        r = self.results_by_path.get(img_path)
        if r is None:
            self.metrics_var.set(
                f"{base}\n\nNo analysis yet.\n"
                f"Method: {self.analysis_method.get()}\n"
                f"Render: {render_mode_str}\n"
                f"Scale: {self.um_per_px.get():.6f} µm/px\n"
                f"FOV: {fov_str}\n"
                f"Area filters: min={float(self.min_area_unit2.get()):.4f} {labels['area']}, max={max_str}"
            )
            return

        max_area_str = f"{r.max_area_unit2:.4f} {labels['area']}" if r.max_area_unit2 is not None else "—"

        if not r.ok:
            self.metrics_var.set(
                f"{base}\n\nStatus: PROBLEM\n{r.message}\n\n"
                f"Method: {r.method}\n"
                f"Render: {render_mode_str}\n"
                f"Scale: {r.um_per_px:.6f} µm/px\n"
                f"FOV: {fov_str}\n"
                f"Filters: min={r.min_area_unit2:.4f} {labels['area']}, max={max_area_str}\n"
                f"Refined: {'YES' if r.refined else 'NO'} ({r.refined_mode})"
            )
            return

        self.metrics_var.set(
            f"Area: {r.area:.4f} {labels['area']}\n"
            f"Perimeter: {r.perimeter:.4f} {labels['len']}\n"
            f"Diameter: {r.diameter:.4f} {labels['len']}\n"
            f"Circularity: {r.circularity:.4f}\n\n"
            f"Method: {r.method}\n"
            f"Render: {render_mode_str}\n"
            f"Scale: {r.um_per_px:.4f} µm/px\n"
        )

    # --------------------------------------------------------
    # FOV controls
    # --------------------------------------------------------
    def _auto_detect_fov_current(self):
        if self._current_img_np is None:
            messagebox.showwarning("FOV", "Select an image first.")
            return

        circ, msg = detect_fov_circle(self._current_img_np)
        if circ is None:
            self._fov_circle = None
            self.fov_status.set(msg)
            self.status_var.set(msg)
        else:
            self._fov_circle = circ
            self.fov_status.set(msg)
            self.status_var.set(msg)

        if self._current_img_path:
            self._show_image(self._current_img_path)

    def _start_manual_fov(self):
        if self._current_img_np is None or self._current_img_path is None:
            messagebox.showwarning("FOV", "Select an image first.")
            return
        self._fov_manual_click_mode = True
        self._fov_pts = []
        self.status_var.set("Manual FOV: left-click CENTER, then left-click EDGE.")
        self.fov_status.set("FOV manual: awaiting clicks")

    def _clear_fov(self):
        self._fov_circle = None
        self._fov_manual_click_mode = False
        self._fov_pts = []
        self.fov_status.set("FOV: not set")
        self.status_var.set("FOV cleared.")
        if self._current_img_path:
            self._show_image(self._current_img_path)

    # --------------------------------------------------------
    # Canvas click handling
    # --------------------------------------------------------
    def _on_toggle_draw_mode(self):
        self.status_var.set("Manual draw ON." if self._draw_mode.get() else "Manual draw OFF.")

    def _canvas_xy_to_image_xy(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        if self._current_img_np is None:
            return None
        base_pt = self._canvas_point_to_base_point(cx, cy)
        if base_pt is None:
            return None
        return self._base_point_to_original_point(base_pt[0], base_pt[1])

    def _on_canvas_left_click(self, evt):
        if self._current_img_np is None or self._current_img_path is None:
            return

        pt = self._canvas_xy_to_image_xy(evt.x, evt.y)
        if pt is None:
            return
        ix, iy = pt

        if self._fov_manual_click_mode:
            self._fov_pts.append((ix, iy))
            if len(self._fov_pts) == 1:
                self.status_var.set("Manual FOV: center set. Now click edge.")
                self.fov_status.set("FOV manual: center set")
                return
            if len(self._fov_pts) >= 2:
                (cx, cy) = self._fov_pts[0]
                (ex, ey) = self._fov_pts[1]
                r = int(round(math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)))
                if r < 10:
                    self.status_var.set("Manual FOV: radius too small, try again.")
                    self.fov_status.set("FOV manual: failed")
                    self._fov_pts = []
                    return
                self._fov_circle = (int(cx), int(cy), int(r))
                self._fov_manual_click_mode = False
                self._fov_pts = []
                msg = f"FOV manual: (cx={cx}, cy={cy}, r={r}px)"
                self.fov_status.set(msg)
                self.status_var.set(msg)
                if self._current_img_path:
                    self._show_image(self._current_img_path)
                return

        if self._draw_mode.get():
            pts = self._poly_pts_by_path.get(self._current_img_path, [])
            if len(pts) >= 3:
                x0, y0 = pts[0]
                if (ix - x0) ** 2 + (iy - y0) ** 2 <= (10 ** 2):
                    pts.append(pts[0])
                    self._poly_pts_by_path[self._current_img_path] = pts
                    self.status_var.set("Manual polygon closed. Use 'Apply poly' to compute.")
                    self._show_image(self._current_img_path)
                    return

            pts.append((ix, iy))
            self._poly_pts_by_path[self._current_img_path] = pts
            self.status_var.set(f"Manual poly: {len(pts)} point(s)")
            self._show_image(self._current_img_path)
            return

        if self._calib_click_mode:
            self._calib_pts.append((ix, iy))
            if len(self._calib_pts) == 1:
                self.status_var.set("Scale calib: first point set. Click second point.")
                return
            if len(self._calib_pts) >= 2:
                p1 = self._calib_pts[0]
                p2 = self._calib_pts[1]
                d_px = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                if d_px < 1e-6:
                    self.status_var.set("Scale calib: points too close. Try again.")
                    self._calib_pts = []
                    return

                known_len = float(self.scale_known_len.get())
                known_unit = self.scale_unit.get().strip().lower()
                if known_unit not in UNIT_TO_UM:
                    messagebox.showerror("Scale", f"Invalid unit: {known_unit}")
                    self._calib_pts = []
                    return

                known_um = known_len * UNIT_TO_UM[known_unit]
                um_per_px = known_um / d_px
                self.um_per_px.set(float(um_per_px))

                self._calib_pts = []
                self._calib_click_mode = False
                self.status_var.set(f"Scale set: {um_per_px:.6f} µm/px")
                self._update_metrics_text(self._current_img_path)
                return

    # --------------------------------------------------------
    # ROI for SAM refinement
    # --------------------------------------------------------
    def _roi_press(self, evt):
        if self._current_img_np is None or self._current_img_path is None:
            return
        self._roi_dragging = True
        self._roi_start_canvas = (evt.x, evt.y)
        self._roi_end_canvas = (evt.x, evt.y)
        self._draw_roi_rect_on_canvas()

    def _roi_drag(self, evt):
        if not self._roi_dragging:
            return
        self._roi_end_canvas = (evt.x, evt.y)
        self._draw_roi_rect_on_canvas()

    def _roi_release(self, evt):
        if not self._roi_dragging:
            return
        self._roi_end_canvas = (evt.x, evt.y)
        self._roi_dragging = False

        roi = self._canvas_roi_to_image_xyxy(self._roi_start_canvas, self._roi_end_canvas)
        if roi is None:
            self.status_var.set("ROI not set (too small or outside image).")
            self._roi_start_canvas = None
            self._roi_end_canvas = None
            if self._current_img_path:
                self._show_image(self._current_img_path)
            return

        self._roi_xyxy_by_path[self._current_img_path] = roi
        self.status_var.set(f"ROI set: {roi}")
        if self._current_img_path:
            self._show_image(self._current_img_path)

    def _draw_roi_rect_on_canvas(self):
        if self._roi_start_canvas is None or self._roi_end_canvas is None:
            return
        self.canvas.delete("roi_rect_drag")
        x1, y1 = self._roi_start_canvas
        x2, y2 = self._roi_end_canvas
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2, tag="roi_rect_drag")

    # --------------------------------------------------------
    # Shared ROI conversion
    # --------------------------------------------------------
    def _canvas_roi_to_image_xyxy(
        self,
        start_canvas: Optional[Tuple[int, int]],
        end_canvas: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if self._current_img_np is None:
            return None
        if start_canvas is None or end_canvas is None:
            return None

        p1 = self._canvas_point_to_base_point(start_canvas[0], start_canvas[1])
        p2 = self._canvas_point_to_base_point(end_canvas[0], end_canvas[1])
        if p1 is None or p2 is None:
            return None

        bx1, by1 = p1
        bx2, by2 = p2

        ox1, oy1 = self._base_point_to_original_point(bx1, by1)
        ox2, oy2 = self._base_point_to_original_point(bx2, by2)

        if abs(ox2 - ox1) < 5 or abs(oy2 - oy1) < 5:
            return None

        x1 = min(ox1, ox2)
        y1 = min(oy1, oy2)
        x2 = max(ox1, ox2)
        y2 = max(oy1, oy2)
        return (x1, y1, x2, y2)

    # --------------------------------------------------------
    # Scale actions
    # --------------------------------------------------------
    def _apply_click_calibration(self):
        if self._current_img_np is None:
            messagebox.showwarning("Scale", "Select an image first.")
            return
        self._calib_click_mode = True
        self._calib_pts = []
        self.status_var.set("Scale click-calib ON: left-click 2 points on scale bar/known length.")

    def _auto_detect_scale(self):
        if self._current_img_np is None:
            messagebox.showwarning("Scale", "Select an image first.")
            return

        px, roi, msg = try_detect_scale_bar_px_and_roi(self._current_img_np)
        if px is None:
            self.auto_scale_px.set(msg)
            self.status_var.set(msg)
            return

        ocr_val = None
        ocr_unit = None
        ocr_msg = ""
        if roi is not None:
            ocr_val, ocr_unit, ocr_msg = try_ocr_scale_value_and_unit(self._current_img_np, roi)

        if ocr_val is not None and ocr_unit is not None:
            self.scale_known_len.set(float(ocr_val))
            self.scale_unit.set(ocr_unit)

            known_um = float(ocr_val) * UNIT_TO_UM[ocr_unit]
            um_per_px = known_um / float(px)
            self.um_per_px.set(float(um_per_px))

            self.auto_scale_px.set(f"{msg} | {ocr_msg}")
            self.status_var.set(f"Auto scale applied (OCR): {um_per_px:.6f} µm/px")
            self._update_metrics_text(self._current_img_path or self.image_paths[0])
            return

        self.auto_scale_px.set(f"{msg} | OCR not applied: {ocr_msg or 'not available'}")

        known_len = float(self.scale_known_len.get())
        known_unit = self.scale_unit.get().strip().lower()
        if known_unit not in UNIT_TO_UM:
            messagebox.showerror("Scale", f"Invalid unit: {known_unit}")
            return
        known_um = known_len * UNIT_TO_UM[known_unit]
        um_per_px = known_um / float(px)
        self.um_per_px.set(float(um_per_px))
        self.status_var.set(f"Auto scale applied: {um_per_px:.6f} µm/px")
        self._update_metrics_text(self._current_img_path or self.image_paths[0])

    # --------------------------------------------------------
    # Manual polygon actions
    # --------------------------------------------------------
    def _clear_manual_poly(self):
        if self._current_img_path is None:
            return
        self._poly_pts_by_path.pop(self._current_img_path, None)
        self.status_var.set("Manual polygon cleared.")
        self._show_image(self._current_img_path)

    def _apply_manual_poly_mask(self):
        if self._current_img_path is None or self._current_img_np is None:
            return
        pts = self._poly_pts_by_path.get(self._current_img_path, [])
        if pts is None or len(pts) < 3:
            messagebox.showwarning("Manual", "Need at least 3 points.")
            return

        h, w = self._current_img_np.shape[:2]
        mask01 = polygon_to_mask01(h, w, pts)
        if mask01 is None:
            messagebox.showwarning("Manual", "Failed to create mask.")
            return

        focus = self._current_focus_mask(h, w)
        if focus is not None:
            mask01 = (mask01.astype(np.uint8) & focus.astype(np.uint8)).astype(np.uint8)

        um_per_px = float(self.um_per_px.get())
        unit = self.report_unit.get()
        metrics = measure_metrics_from_mask(mask01, um_per_px, unit)
        self.masks_by_path[self._current_img_path] = mask01
        self.metrics_by_path[self._current_img_path] = metrics

        # Keep the resized render cache synchronized after manual refinement.
        if self._should_render_resized_for_path(self._current_img_path):
            render_img = resize_image_for_model(to_bgr3(self._current_img_np))
            render_mask = cv2.resize(mask01.astype(np.uint8), (render_img.shape[1], render_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.render_images_by_path[self._current_img_path] = render_img
            self.render_masks_by_path[self._current_img_path] = (render_mask > 0).astype(np.uint8)
            self.render_meta_by_path[self._current_img_path] = build_unetpp_render_meta_from_original(self._current_img_np)

        _, _, min_u2, max_u2 = self._area_filters_px2(um_per_px, unit)

        r = AnalysisResult(
            file=os.path.basename(self._current_img_path),
            path=self._current_img_path,
            ok=True,
            message="OK (manual)",
            method=self.results_by_path.get(
                self._current_img_path,
                AnalysisResult("", "", False, "", "manual", 0, "", 0, None, None, None, None, None),
            ).method if self._current_img_path in self.results_by_path else "manual",
            um_per_px=um_per_px,
            report_unit=unit,
            min_area_unit2=min_u2,
            max_area_unit2=max_u2,
            area=float(metrics["Area"]),
            perimeter=float(metrics["Perimeter"]),
            diameter=float(metrics["Diameter"]),
            circularity=float(metrics["Circularity"]),
            refined=True,
            refined_mode="manual",
        )
        self.results_by_path[self._current_img_path] = r
        self._refresh_list_colors()
        self.status_var.set("Manual mask applied.")
        self._show_image(self._current_img_path)

    # --------------------------------------------------------
    # SAM actions
    # --------------------------------------------------------
    def _load_sam_model(self):
        if SAM_CLASS is None:
            messagebox.showerror("SAM", f"SAM unavailable: {SAM_IMPORT_ERROR}")
            return
        name = self.sam_choice.get()
        if name == "Pick a SAM Model":
            messagebox.showwarning("SAM", "Pick a SAM model first.")
            return
        weights = self.sam_models.get(name)
        if weights is None:
            messagebox.showerror("SAM", f"Unknown/unavailable model: {name}")
            return
        try:
            self.sam_model = SAM_CLASS(weights)
            self.status_var.set(f"SAM loaded: {name}")
        except Exception as e:
            messagebox.showerror("SAM", f"Failed to load SAM: {e}")

    def _on_enter_refine(self, _evt=None):
        self._refine_selected_with_sam()

    def _refine_selected_with_sam(self):
        if self._current_img_path is None or self._current_img_np is None:
            return
        if self.sam_model is None:
            messagebox.showwarning("SAM", "Load a SAM model first.")
            return

        roi = self._roi_xyxy_by_path.get(self._current_img_path)
        if roi is None:
            messagebox.showwarning("SAM", "Set an ROI first (right-drag on viewer).")
            return

        x1, y1, x2, y2 = roi
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        mask01, score, msg = sam_predict_mask01_from_bbox(self.sam_model, self._current_img_np, bbox)
        if mask01 is None:
            self.status_var.set(f"SAM failed: {msg}")
            return

        h, w = mask01.shape[:2]
        focus = self._current_focus_mask(h, w)
        if focus is not None:
            mask01 = (mask01.astype(np.uint8) & focus.astype(np.uint8)).astype(np.uint8)

        um_per_px = float(self.um_per_px.get())
        unit = self.report_unit.get()
        min_px2, max_px2, min_u2, max_u2 = self._area_filters_px2(um_per_px, unit)

        area_px2 = int(mask01.sum())
        if area_px2 < min_px2:
            self.status_var.set("SAM mask rejected: below min area filter.")
            return
        if max_px2 is not None and area_px2 > max_px2:
            self.status_var.set("SAM mask rejected: above max area filter.")
            return

        metrics = measure_metrics_from_mask(mask01, um_per_px, unit)
        self.masks_by_path[self._current_img_path] = mask01
        self.metrics_by_path[self._current_img_path] = metrics

        if self._should_render_resized_for_path(self._current_img_path):
            render_img = resize_image_for_model(to_bgr3(self._current_img_np))
            render_mask = cv2.resize(mask01.astype(np.uint8), (render_img.shape[1], render_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.render_images_by_path[self._current_img_path] = render_img
            self.render_masks_by_path[self._current_img_path] = (render_mask > 0).astype(np.uint8)
            self.render_meta_by_path[self._current_img_path] = build_unetpp_render_meta_from_original(self._current_img_np)

        prev_method = self.results_by_path[self._current_img_path].method if self._current_img_path in self.results_by_path else self.analysis_method.get()

        r = AnalysisResult(
            file=os.path.basename(self._current_img_path),
            path=self._current_img_path,
            ok=True,
            message=f"OK (sam, score={score})" if score is not None else "OK (sam)",
            method=prev_method,
            um_per_px=um_per_px,
            report_unit=unit,
            min_area_unit2=min_u2,
            max_area_unit2=max_u2,
            area=float(metrics["Area"]),
            perimeter=float(metrics["Perimeter"]),
            diameter=float(metrics["Diameter"]),
            circularity=float(metrics["Circularity"]),
            refined=True,
            refined_mode="sam",
        )
        self.results_by_path[self._current_img_path] = r
        self._refresh_list_colors()
        self.status_var.set("SAM refinement applied.")
        self._show_image(self._current_img_path)

    # --------------------------------------------------------
    # Analysis worker
    # --------------------------------------------------------
    def _start_analysis(self):
        if not self.image_paths:
            messagebox.showwarning("Analyze", "Load images first.")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("Analyze", "Analysis already running.")
            return

        method = self.analysis_method.get()
        if method == "unetpp" and self.unetpp_model is None:
            try:
                self._load_unetpp_model()
            except Exception:
                pass
            if self.unetpp_model is None:
                messagebox.showerror("U-Net++", "U-Net++ model is not loaded.")
                return

        if self.use_fov.get() and self._fov_circle is None:
            src = self._current_img_np
            if src is None:
                try:
                    src = read_image_any(self.image_paths[0])
                except Exception:
                    src = None
            if src is not None:
                circ, msg = detect_fov_circle(src)
                self._fov_circle = circ
                self.fov_status.set(msg if circ is not None else f"FOV auto failed: {msg}")

        self._stop_flag.clear()
        self._problems_buffer = []
        self.prob_text.delete("1.0", tk.END)
        self.status_var.set("Analyzing...")
        self._analysis_has_run = True

        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._worker_thread.start()

    def _analysis_worker(self):
        um_per_px = float(self.um_per_px.get())
        unit = self.report_unit.get()
        min_px2, max_px2, min_u2, max_u2 = self._area_filters_px2(um_per_px, unit)

        use_fov = bool(self.use_fov.get())
        fov_circle = self._fov_circle
        inner_margin = int(self.fov_inner_margin_px.get())
        method = self.analysis_method.get()

        total = len(self.image_paths)
        for i, p in enumerate(self.image_paths):
            if self._stop_flag.is_set():
                self._q.put(("status", "Stopped."))
                break

            try:
                img = read_image_any(p)
            except Exception as e:
                msg = f"Read error: {e}"
                self._q.put(("result", p, None, None, None, None, False, msg, method, um_per_px, unit, min_u2, max_u2))
                continue

            try:
                if method == "physics":
                    mask01, msg = segment_organoid_mask_physics(
                        img_bgr_or_gray=img,
                        min_area_px2=min_px2,
                        max_area_px2=max_px2,
                        use_fov=use_fov,
                        fov_circle=fov_circle,
                        fov_inner_margin_px=inner_margin,
                    )
                    render_img = None
                    render_mask = None
                    render_meta = None

                elif method == "unetpp":
                    if self.unetpp_model is None or self.unetpp_device is None:
                        raise RuntimeError("U-Net++ model not loaded")

                    img_rgb = bgr_to_rgb3(img)
                    pred_out = predict_mask_unetpp(
                        model=self.unetpp_model,
                        device=self.unetpp_device,
                        img_rgb=img_rgb,
                        threshold=UNETPP_THRESHOLD,
                    )

                    pred_mask_orig = pred_out["pred_orig"]
                    pred_mask_resized = pred_out["pred_resized"]

                    pred_mask_orig = clean_small_components(
                        pred_mask_orig,
                        min_size_px=MIN_COMPONENT_SIZE_PX,
                        keep_only_largest=KEEP_ONLY_LARGEST_COMPONENT,
                    )

                    focus = None
                    if use_fov and fov_circle is not None:
                        h, w = pred_mask_orig.shape[:2]
                        focus = make_fov_focus_mask(h, w, fov_circle, inner_margin)

                    pred_mask_orig = apply_focus_mask_if_needed(pred_mask_orig, focus)
                    mask01, msg = filter_mask_by_area(pred_mask_orig, min_px2, max_px2)

                    render_img = resize_image_for_model(to_bgr3(img))
                    render_mask = pred_mask_resized
                    render_meta = build_unetpp_render_meta_from_original(img)

                    if mask01 is not None:
                        render_mask = cv2.resize(mask01.astype(np.uint8), (render_img.shape[1], render_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        render_mask = (render_mask > 0).astype(np.uint8)

                else:
                    raise RuntimeError(f"Unknown analysis method: {method}")

                if mask01 is None:
                    self._q.put(("result", p, None, None, render_img, render_mask, False, msg, method, um_per_px, unit, min_u2, max_u2, render_meta))
                else:
                    metrics = measure_metrics_from_mask(mask01, um_per_px, unit)
                    self._q.put(("result", p, mask01, metrics, render_img, render_mask, True, msg, method, um_per_px, unit, min_u2, max_u2, render_meta))

            except Exception as e:
                self._q.put(("result", p, None, None, None, None, False, f"{method} failed: {e}", method, um_per_px, unit, min_u2, max_u2, None))

            self._q.put(("progress", i + 1, total, os.path.basename(p)))

        self._q.put(("done",))

    def _stop_analysis(self):
        self._stop_flag.set()
        self.status_var.set("Stopping...")

    # --------------------------------------------------------
    # Queue polling
    # --------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                item = self._q.get_nowait()
                kind = item[0]

                if kind == "progress":
                    done, total, name = item[1], item[2], item[3]
                    self.status_var.set(f"Analyzing {done}/{total}: {name}")

                elif kind == "status":
                    self.status_var.set(item[1])

                elif kind == "result":
                    if len(item) == 13:
                        _, path, mask01, metrics, render_img, render_mask, ok, msg, method, um_per_px, unit, min_u2, max_u2 = item
                        render_meta = None
                    elif len(item) == 14:
                        _, path, mask01, metrics, render_img, render_mask, ok, msg, method, um_per_px, unit, min_u2, max_u2, render_meta = item
                    else:
                        raise ValueError(f"Unexpected result tuple length: {len(item)}")

                    if render_img is not None:
                        self.render_images_by_path[path] = render_img
                    if render_mask is not None:
                        self.render_masks_by_path[path] = render_mask
                    if render_meta is not None:
                        self.render_meta_by_path[path] = render_meta

                    if ok and mask01 is not None and metrics is not None:
                        self.masks_by_path[path] = mask01
                        self.metrics_by_path[path] = metrics

                        r = AnalysisResult(
                            file=os.path.basename(path),
                            path=path,
                            ok=True,
                            message=msg,
                            method=method,
                            um_per_px=um_per_px,
                            report_unit=unit,
                            min_area_unit2=min_u2,
                            max_area_unit2=max_u2,
                            area=float(metrics["Area"]),
                            perimeter=float(metrics["Perimeter"]),
                            diameter=float(metrics["Diameter"]),
                            circularity=float(metrics["Circularity"]),
                            refined=False,
                            refined_mode="auto",
                        )
                        self.results_by_path[path] = r
                    else:
                        r = AnalysisResult(
                            file=os.path.basename(path),
                            path=path,
                            ok=False,
                            message=msg,
                            method=method,
                            um_per_px=um_per_px,
                            report_unit=unit,
                            min_area_unit2=min_u2,
                            max_area_unit2=max_u2,
                            area=None,
                            perimeter=None,
                            diameter=None,
                            circularity=None,
                            refined=False,
                            refined_mode="auto",
                        )
                        self.results_by_path[path] = r
                        self._problems_buffer.append(f"{os.path.basename(path)}: {msg}")

                    self._refresh_list_colors()
                    if self._current_img_path == path:
                        self._show_image(path)

                elif kind == "done":
                    if self._problems_buffer:
                        self.prob_text.delete("1.0", tk.END)
                        self.prob_text.insert(tk.END, "\n".join(self._problems_buffer))
                    else:
                        self.prob_text.delete("1.0", tk.END)
                        self.prob_text.insert(tk.END, "No problems.")
                    self.status_var.set("Analysis complete.")
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------
    def _export_any(self, force_key: Optional[str] = None):
        if not self.results_by_path:
            messagebox.showwarning("Export", "No results to export.")
            return

        key = force_key
        if key is None:
            chosen_label = self.export_label.get()
            key = self._export_key_by_label.get(chosen_label, "excel")

        exporter = EXPORTERS.get(key)
        if exporter is None:
            messagebox.showerror("Export", f"Unknown exporter: {key}")
            return

        out = ask_save_path_for_export(self, exporter)
        if not out:
            return

        df = build_results_dataframe(self.image_paths, self.results_by_path)
        meta = build_export_meta(self)

        try:
            exporter.export(out, df, meta)
        except ExportError as e:
            messagebox.showerror("Export", str(e))
            return
        except Exception as e:
            messagebox.showerror("Export", f"Export failed: {e}")
            return

        self.status_var.set(f"Exported: {out}")


if __name__ == "__main__":
    app = OrganoidBatchGUI()
    app.mainloop()
