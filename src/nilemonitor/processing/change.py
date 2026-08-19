"""Temporal change detection and a construction/activity index.

The construction index is a pixel-level measure of "how much changed here"
over the observation window:

  activity = std(backscatter_dB) over time  (SAR)
             +
             fraction of scenes classified as water/not-water flip (optical/SAR)

Newly built structures, dam infill, reservoir flooding and land clearing all
produce sustained, localised SAR backscatter change and/or water-state flips,
so high activity over time is a strong cue for active human construction.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .water import water_mask_sar


def stack_activity(backscatter_db_stack: Sequence[np.ndarray]) -> np.ndarray:
    """Pixel-wise temporal std of SAR backscatter (dB). NaN where < 2 valid dates."""
    if not backscatter_db_stack:
        return np.zeros((0, 0), dtype="float32")
    stack = np.stack([a.astype("float32") for a in backscatter_db_stack])
    count = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.nanstd(stack, axis=0)
    std[count < 2] = np.nan
    return std


def water_flip_fraction(water_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Fraction of observations where the water/land state changes vs. the majority state."""
    if not water_masks:
        return np.zeros((0, 0), dtype="float32")
    stack = np.stack([m.astype("float32") for m in water_masks])
    frac_water = np.nanmean(stack, axis=0)
    flip = np.minimum(frac_water, 1.0 - frac_water)
    flip[~np.isfinite(frac_water)] = np.nan
    return flip.astype("float32")


def construction_index(
    activity: np.ndarray,
    flip_fraction: np.ndarray,
    activity_p95: float | None = None,
    flip_min: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Combine SAR activity and water-state flips into a 0..1 construction index.

    Returns (index raster, threshold used for "high activity" pixels).
    """
    a = activity.astype("float32")
    f = flip_fraction.astype("float32")
    finite = np.isfinite(a) & np.isfinite(f)
    if not finite.any():
        return np.zeros_like(a), 0.0

    if activity_p95 is None:
        activity_p95 = float(np.nanpercentile(a[finite], 95))
    if activity_p95 <= 0:
        activity_p95 = 1.0

    a_norm = np.clip(a / activity_p95, 0, 1)
    f_norm = np.clip(f / max(flip_min, 1e-6), 0, 1)
    index = np.where(finite, 0.6 * a_norm + 0.4 * f_norm, np.nan)
    return index.astype("float32"), float(activity_p95)


def classify_change(
    water_now: np.ndarray,
    water_baseline: np.ndarray,
    min_pixels: int = 4,
) -> np.ndarray:
    """Change classes relative to baseline: 0 none, 1 gained water, -1 lost water."""
    out = np.zeros(water_now.shape, dtype="int8")
    gained = water_now & ~water_baseline
    lost = ~water_now & water_baseline
    out[gained] = 1
    out[lost] = -1
    return out
