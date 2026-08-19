"""Small helpers to render satellite scene previews as PNG bytes."""
from __future__ import annotations

import io

import numpy as np


def render_rgb_png(rgb: np.ndarray, low: float = 2.0, high: float = 98.0) -> bytes:
    """Encode an (h, w, 3) float array (0..1) as PNG bytes with percentile stretch."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return b""
    arr = np.nan_to_num(rgb[..., :3].astype("float64"), nan=0.0)
    p_low, p_high = np.nanpercentile(arr, [low, high])
    if p_high - p_low < 1e-9:
        p_low, p_high = arr.min(), max(arr.max(), p_low + 1e-9)
    scaled = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
    scaled = (scaled * 255).astype("uint8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(scaled, interpolation="nearest")
    ax.axis("off")
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


def render_db_png(db: np.ndarray, vmin: float = -25.0, vmax: float = -2.0) -> bytes:
    """Encode a sigma0 dB array as a colour-mapped PNG (water dark, land bright)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("db", ["#0b1220", "#1c3a5e", "#4e79a7", "#c7d8e8"])
    buf = io.BytesIO()
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(db, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()
