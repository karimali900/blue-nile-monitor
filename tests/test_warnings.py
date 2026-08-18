"""Tests for the construction-warning detector (Sentinel-1 backscatter gain)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nilemonitor.analysis.warnings import detect_construction  # noqa: E402


def _grid(seed=0, rows=200, cols=200):
    rng = np.random.default_rng(seed)
    vv = rng.normal(-12.0, 0.6, (rows, cols)).astype("float32")
    vh = rng.normal(-19.0, 0.8, (rows, cols)).astype("float32")
    wm = np.zeros((rows, cols), dtype=bool)
    wm[150:190, 10:40] = True  # a water patch (radar shadow proxy)
    return vv, vh, wm


def test_no_alerts_on_unchanged_scene():
    vv0, vh0, w0 = _grid(seed=1)
    vv1, vh1, w1 = _grid(seed=2)  # different speckle, same statistics
    dates = [dt.date(2026, 8, 2), dt.date(2026, 8, 14)]
    alerts = detect_construction(dates, [vv0, vv1], [vh0, vh1], [w0, w1])
    assert alerts == []


def test_detects_planted_land_gain():
    vv0, vh0, w0 = _grid(seed=3)
    vv1, vh1, w1 = _grid(seed=4)
    vv1[60:70, 90:100] += 5.0  # planted construction patch (+5 dB VV, +5 dB VH)
    vh1[60:70, 90:100] += 4.0
    alerts = detect_construction(
        [dt.date(2026, 8, 2), dt.date(2026, 8, 14)], [vv0, vv1], [vh0, vh1], [w0, w1]
    )
    assert len(alerts) == 1
    a = alerts[0]
    assert a.kind == "land"
    assert 80 <= a.pixels <= 130  # planted patch ~100 px, clustering-tolerant
    assert a.vv_gain_db >= 3.5
    # pixel-space fallback: lon = x*deg, lat = -y*deg
    assert 0 <= a.lon * 2000 <= 200 and 0 <= -a.lat * 2000 <= 200


def test_ignores_water_fill_change():
    vv0, vh0, w0 = _grid(seed=5)
    vv1, vh1, w1 = _grid(seed=6)
    # reservoir filling = land losing backscatter (or water growing): a strong
    # DECREASE in the water patch must not fire any alert
    vv1[160:180, 20:30] -= 8.0
    vh1[160:180, 20:30] -= 8.0
    alerts = detect_construction(
        [dt.date(2026, 8, 2), dt.date(2026, 8, 14)], [vv0, vv1], [vh0, vh1], [w0, w1]
    )
    assert alerts == []
