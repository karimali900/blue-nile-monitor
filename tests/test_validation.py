"""Unit tests for the validation metrics and storage-volume math."""
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nilemonitor.validation.metrics import confusion_metrics, storage_volume


def _grid(rows, cols, res=0.0005, origin=(34.8, 11.5)):
    return Affine(res, 0, origin[0], 0, -res, origin[1])


def test_confusion_metrics_perfect_match():
    mask = np.zeros((10, 10), dtype="float32")
    mask[2:8, 3:9] = 1.0
    ref = mask.astype(bool)
    m = confusion_metrics(mask, _grid(10, 10), ref, _grid(10, 10, res=0.0005))
    assert m["tp_pixels"] == 36
    assert m["fp_pixels"] == 0
    assert m["fn_pixels"] == 0
    assert m["tn_pixels"] == 64
    assert m["overall_accuracy"] == 1.0
    assert m["precision_user"] == 1.0
    assert m["recall_producer"] == 1.0
    assert m["iou"] == 1.0
    assert m["kappa"] == 1.0


def test_confusion_metrics_hand_computed():
    pred = np.zeros((4, 4), dtype="float32")
    pred[0:2, 0:2] = 1.0
    pred[3, 3] = 1.0
    ref = np.zeros((4, 4), dtype=bool)
    ref[0:2, 0:1] = True
    ref[2:4, 2:4] = True
    m = confusion_metrics(pred, _grid(4, 4), ref, _grid(4, 4, res=0.0005))
    # tp: pred & ref = rows0-1 col0 (2) + pred[3,3] & ref[3,3] (1) = 3
    # fp: pred minus ref = rows0-1 col1 (2); fn: ref minus pred = ref rows2-3 cols2-3 minus [3,3] (3)
    assert m["tp_pixels"] == 3
    assert m["fp_pixels"] == 2
    assert m["fn_pixels"] == 3
    assert m["tn_pixels"] == 8
    assert m["overall_accuracy"] == pytest.approx(round(11 / 16, 4))
    assert m["precision_user"] == pytest.approx(3 / 5)
    assert m["recall_producer"] == pytest.approx(3 / 6)
    assert m["iou"] == pytest.approx(3 / 8)
    po = 11 / 16
    pc = (5 / 16) * (6 / 16) + (11 / 16) * (10 / 16)
    assert m["kappa"] == pytest.approx(round((po - pc) / (1 - pc), 4))


def test_confusion_metrics_ignores_nodata():
    pred = np.full((4, 4), np.nan, dtype="float32")
    pred[1:3, 1:3] = 1.0
    ref = np.zeros((4, 4), dtype=bool)
    ref[1:3, 1:2] = True
    m = confusion_metrics(pred, _grid(4, 4), ref, _grid(4, 4, res=0.0005))
    assert m["tp_pixels"] + m["fp_pixels"] + m["fn_pixels"] + m["tn_pixels"] == 4


def test_storage_volume_flat_bed():
    mask = np.ones((10, 10), dtype="float32")
    dem = np.full((10, 10), 600.0)
    tf = _grid(10, 10, res=0.001)
    m = storage_volume(mask, tf, dem, tf, crest_level_m=655.0)
    cell = (0.001 * 111_320 * np.cos(np.radians(11.5))) * (0.001 * 110_574)
    expected_vol = 100 * cell * 55 / 1e9
    assert m["volume_km3"] == pytest.approx(round(expected_vol, 3), abs=1e-9)
    assert m["mean_depth_m"] == pytest.approx(55.0)
    assert m["area_km2"] == pytest.approx(round(100 * cell / 1e6, 2))


def test_storage_volume_negative_depth_clipped():
    mask = np.ones((5, 5), dtype="float32")
    dem = np.full((5, 5), 700.0)
    tf = _grid(5, 5, res=0.001)
    m = storage_volume(mask, tf, dem, tf, crest_level_m=655.0)
    assert m["volume_km3"] == 0.0
    assert m["mean_depth_m"] == 0.0