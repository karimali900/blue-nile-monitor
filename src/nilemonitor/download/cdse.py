"""Copernicus Data Space (CDSE) downloader for Sentinel-1 GRD products.

Used for the SAR layer because the public AWS bucket for S1 GRD is
requester-pays. Requires CDSE credentials in .env.

We never download whole .SAFE zips — only the VV/VH measurement GeoTIFFs (plus
the annotation XML carrying the geolocation grid) via the OData node tree and
$value endpoint, cached locally by product id.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import requests
from rasterio.transform import Affine

from ..auth import CdseAuth

log = logging.getLogger(__name__)

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"

BAND_POL_RE = re.compile(r"-(vv|vh|hh|hv)-")


def _ring_from_bbox(bbox: tuple[float, float, float, float]) -> str:
    lon_min, lat_min, lon_max, lat_max = bbox
    ring = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    ]
    return "SRID=4326;POLYGON((" + ",".join(f"{x:.6f} {y:.6f}" for x, y in ring) + "))"


def search_s1_grd(
    auth: CdseAuth,
    bbox: tuple[float, float, float, float],
    date_from: dt.date,
    date_to: dt.date,
    max_items: int = 6,
) -> list[dict]:
    """Search CDSE for IW GRD dual-pol (VV+VH) products over bbox, newest first."""
    ring = _ring_from_bbox(bbox)
    filters = (
        "OData.CSC.Intersects(area=geography'" + ring + "')"
        " and Collection/Name eq 'SENTINEL-1'"
        " and contains(Name, 'IW_GRDH')"
        " and contains(Name, 'SDV')"
        " and not contains(Name, 'COG')"
        f" and ContentDate/Start gt {date_from.isoformat()}T00:00:00.000Z"
        f" and ContentDate/Start lt {date_to.isoformat()}T23:59:59.000Z"
    )
    resp = requests.get(
        f"{CATALOGUE}/Products",
        params={"$filter": filters, "$top": 100, "$orderby": "ContentDate/Start desc"},
        timeout=60,
    )
    resp.raise_for_status()
    products = resp.json().get("value", [])
    return products[:max_items]


def search_landsat(
    auth: CdseAuth,
    bbox: tuple[float, float, float, float],
    date_from: dt.date,
    date_to: dt.date,
    max_items: int = 6,
) -> list[dict]:
    """Search CDSE for Landsat C2 L1TP products over bbox, newest first."""
    ring = _ring_from_bbox(bbox)
    filters = (
        "OData.CSC.Intersects(area=geography'" + ring + "')"
        " and contains(Name, 'LC0')"
        " and contains(Name, 'L1TP')"
        f" and ContentDate/Start gt {date_from.isoformat()}T00:00:00.000Z"
        f" and ContentDate/Start lt {date_to.isoformat()}T23:59:59.000Z"
    )
    resp = requests.get(
        f"{CATALOGUE}/Products",
        params={"$filter": filters, "$top": 100, "$orderby": "ContentDate/Start desc"},
        timeout=60,
    )
    resp.raise_for_status()
    products = resp.json().get("value", [])
    return products[:max_items]


def _download_node(token: str, node_uri: str, dest: Path) -> None:
    """Download a single file node via the $value endpoint."""
    url = node_uri.rstrip("/") + "/$value"
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=180) as resp:
        if resp.status_code == 403:
            raise PermissionError(
                "CDSE download forbidden (DAT-ZIP-608). For Landsat products the USGS "
                "Collection 2 data agreement must be accepted in the Copernicus Browser "
                "at https://browser.dataspace.copernicus.eu (select a Landsat scene -> "
                "download -> accept terms), and ESA/Sentinel downloads require the "
                "Copernicus Terms of Use."
            )
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def download_landsat_bands(
    auth: CdseAuth,
    product: dict,
    out_dir: Path,
) -> dict[str, Path]:
    """Download green (B3), SWIR1 (B6), QA_PIXEL and MTL of one Landsat product.

    Returns {"green": Path, "swir16": Path, "qa_pixel": Path, "mtl": Path}.
    Skips files already cached.
    """
    product_id = product["Id"]
    name = product["Name"]
    product_dir = out_dir / "landsat" / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        "green": "_B3.TIF",
        "swir16": "_B6.TIF",
        "qa_pixel": "_QA_PIXEL.TIF",
        "mtl": "_MTL.txt",
    }
    cached: dict[str, Path] = {}
    for key, suffix in wanted.items():
        candidate = product_dir / f"{key}{suffix}"
        if candidate.exists() and candidate.stat().st_size > 1000:
            cached[key] = candidate
    if set(cached) == set(wanted):
        return cached

    token = auth.get_token()
    root = f"{_product_url(product_id)}/Nodes({name})"
    resp = requests.get(root + "/Nodes", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()

    node_map: dict[str, str] = {}
    for node in resp.json().get("result", []):
        node_name = node.get("Name", "")
        for key, suffix in wanted.items():
            if node_name.endswith(suffix):
                node_map[key] = root + "/Nodes(" + node_name + ")"
                break

    for key, suffix in wanted.items():
        if key in cached:
            continue
        uri = node_map.get(key)
        if not uri:
            log.warning("missing %s file in %s", key, name)
            continue
        dest = product_dir / f"{key}{suffix}"
        log.info("Downloading %s for %s", dest.name, name)
        _download_node(token, uri, dest)
        cached[key] = dest
    return cached


def _product_url(product_id: str) -> str:
    return f"{DOWNLOAD}/Products({product_id})"


def _download_annotation(auth: CdseAuth, product: dict, product_dir: Path) -> Path | None:
    """Download the VV annotation XML (geolocation grid) of a GRD product."""
    dest = product_dir / "vv.xml"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    product_dir.mkdir(parents=True, exist_ok=True)
    token = auth.get_token()
    root = f"{_product_url(product['Id'])}/Nodes({product['Name']})"
    resp = requests.get(root + "/Nodes(annotation)/Nodes", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    leaf = None
    for node in resp.json().get("result", []):
        node_name = node.get("Name", "")
        if "grd-vv-" in node_name and node_name.endswith(".xml"):
            leaf = node_name
            break
    if not leaf:
        log.warning("no VV annotation XML found in %s", product["Name"])
        return None
    url = f"{root}/Nodes(annotation)/Nodes({leaf})/$value"
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def _download_calibration(auth: CdseAuth, product: dict, product_dir: Path, pol: str) -> Path | None:
    """Download the calibration XML (per-line sigmaNought) for one polarization."""
    dest = product_dir / f"cal-{pol}.xml"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    token = auth.get_token()
    root = f"{_product_url(product['Id'])}/Nodes({product['Name']})"
    resp = requests.get(
        root + "/Nodes(annotation)/Nodes(calibration)/Nodes",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    leaf = None
    for node in resp.json().get("result", []):
        node_name = node.get("Name", "")
        if node_name.startswith("calibration-") and f"-{pol}-" in node_name and node_name.endswith(".xml"):
            leaf = node_name
            break
    if not leaf:
        log.warning("no %s calibration XML found in %s", pol, product["Name"])
        return None
    url = f"{root}/Nodes(annotation)/Nodes(calibration)/Nodes({leaf})/$value"
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def sigma0_factors(xml_path: Path, row_off: int, n_rows: int) -> np.ndarray:
    """Per-row multiplicative factor turning measurement DN into sigma0 (linear).

    sigma0 = DN / sigmaNought(line). sigmaNought varies only mildly across the
    swath (range), so we use its per-line median, interpolated over absolute
    line indices ``[row_off, row_off + n_rows)``.
    """
    lines, sn = [], []
    for vec in ET.parse(xml_path).getroot().iter("calibrationVector"):
        line_el = vec.find("line")
        sn_el = vec.find("sigmaNought")
        if line_el is None or sn_el is None or sn_el.text is None:
            continue
        vals = np.asarray([float(x) for x in sn_el.text.split()])
        if vals.size == 0:
            continue
        lines.append(int(line_el.text))
        sn.append(float(np.median(vals)))
    if not lines:
        log.warning("no calibration vectors in %s", xml_path.name)
        return np.ones(n_rows, dtype="float32")
    cal_lines = np.asarray(lines)
    cal_sn = np.asarray(sn)
    row_idx = np.arange(row_off, row_off + n_rows, dtype="float64")
    factor = 1.0 / np.interp(row_idx, cal_lines, cal_sn)
    return factor.astype("float32")


def georef_from_annotation(xml_path: Path, anchor_lonlat: tuple[float, float] | None = None) -> Affine | None:
    """Build a local affine (EPSG:4326) from the S1 annotation geolocation grid.

    The measurement GeoTIFFs ship un-georeferenced; the annotation XML carries a
    sparse geolocation grid (pixel/line -> lon/lat). We fit a degree-2 polynomial
    in (pixel, line) for lon and lat, then linearize it at the scene center
    (or ``anchor_lonlat`` if given) to get a local affine valid around that point.
    """
    points = []
    for gp in ET.parse(xml_path).getroot().iter("geolocationGridPoint"):
        def _t(tag: str) -> float:
            el = gp.find(tag)
            return float(el.text) if el is not None and el.text else float("nan")

        line, pixel = _t("line"), _t("pixel")
        lat, lon = _t("latitude"), _t("longitude")
        if all(np.isfinite(x) for x in (line, pixel, lat, lon)):
            points.append((pixel, line, lon, lat))
    if len(points) < 9:
        log.warning("geolocation grid too sparse in %s (%d pts)", xml_path.name, len(points))
        return None
    arr = np.asarray(points)
    px, ln, lo, la = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    # Fit in normalized (pixel, line) space: raw coords span 0..25000+, so a
    # degree-2 design matrix on raw values is catastrophically ill-conditioned.
    pmean, pstd = px.mean(), px.std()
    lmean, lstd = ln.mean(), ln.std()
    px_n, ln_n = (px - pmean) / pstd, (ln - lmean) / lstd
    design = np.column_stack([np.ones_like(px_n), px_n, ln_n, px_n * px_n, ln_n * ln_n, px_n * ln_n])
    clon = np.linalg.lstsq(design, lo, rcond=None)[0]
    clat = np.linalg.lstsq(design, la, rcond=None)[0]

    def eval_poly(c: np.ndarray, p_n: float, l_n: float) -> float:
        return c[0] + c[1] * p_n + c[2] * l_n + c[3] * p_n * p_n + c[4] * l_n * l_n + c[5] * p_n * l_n

    p0_n, l0_n = 0.0, 0.0
    if anchor_lonlat is not None:
        # Iterate the linearization anchor to the AOI center: solve the local
        # linear system at (p0_n, l0_n) for the (pixel, line) of the AOI center,
        # then rebuild the linearization exactly at that anchor.
        lon_a, lat_a = anchor_lonlat
        lon0, lat0 = eval_poly(clon, p0_n, l0_n), eval_poly(clat, p0_n, l0_n)
        a0 = clon[1] + 2 * clon[3] * p0_n + clon[5] * l0_n   # dlon/dp'
        b0 = clon[2] + 2 * clon[4] * l0_n + clon[5] * p0_n   # dlon/dl'
        d0 = clat[1] + 2 * clat[3] * p0_n + clat[5] * l0_n   # dlat/dp'
        e0 = clat[2] + 2 * clat[4] * l0_n + clat[5] * p0_n   # dlat/dl'
        det = a0 * e0 - b0 * d0
        if abs(det) > 1e-15:
            rx, ry = lon_a - lon0, lat_a - lat0
            p0_n = p0_n + (rx * e0 - b0 * ry) / det
            l0_n = l0_n + (a0 * ry - d0 * rx) / det

    p0, l0 = pmean + p0_n * pstd, lmean + l0_n * lstd
    lon0, lat0 = eval_poly(clon, p0_n, l0_n), eval_poly(clat, p0_n, l0_n)
    a = (clon[1] + 2 * clon[3] * p0_n + clon[5] * l0_n) / pstd
    b = (clon[2] + 2 * clon[4] * l0_n + clon[5] * p0_n) / lstd
    c = lon0 - a * p0 - b * l0
    d = (clat[1] + 2 * clat[3] * p0_n + clat[5] * l0_n) / pstd
    e = (clat[2] + 2 * clat[4] * l0_n + clat[5] * p0_n) / lstd
    f = lat0 - d * p0 - e * l0
    affine = Affine(a, b, c, d, e, f)
    if not (32.0 <= c <= 39.0 and 6.0 <= f <= 18.0) or abs(a) > 1e-4 or abs(d) > 1e-4:
        log.warning("georef fit implausible in %s (a=%.3g d=%.3g c=%.3f f=%.3f) — dropping",
                    xml_path.name, a, d, c, f)
        return None
    return affine


def download_measurement_tiffs(
    auth: CdseAuth,
    product: dict,
    out_dir: Path,
    bands: tuple[str, ...] = ("vv", "vh"),
) -> dict[str, Path]:
    """Download VV/VH measurement GeoTIFFs of one GRD product into out_dir.

    Returns {band: local_path}. Skips bands already cached.
    """
    product_id = product["Id"]
    name = product["Name"]
    product_dir = out_dir / "s1" / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    token = auth.get_token()
    cached: dict[str, Path] = {}
    for band in bands:
        candidate = product_dir / f"{band}.tiff"
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            cached[band] = candidate

    if set(cached) == set(bands):
        annotation = _download_annotation(auth, product, product_dir)
        if annotation is not None:
            cached["annotation"] = annotation
        for pol in bands:
            cal = _download_calibration(auth, product, product_dir, pol)
            if cal is not None:
                cached[f"cal_{pol}"] = cal
        return cached

    root = f"{_product_url(product_id)}/Nodes({name})"
    log.info("Walking node tree for %s", name)
    resp = requests.get(root + "/Nodes", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    measurement_uri = None
    for node in resp.json().get("result", []):
        if node.get("Name") == "measurement":
            measurement_uri = node.get("Nodes", {}).get("uri")
    if not measurement_uri:
        raise RuntimeError(f"no measurement node in {name}")

    resp = requests.get(
        measurement_uri,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    leaves = {}
    for node in resp.json().get("result", []):
        leaf_name = node.get("Name", "")
        m = BAND_POL_RE.search(leaf_name)
        if m and m.group(1) in bands:
            leaves[m.group(1)] = leaf_name

    for band in bands:
        if band in cached:
            continue
        leaf_name = leaves.get(band)
        if not leaf_name:
            log.warning("band %s missing in %s", band, name)
            continue
        leaf_base = measurement_uri.rsplit("/Nodes", 1)[0]
        url = f"{leaf_base}/Nodes({leaf_name})/$value"
        dest = product_dir / f"{band}.tiff"
        log.info("Downloading %s (%s) — this is large (~0.8 GB), cached afterwards", leaf_name, band)
        with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        cached[band] = dest

    annotation = _download_annotation(auth, product, product_dir)
    if annotation is not None:
        cached["annotation"] = annotation
    for pol in bands:
        cal = _download_calibration(auth, product, product_dir, pol)
        if cal is not None:
            cached[f"cal_{pol}"] = cal
    return cached


def _calibrate_scene_bands(scene, band_arrays: dict[str, np.ndarray], meta) -> None:
    """Convert raw S1 measurement DN to sigma0 (linear) using the calibration XMLs.

    Mutates ``band_arrays`` in place. ``meta`` is the read window transform.
    """
    col_off, row_off = (~Affine(*scene.georef_override()[0])) * (meta[0].c, meta[0].f)
    row_off = int(round(row_off))
    for pol in ("vv", "vh"):
        key = f"cal_{pol}"
        cal_path = scene.assets.get(key)
        if pol not in band_arrays or not cal_path:
            continue
        factors = sigma0_factors(Path(cal_path), row_off, band_arrays[pol].shape[0])
        band_arrays[pol] = band_arrays[pol] * factors[:, None]


def scene_aoi_coverage(auth: CdseAuth, product: dict, out_dir: Path, bbox_4326: tuple[float, float, float, float], anchor_lonlat: tuple[float, float]) -> float | None:
    """Fraction of the AOI bbox inside the GRD image, without downloading bands.

    Uses only the small annotation XML. Returns None if the geolocation grid
    cannot be built (product unusable). Used to skip sliver scenes before the
    ~1.7 GB measurement download.
    """
    product_dir = out_dir / "s1" / product["Id"]
    annotation = _download_annotation(auth, product, product_dir)
    if annotation is None:
        return None
    tf = georef_from_annotation(annotation, anchor_lonlat=anchor_lonlat)
    if tf is None:
        return None
    width, height = 25177, 16784
    inv = ~tf
    cols = [inv * p for p in ((bbox_4326[0], bbox_4326[3]), (bbox_4326[2], bbox_4326[3]), (bbox_4326[0], bbox_4326[1]), (bbox_4326[2], bbox_4326[1]))]
    col_off = min(c[0] for c in cols)
    row_off = min(c[1] for c in cols)
    col_max = max(c[0] for c in cols)
    row_max = max(c[1] for c in cols)
    win_w = max(0.0, min(col_max, width) - max(col_off, 0.0))
    win_h = max(0.0, min(row_max, height) - max(row_off, 0.0))
    total = max(1e-9, (col_max - col_off) * (row_max - row_off))
    return float(min(1.0, win_w * win_h / total))


def to_scenes(products: list[dict], local_root: Path) -> list[dict]:
    """Map CDSE products to lightweight scene descriptors with local paths."""
    scenes = []
    for product in products:
        scenes.append(
            {
                "id": product["Id"],
                "name": product["Name"],
                "date": dt.datetime.fromisoformat(product["ContentDate"]["Start"].replace("Z", "+00:00")).date(),
                "assets": {"product_dir": str(local_root / "s1" / product["Id"])},
                "product": product,
            }
        )
    return scenes
