"""Goldmark tissue mask (classical, whole-slide thumbnail).

Vendored from GOLDMARK / MSK SlideTileExtractor
(https://github.com/chadvanderbilt/GOLDMARK). Marker detection, PIL blur, Otsu
(pure-white pixels masked), low-std rejection, and max-pooling. Native tile-grid
generation is omitted: TRIDENT consumes GeoJSON contours and runs its own coords
stage.

Polygon vertices follow the outer sides of occupied mask cells (occupancy
outline), not OpenCV pixel centers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import geopandas as gpd
import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, label

# Occupancy-grid defaults for the *tissue mask*, not encoder tiles.
# Do NOT change MASK_SIZE / TILE_SIZE to match UNI (256), Virchow (224), etc.
# Downstream patch size is set at coords/feat via `--patch_size` / `--mag`
# (e.g. UNI: 256@20x, Virchow: 224@20x). These 224s are the original MSK
# extractor defaults; TILE_SIZE is only used to scale min_cc_size.
DEFAULT_TILE_SIZE = 224
DEFAULT_MASK_SIZE = 224
DEFAULT_MPP = 0.5
DEFAULT_GRID_MULT = 4
DEFAULT_MIN_CC_SIZE = 10


def thumbnail_wh(
    width: int,
    height: int,
    size: int,
    mpp: float,
    base_mpp: float,
    mult: int,
) -> tuple[int, int]:
    """Thumbnail size used by ``threshold``: one cell per ``size`` pixels at ``mpp``."""
    cell = size * mpp / base_mpp
    w = int(np.round(width / cell)) * mult
    h = int(np.round(height / cell)) * mult
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Goldmark thumbnail size collapsed to {(w, h)} for slide "
            f"{width}x{height} at mask_size={size}, mpp={mpp}, base_mpp={base_mpp}, mult={mult}."
        )
    return w, h


def image2array(img: Image.Image) -> np.ndarray:
    if img.mode in {"RGB", "RGBA"}:
        arr = np.array(img)
        return np.uint8(arr[:, :, :3])
    raise ValueError("Unsupported image mode")


def detect_marker(thumb: np.ndarray, mult: float) -> Optional[np.ndarray]:
    blurred = Image.fromarray(thumb).filter(ImageFilter.GaussianBlur(radius=1.0))
    img = np.asarray(blurred).astype(np.int16)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    black_marker = (r < 125) & (g < 125) & (b < 125)
    blue_marker = (b > 80) & (b > r + 25) & (b > g + 10)
    green_marker = (g > 80) & (g > r + 20) & (g > b + 5)
    mask = black_marker | blue_marker | green_marker
    if np.count_nonzero(mask) == 0:
        return None
    for _ in range(max(1, int(max(1, mult)))):
        mask = binary_erosion(mask)
    for _ in range(max(1, int(max(1, mult)) * 3)):
        mask = binary_dilation(mask)
    return (mask.astype(np.uint8) * 255) if np.count_nonzero(mask) > 0 else None


def filter_regions(img: np.ndarray, min_size: int) -> np.ndarray:
    labeled, n_regions = label(img, return_num=True)
    for idx in range(1, n_regions + 1):
        if labeled[labeled == idx].size < min_size:
            labeled[labeled == idx] = 0
    return labeled


def occupancy_polygons(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Outer occupancy outline: pixel (x, y) covers the square [x, x+1] x [y, y+1].

    Internal edges cancel, so the remaining rings follow the sides of occupied
    cells rather than OpenCV's top-left pixel indices. Hole rings are dropped
    (same fill behavior as RETR_EXTERNAL). Vertices are collinear-simplified.
    """
    binary = np.asarray(mask) > 0
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add_edge(p: tuple[int, int], q: tuple[int, int]) -> None:
        rev = (q, p)
        if rev in edges:
            edges.remove(rev)
        else:
            edges.add((p, q))

    ys, xs = np.nonzero(binary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        x_i = int(x)
        y_i = int(y)
        add_edge((x_i, y_i), (x_i + 1, y_i))
        add_edge((x_i + 1, y_i), (x_i + 1, y_i + 1))
        add_edge((x_i + 1, y_i + 1), (x_i, y_i + 1))
        add_edge((x_i, y_i + 1), (x_i, y_i))

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        outgoing[a].append(b)

    unused = set(edges)
    rings: list[list[tuple[int, int]]] = []
    while unused:
        start, _ = next(iter(unused))
        ring = [start]
        curr = start
        while True:
            nxt = None
            for cand in outgoing[curr]:
                if (curr, cand) in unused:
                    nxt = cand
                    break
            if nxt is None:
                break
            unused.remove((curr, nxt))
            if nxt == start:
                break
            ring.append(nxt)
            curr = nxt
        if len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        area = 0
        for i in range(len(ring) - 1):
            x1, y1 = ring[i]
            x2, y2 = ring[i + 1]
            area += x1 * y2 - x2 * y1
        if area <= 0:
            continue
        ring = list(reversed(ring))
        rings.append(_collapse_collinear(ring))
    return rings


def _collapse_collinear(ring: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(ring) < 4:
        return ring
    body = ring[:-1]
    n = len(body)
    kept: list[tuple[int, int]] = []
    for i in range(n):
        prev = body[i - 1]
        curr = body[i]
        nxt = body[(i + 1) % n]
        if (curr[0] - prev[0]) * (nxt[1] - curr[1]) == (curr[1] - prev[1]) * (nxt[0] - curr[0]):
            continue
        kept.append(curr)
    if len(kept) < 3:
        return ring
    kept.append(kept[0])
    return kept


def occupancy_polygons_to_gdf(
    rings: list[list[tuple[int, int]]],
    scale_x: float,
    scale_y: float,
) -> gpd.GeoDataFrame:
    """Scale occupancy rings from mask pixels to level-0 coordinates."""
    rows = []
    tissue_id = 0
    for ring in rings:
        if len(ring) < 4:
            continue
        coords = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in ring]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        geoms = []
        if polygon.geom_type == "Polygon":
            geoms = [polygon]
        elif polygon.geom_type == "MultiPolygon":
            geoms = list(polygon.geoms)
        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            rows.append({"tissue_id": tissue_id, "geometry": geom})
            tissue_id += 1
    if not rows:
        return gpd.GeoDataFrame({"tissue_id": [], "geometry": []}, geometry="geometry")
    return gpd.GeoDataFrame(rows, geometry="geometry")


def threshold_rgb(
    img_c: np.ndarray,
    marker_mult: float,
    pool_mult: int,
) -> tuple[np.ndarray, float]:
    """Binary tissue mask from an RGB thumbnail.

    ``pool_mult`` > 1 max-pools ``(mult x mult)`` blocks so the returned mask
    has one cell per mask-size block at the working MPP.
    """
    if img_c.ndim != 3 or img_c.shape[2] < 3:
        raise ValueError(f"Expected RGB array (H, W, 3), got shape={img_c.shape}")
    h, w = img_c.shape[:2]
    std = np.std(img_c, axis=-1)
    img_g = np.asarray(
        0.299 * img_c[:, :, 0] + 0.587 * img_c[:, :, 1] + 0.114 * img_c[:, :, 2],
        dtype=np.uint8,
    )
    marker = detect_marker(img_c, marker_mult)
    img_g = np.asarray(
        Image.fromarray(img_g).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.uint8,
    )
    t = 0.0
    if marker is not None:
        masked = np.ma.masked_array(img_g, (marker > 0) | (img_g == 255))
        compressed = masked.compressed()
        if compressed.size:
            t = float(threshold_otsu(compressed))
            img_g = np.where(img_g > t, 255, 0).astype(np.uint8)
            img_g = np.clip(
                (255 - img_g).astype(np.int16) - marker.astype(np.int16), 0, 255
            ).astype(np.uint8)
        else:
            img_g = np.zeros_like(img_g)
    else:
        masked = np.ma.masked_array(img_g, img_g == 255)
        compressed = masked.compressed()
        if compressed.size:
            t = float(threshold_otsu(compressed))
            img_g = np.where(img_g > t, 255, 0).astype(np.uint8)
            img_g = 255 - img_g
        else:
            img_g = np.zeros_like(img_g)
    img_g[std < 5] = 0
    if pool_mult > 1:
        if h % pool_mult or w % pool_mult:
            raise ValueError(
                f"Thumbnail {(h, w)} is not divisible by grid mult {pool_mult}."
            )
        img_g = img_g.reshape(h // pool_mult, pool_mult, w // pool_mult, pool_mult).max(
            axis=(1, 3)
        )
    return img_g, t
