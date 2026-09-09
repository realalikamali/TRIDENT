import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from trident.segmentation_models import (
    GoldmarkSegmenter,
    is_cpu_segmenter,
    segmentation_model_factory,
)
from trident.segmentation_models.model_zoo.goldmark import (
    occupancy_polygons,
    occupancy_polygons_to_gdf,
    thumbnail_wh,
    threshold_rgb,
)
from trident.wsi_objects.ImageWSI import ImageWSI


def _synthetic_he_rgb(height=512, width=768):
    """Near-white background with a noisy pink tissue rectangle."""
    rng = np.random.default_rng(0)
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    y0, y1, x0, x1 = height // 4, 3 * height // 4, width // 4, 3 * width // 4
    tissue = np.stack(
        [
            rng.integers(150, 200, size=(y1 - y0, x1 - x0)),
            rng.integers(60, 110, size=(y1 - y0, x1 - x0)),
            rng.integers(90, 140, size=(y1 - y0, x1 - x0)),
        ],
        axis=-1,
    ).astype(np.uint8)
    img[y0:y1, x0:x1] = tissue
    return img


class TestGoldmarkSegmenter(unittest.TestCase):
    def test_is_cpu_segmenter(self):
        self.assertTrue(is_cpu_segmenter("goldmark"))
        self.assertTrue(is_cpu_segmenter("otsu"))
        self.assertFalse(is_cpu_segmenter("hest"))

    def test_factory_builds_goldmark_segmenter(self):
        seg = segmentation_model_factory("goldmark")
        self.assertIsInstance(seg, GoldmarkSegmenter)
        self.assertTrue(seg.whole_slide)
        self.assertEqual(seg.mask_size, 224)
        self.assertEqual(seg.grid_mult, 4)
        self.assertEqual(seg.target_mpp, 0.5)

    def test_forward_is_not_the_inference_path(self):
        seg = GoldmarkSegmenter()
        with self.assertRaises(RuntimeError):
            seg(torch.zeros((1, 3, 32, 32)))

    def test_occupancy_polygons_outline_unit_squares(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        rings = occupancy_polygons(mask)
        self.assertEqual(len(rings), 1)
        gdf = occupancy_polygons_to_gdf(rings, scale_x=10.0, scale_y=20.0)
        self.assertEqual(len(gdf), 1)
        minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
        self.assertEqual((minx, miny, maxx, maxy), (10.0, 20.0, 40.0, 60.0))

    def test_threshold_rgb_keeps_dark_tissue(self):
        rgb = _synthetic_he_rgb(64, 64)
        # 64 is divisible by pool_mult=4 → 16x16 mask.
        mask, thresh = threshold_rgb(rgb, marker_mult=4.0, pool_mult=4)
        self.assertEqual(mask.shape, (16, 16))
        self.assertGreater(thresh, 0)
        self.assertGreater(int((mask > 0).sum()), 0)

    def test_thumbnail_wh_matches_original_formula(self):
        w, h = thumbnail_wh(4480, 2240, size=224, mpp=0.5, base_mpp=0.5, mult=4)
        self.assertEqual((w, h), (80, 40))

    def test_segment_tissue_writes_trident_artifacts_and_coords(self):
        rgb = _synthetic_he_rgb(2048, 2048)
        with tempfile.TemporaryDirectory() as tmpdir:
            slide_path = os.path.join(tmpdir, "synthetic.png")
            Image.fromarray(rgb).save(slide_path)
            wsi = ImageWSI(slide_path=slide_path, mpp=0.5, lazy_init=False)
            job_dir = os.path.join(tmpdir, "job")
            seg = segmentation_model_factory("goldmark")
            geojson_path = wsi.segment_tissue(
                segmentation_model=seg,
                target_mag=seg.target_mag,
                job_dir=job_dir,
                device="cpu",
            )
            self.assertTrue(os.path.isfile(geojson_path))
            self.assertTrue(os.path.isfile(os.path.join(job_dir, "thumbnails", "synthetic.jpg")))
            self.assertTrue(os.path.isfile(os.path.join(job_dir, "contours", "synthetic.jpg")))
            self.assertGreater(len(wsi.gdf_contours), 0)

            coords_path = wsi.extract_tissue_coords(
                target_mag=20,
                patch_size=256,
                save_coords=job_dir,
                overlap=0,
                min_tissue_proportion=0.0,
            )
            self.assertTrue(os.path.isfile(coords_path))
            from trident.IO import read_coords
            _attrs, coords = read_coords(coords_path)
            self.assertGreater(len(coords), 0)


if __name__ == "__main__":
    unittest.main()
