from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from masking_ita import (  # noqa: E402
    MaskingITAConfig,
    MaskingITAProcessor,
    MaskingITAResult,
    ProcessingStatus,
)
from run_member1_batch import (  # noqa: E402
    DISEASE_LABELS,
    archive_lineage_id,
    build_pilot_manifest,
    quality_metrics,
    read_csv,
    run_phase0,
    signature_correlation,
    write_csv,
)


class ColourAndITATests(unittest.TestCase):
    def test_opencv_lab_is_corrected_to_cielab_ranges(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        lab = MaskingITAProcessor.rgb_to_opencv_lab(image)
        self.assertAlmostEqual(float(lab[0, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(lab[0, 1, 0]), 100.0, places=4)
        self.assertAlmostEqual(float(lab[0, 0, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(lab[0, 0, 2]), 0.0, places=4)

    def test_ita_formula_brackets_and_zero_guard(self):
        processor = MaskingITAProcessor()
        expected = float(np.degrees(np.arctan((60.0 - 50.0) / 20.0)))
        self.assertAlmostEqual(processor.calculate_ita(60.0, 20.0), expected)
        self.assertEqual(processor.assign_bracket(27.999), "Darkest")
        self.assertEqual(processor.assign_bracket(28.0), "Medium")
        self.assertEqual(processor.assign_bracket(41.0), "Medium")
        self.assertEqual(processor.assign_bracket(41.001), "Lightest")
        with self.assertRaisesRegex(ValueError, "ITA_UNSTABLE_B_ZERO"):
            processor.calculate_ita(60.0, 0.0009)


class MaskingTests(unittest.TestCase):
    def permissive_config(self):
        return MaskingITAConfig(
            k_selection_max_side=64, max_k_selection_pixels=4096,
            max_silhouette_samples=1000,
            l_min=0, l_max=100, a_min=-128, a_max=127, b_min=-128, b_max=127,
            shadow_l_threshold=-1, highlight_l_threshold=101,
            maximum_shadow_fraction=1, maximum_highlight_fraction=1,
            minimum_pixel_plausibility_fraction=1,
            mask_area_min_percent=1, mask_area_max_percent=99,
            minimum_valid_pixel_count=1,
            minimum_short_side=1, minimum_laplacian_variance=0,
            maximum_clipped_fraction=1, minimum_luminance_span=0,
        )

    def test_dynamic_k_is_reproducible_and_returns_original_mask(self):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        image[:, :30] = [185, 125, 95]
        image[:20, 30:] = [40, 50, 60]
        image[20:, 30:] = [230, 210, 190]
        processor = MaskingITAProcessor(self.permissive_config())
        first = processor.process_image(image)
        second = processor.process_image(image)
        self.assertTrue(first.calibration_eligible)
        self.assertEqual(first.selected_k, second.selected_k)
        self.assertAlmostEqual(first.ita, second.ita)
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertEqual(first.mask.shape, image.shape[:2])
        self.assertEqual(set(first.k_scores), {"2", "3", "4", "5"})

    def test_fallback_mask_is_mapped_to_full_image(self):
        processor = MaskingITAProcessor(self.permissive_config())
        failed = MaskingITAResult("x", ProcessingStatus.MASK_FAILED.value, failure_reason="bad")
        accepted = MaskingITAResult(
            "x", ProcessingStatus.MASK_FALLBACK.value, "center_weighted_crop",
            selected_k=3, mean_l=60, mean_b=20, ita=26.5, bracket="Darkest",
            calibration_eligible=True, mask=np.ones((6, 6), dtype=bool),
        )
        with patch.object(processor, "_attempt", side_effect=[failed, accepted]):
            result = processor.process_image(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result.mask.shape, (10, 10))
        self.assertEqual(int(result.mask.sum()), 36)
        self.assertEqual(result.primary_failure_reason, "bad")

    def test_double_failure_never_returns_ita(self):
        processor = MaskingITAProcessor(self.permissive_config())
        failed = MaskingITAResult("x", ProcessingStatus.MASK_FAILED.value, failure_reason="bad")
        with patch.object(processor, "_attempt", side_effect=[failed, failed]):
            result = processor.process_image(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(result.calibration_eligible)
        self.assertIsNone(result.ita)
        self.assertEqual(result.user_message, "No plausible skin detected; please retake image.")


class StandardizationAndPrepTests(unittest.TestCase):
    QUALITY = {
        "minimum_short_side": 256, "minimum_laplacian_variance": 50,
        "black_luma_max": 5, "white_luma_min": 250,
        "maximum_clipped_fraction": 0.30, "minimum_luminance_span": 20,
    }

    def test_resolution_quality_gate(self):
        image = np.full((255, 400, 3), 128, dtype=np.uint8)
        status, reason, _ = quality_metrics(image, self.QUALITY)
        self.assertEqual((status, reason), ("REJECTED", "RESOLUTION_TOO_LOW"))

    def test_exif_orientation_and_alpha_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jpeg = root / "oriented.jpg"
            image = Image.new("RGB", (20, 10), "red")
            exif = image.getexif(); exif[274] = 6
            image.save(jpeg, exif=exif)
            loaded = MaskingITAProcessor.load_image(jpeg)
            self.assertEqual(loaded.shape[:2], (20, 10))
            transparent = root / "transparent.png"
            Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(transparent)
            with self.assertRaisesRegex(ValueError, "NON_OPAQUE_IMAGE"):
                MaskingITAProcessor.load_image(transparent)

    def test_archive_lineage_groups_roboflow_derivatives(self):
        first = archive_lineage_id("FU-athlete-foot-1-_png.rf.abc123.jpg")
        second = archive_lineage_id("FU-athlete-foot-1-_jpeg.rf.def456.jpg")
        self.assertEqual(first, second)

    def test_similarity_requires_identical_visual_direction(self):
        left = np.arange(64, dtype=np.float32)
        self.assertAlmostEqual(signature_correlation(left, left.copy()), 1.0, places=6)
        self.assertLess(signature_correlation(left, left[::-1].copy()), 0)

    def test_pilot_builder_creates_160_review_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for disease in DISEASE_LABELS:
                for index in range(20):
                    rows.append({
                        "image_id": f"{disease}-{index}", "disease_label": disease,
                        "source_repository": f"source-{index % 2}", "data_origin": "public",
                        "split": "train", "image_quality_status": "PASS",
                        "duplicate_status": "UNIQUE", "proxy_skin_tone_group": "",
                    })
            source = root / "prepared.csv"
            output = root / "pilot.csv"
            write_csv(source, rows)
            config = Path(__file__).resolve().parents[1] / "backend" / "member1_phase0_config.json"
            build_pilot_manifest(source, output, config)
            pilot = read_csv(output)
            self.assertEqual(len(pilot), 160)
            self.assertEqual({row["disease_label"] for row in pilot}, DISEASE_LABELS)
            self.assertTrue(all("consensus_mask_acceptable" in row for row in pilot))

    def test_phase0_batch_exports_member2_compatible_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, disease in enumerate(sorted(DISEASE_LABELS)):
                rows.append({
                    "image_id": f"id-{index}", "processed_image_path": str(root / f"{index}.png"),
                    "disease_label": disease, "source_repository": "source",
                    "data_origin": "public", "split": "train",
                    "image_quality_status": "PASS", "duplicate_status": "UNIQUE",
                    "proxy_skin_tone_group": "A",
                })
            manifest = root / "prepared.csv"
            write_csv(manifest, rows)
            source_config = Path(__file__).resolve().parents[1] / "backend" / "member1_phase0_config.json"
            config_data = json.loads(source_config.read_text(encoding="utf-8"))
            config_data["status"] = "FROZEN"
            config = root / "config.json"
            config.write_text(json.dumps(config_data), encoding="utf-8")
            result = MaskingITAResult(
                filename="x.png", status=ProcessingStatus.MASK_SUCCESS.value,
                mask_method="primary", selected_k=3, silhouette_score=0.5,
                mask_area_pixels=4, mask_area_percent=100, mean_l=60,
                mean_a=10, mean_b=20, ita=26.565, bracket="Darkest",
                calibration_eligible=True, mask=np.ones((2, 2), dtype=bool),
            )
            with patch.object(MaskingITAProcessor, "process_file", return_value=result):
                output = run_phase0(manifest, root / "output", config)
            exported = read_csv(output)
            self.assertEqual(len(exported), 8)
            self.assertTrue(all(row["mask_path"] for row in exported))
            self.assertTrue(all(row["ITA"] == row["ita"] for row in exported))
            self.assertTrue((root / "output" / "frozen_member1_config.json").exists())


if __name__ == "__main__":
    unittest.main()
