"""Regression tests; QR fixtures are generated in memory, no network needed."""

import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import zxingcpp

import Do_an_XLA as xla


def barcode(text="XLA", format=zxingcpp.BarcodeFormat.QRCode):
    return np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(text, format), scale=8
    )).copy()


def detection(text="XLA", engine="ZXing", x=10, y=10, size=60):
    return {"text": text, "engine": engine,
            "points": np.array([[x, y], [x + size, y],
                                [x + size, y + size], [x, y + size]], np.float32)}


class GeometryTests(unittest.TestCase):
    def test_diamond_all_corner_permutations(self):
        diamond = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], np.float32)
        for points in itertools.permutations(diamond):
            ordered = xla.order_points(points)
            self.assertEqual(len(np.unique(ordered, axis=0)), 4)
            self.assertEqual(cv2.contourArea(ordered), 5000)

    def test_invalid_geometry_is_rejected(self):
        for value in (None, [], [1, 2, 3], [[0, 0]] * 4,
                      [[0, 0], [1, 0], [2, 0], [3, 0]],
                      [[0, 0], [1, 0], [1, np.nan], [0, 1]]):
            with self.subTest(value=value):
                self.assertIsNone(xla.normalize_quad(value, 100, 100))

    def test_single_character_and_full_image_quad(self):
        result = xla.filter_and_dedup_results([detection("A", x=0, y=0, size=99)], 100, 100)
        self.assertEqual(result[0]["text"], "A")

    def test_payload_whitespace_is_preserved(self):
        result = xla.filter_and_dedup_results([detection("  XLA  ")], 100, 100)
        self.assertEqual(result[0]["text"], "  XLA  ")

    def test_same_payload_at_distinct_locations_is_not_removed(self):
        result = xla.filter_and_dedup_results(
            [detection(), detection(engine="WeChat"), detection(x=150)], 300, 100
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["engine"], "WeChat")

    def test_inverse_rotation_scale_and_translation(self):
        source = detection()
        scale = np.array([[1.6, 0, 0.3], [0, 1.6, 0.3], [0, 0, 1]])
        _, rotation = xla.rotate_bound(np.zeros((160, 320), np.uint8), 45)
        transform = rotation @ scale
        work = {**source, "points": cv2.perspectiveTransform(
            source["points"].reshape(-1, 1, 2), transform
        ).reshape(-1, 2)}
        mapped = xla.map_results([work], transform, 200, 100)
        np.testing.assert_allclose(mapped[0]["points"], source["points"], atol=1e-4)

    def test_rotated_non_square_image_keeps_all_corners(self):
        image = np.full((80, 300), 255, np.uint8)
        corners = np.array([[0, 0], [299, 0], [299, 79], [0, 79]], np.float32)
        for angle in (45, 90, 135, 270):
            work, transform = xla.rotate_bound(image, angle)
            points = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform).reshape(-1, 2)
            self.assertGreaterEqual(points.min(), -1e-4)
            self.assertLessEqual(points[:, 0].max(), work.shape[1])
            self.assertLessEqual(points[:, 1].max(), work.shape[0])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = xla.PipelineMaster(time_budget=3, strong_only=True)

    def test_real_short_qr_uses_raw_path_without_preprocessing(self):
        with patch.object(self.pipeline, "pipeline_preprocess", side_effect=AssertionError("unnecessary preprocessing")):
            results, _, method = self.pipeline.decode_with_adjustment_loop(barcode("A"))
        self.assertEqual([r["text"] for r in results], ["A"])
        self.assertTrue(method.startswith("Raw"))

    def test_real_inverted_and_rotated_qr(self):
        for angle in (0, 45, 90):
            image, _ = xla.rotate_bound(barcode("XLA-ROTATE"), angle)
            results, _, _ = self.pipeline.decode_with_adjustment_loop(255 - image)
            self.assertIn("XLA-ROTATE", [r["text"] for r in results])

    def test_two_identical_physical_codes(self):
        qr = barcode("XLA-SAME")
        image = np.pad(qr, ((30, 30), (30, 30)), constant_values=255)
        image = np.hstack([image, image])
        results, _, _ = self.pipeline.decode_with_adjustment_loop(image)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["text"] == "XLA-SAME" for r in results))

    def test_real_datamatrix(self):
        results, _, _ = self.pipeline.decode_with_adjustment_loop(
            barcode("XLA-DM", zxingcpp.BarcodeFormat.DataMatrix)
        )
        self.assertIn("XLA-DM", [r["text"] for r in results])

    def test_roi_coordinates_are_in_source_image(self):
        qr = barcode("XLA-ROI")
        image = np.pad(qr, ((110, 70), (140, 80)), constant_values=255)
        raw, _ = self.pipeline._decode_all_engines(image)
        original = xla.filter_and_dedup_results(raw, image.shape[1], image.shape[0])[0]
        decoded = self.pipeline._decode_roi(image, original["points"], math.inf)
        self.assertEqual(decoded[0]["text"], "XLA-ROI")
        self.assertGreater(xla.bbox_iou(original["bbox"], decoded[0]["bbox"]), 0.85)

    def test_budget_expiry_skips_preprocessing(self):
        pipeline = xla.PipelineMaster(time_budget=0.001)
        def slow_engine(*args):
            time.sleep(0.005)
            return [], []
        with patch.object(pipeline, "_decode_all_engines", side_effect=slow_engine), \
                patch.object(pipeline, "pipeline_preprocess", side_effect=AssertionError("budget exceeded")):
            results, _, status = pipeline.decode_with_adjustment_loop(np.full((100, 100), 255, np.uint8))
        self.assertEqual(results, [])
        self.assertEqual(status, "time_budget_exceeded")

    def test_libdmtx_bottom_left_coordinates(self):
        result = SimpleNamespace(data=b"DM", rect=SimpleNamespace(left=20, top=30, width=40, height=50))
        pipeline = xla.PipelineMaster()
        pipeline.detector = None
        pipeline.cv_qr = SimpleNamespace(
            detectAndDecodeMulti=lambda _: (False, (), None, None),
            detectAndDecode=lambda _: ("", None, None),
        )
        with patch.object(xla, "decode_dmtx", return_value=[result]), \
                patch.object(xla, "decode_pyzbar", None), \
                patch.object(zxingcpp, "read_barcodes", return_value=[]):
            codes, _ = pipeline._decode_all_engines(np.full((200, 200), 255, np.uint8))
        self.assertEqual(xla.bbox_from_pts(codes[0]["points"]), (20, 119, 60, 169))

    def test_unicode_input_json_report_and_no_gui(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "ảnh tiếng Việt.PNG"
            cv2.imencode(".png", barcode("XLA-FILE"))[1].tofile(path)
            bad = root / "corrupt.png"
            bad.write_bytes(b"invalid image")
            (root / "ignore.txt").write_text("not an image", encoding="utf-8")
            report = root / "results.json"
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(cv2, "imshow", side_effect=AssertionError("GUI")), \
                    patch.object(cv2, "waitKey", side_effect=AssertionError("GUI")):
                summary = self.pipeline.run(root, report)
            self.assertEqual((summary["total"], summary["success"], summary["errors"]), (2, 1, 1))
            self.assertGreater(summary["elapsed_seconds"], 0)
            self.assertLess(summary["elapsed_seconds"], 30)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["success"], 1)
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.pipeline.run(path, report)

    def test_missing_input_and_empty_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "detect.caffemodel").touch()
            self.assertFalse(xla.wechat_models_ok(root))
            self.assertEqual(xla.collect_images(root), [])
            with self.assertRaises(ValueError):
                self.pipeline.run(root)


if __name__ == "__main__":
    unittest.main()
