import unittest

from r2d2_rl.tests import BASE_CONFIG_PATH

import cv2
import numpy as np

from r2d2_rl.hybrid_control_rl.config import load_yaml_config
from r2d2_rl.perception.color_block_detector import ColorBlockDetector


class ColorDetectorSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_yaml_config(str(BASE_CONFIG_PATH))

    def test_detects_single_red_block_centroid(self):
        image = np.zeros((160, 220, 3), dtype=np.uint8)
        image[:] = np.array([184, 173, 169], dtype=np.uint8)
        cv2.rectangle(image, (60, 40), (110, 90), (255, 0, 0), thickness=-1)

        detector = ColorBlockDetector(self.config)
        detections = detector.detect(image, target_color="red")

        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.color, "red")
        np.testing.assert_allclose(det.centroid_uv, np.array([85.0, 65.0]), atol=1.0)
        self.assertGreater(det.area_px, 2000)
        self.assertGreater(det.confidence, 0.5)
        self.assertEqual(det.covariance_uv.shape, (2, 2))

    def test_filters_small_components(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[:] = np.array([184, 173, 169], dtype=np.uint8)
        cv2.circle(image, (40, 40), 2, (0, 255, 0), thickness=-1)

        detector = ColorBlockDetector(self.config)
        detections = detector.detect(image, target_color="green")

        self.assertEqual(detections, [])

    def test_target_color_filters_other_colors(self):
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        image[:] = np.array([184, 173, 169], dtype=np.uint8)
        cv2.rectangle(image, (20, 50), (70, 100), (255, 0, 0), thickness=-1)
        cv2.rectangle(image, (140, 50), (190, 100), (0, 0, 255), thickness=-1)

        detector = ColorBlockDetector(self.config)
        detections = detector.detect(image, target_color="blue")

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].color, "blue")
        np.testing.assert_allclose(detections[0].centroid_uv, np.array([165.0, 75.0]), atol=1.0)

    def test_detects_all_configured_colors(self):
        image = np.zeros((180, 260, 3), dtype=np.uint8)
        image[:] = np.array([184, 173, 169], dtype=np.uint8)
        cv2.rectangle(image, (20, 50), (60, 90), (255, 0, 0), thickness=-1)
        cv2.rectangle(image, (85, 50), (125, 90), (0, 255, 0), thickness=-1)
        cv2.rectangle(image, (150, 50), (190, 90), (0, 0, 255), thickness=-1)
        cv2.rectangle(image, (210, 50), (250, 90), (255, 255, 0), thickness=-1)

        detector = ColorBlockDetector(self.config)
        detections = detector.detect(image)
        colors = {det.color for det in detections}

        self.assertTrue({"red", "green", "blue", "yellow"}.issubset(colors))

    def test_unknown_target_color_raises(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        detector = ColorBlockDetector(self.config)

        with self.assertRaises(ValueError):
            detector.detect(image, target_color="chartreuse")

    def test_non_uint8_image_raises(self):
        image = np.zeros((80, 80, 3), dtype=np.float32)
        detector = ColorBlockDetector(self.config)

        with self.assertRaises(ValueError):
            detector.detect(image, target_color="red")


if __name__ == "__main__":
    unittest.main()
