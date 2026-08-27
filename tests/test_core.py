import unittest

import numpy as np

from glcmfeatures import CORE_FEATURES, compute_glcm, glcm_features, quantize


class TestQuantize(unittest.TestCase):
    def test_range_and_dtype(self):
        img = np.array([[0, 10], [50, 100]], dtype=float)
        q = quantize(img, levels=16)
        self.assertEqual(q.dtype, np.uint8)
        self.assertGreaterEqual(q.min(), 0)
        self.assertLessEqual(q.max(), 15)

    def test_fixed_range_is_comparable_across_windows(self):
        # Two windows with different local ranges but quantized against
        # the same fixed vmin/vmax should map the same raw value to the
        # same grey level.
        w1 = np.array([[10, 20], [30, 40]], dtype=float)
        w2 = np.array([[10, 60], [70, 80]], dtype=float)
        q1 = quantize(w1, levels=32, vmin=0, vmax=100)
        q2 = quantize(w2, levels=32, vmin=0, vmax=100)
        self.assertEqual(q1[0, 0], q2[0, 0])  # both have raw value 10

    def test_rejects_bad_range(self):
        with self.assertRaises(ValueError):
            quantize(np.zeros((2, 2)), vmin=5, vmax=5)


class TestGLCMFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.window = rng.integers(0, 256, size=(24, 24)).astype(float)

    def test_returns_all_core_features(self):
        feats = glcm_features(self.window, levels=32)
        self.assertEqual(set(feats.keys()), set(CORE_FEATURES))
        for name, value in feats.items():
            self.assertIsInstance(value, float)
            self.assertFalse(np.isnan(value), f"{name} is NaN")

    def test_uniform_image_has_zero_contrast_and_entropy(self):
        flat = np.full((20, 20), 7.0)
        feats = glcm_features(flat, levels=8)
        self.assertAlmostEqual(feats["contrast"], 0.0)
        self.assertAlmostEqual(feats["entropy"], 0.0)

    def test_subset_of_features(self):
        feats = glcm_features(self.window, levels=16, features=("contrast", "entropy"))
        self.assertEqual(set(feats.keys()), {"contrast", "entropy"})

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            glcm_features(self.window, features=("not_a_real_feature",))

    def test_per_angle_shape(self):
        glcm = compute_glcm(self.window, levels=16)
        feats = glcm_features(self.window, levels=16, per_angle=True)
        n_distances, n_angles = glcm.shape[2], glcm.shape[3]
        for value in feats.values():
            self.assertEqual(value.shape, (n_distances, n_angles))

    def test_fixed_vmin_vmax_changes_quantization_not_crash(self):
        feats = glcm_features(self.window, levels=32, vmin=0, vmax=255)
        self.assertEqual(set(feats.keys()), set(CORE_FEATURES))


if __name__ == "__main__":
    unittest.main()
