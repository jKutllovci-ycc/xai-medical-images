import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cf = load_script("06_counterfactual.py")
export = load_script("10_export_clear_counterfactuals.py")


class BrightnessClassifier(torch.nn.Module):
    def forward(self, x):
        brightness = x.mean(dim=(1, 2, 3))
        return torch.stack((-brightness, brightness), dim=1)


class CounterfactualTests(unittest.TestCase):
    def test_gan_models_preserve_image_shape_and_bounds(self):
        generator = cf.GANGenerator(base_channels=8, residual_blocks=1)
        discriminator = cf.GANDiscriminator(base_channels=8)
        source = torch.rand(2, 1, 64, 64) * 2 - 1
        generated = generator(source)
        score = discriminator(generated)
        self.assertEqual(generated.shape, source.shape)
        self.assertEqual(score.shape[0], source.shape[0])
        self.assertLessEqual(float(generated.detach().max()), 1.0)
        self.assertGreaterEqual(float(generated.detach().min()), -1.0)

    def test_gan_classifier_conversion_is_three_channel(self):
        source = torch.zeros(2, 1, 64, 64)
        converted = cf.gan_to_classifier_input(source)
        self.assertEqual(converted.shape, (2, 3, 224, 224))

    def test_pneumonia_opacity_is_visible_bounded_and_strength_dependent(self):
        original = np.full((224, 224, 3), 0.4, dtype=np.float32)
        mask = np.ones((224, 224), dtype=np.float32)
        mild = cf.semantic_edit(original, mask, 1, 6, 42)
        strong = cf.semantic_edit(original, mask, 1, 12, 42)
        self.assertGreater(float(np.percentile(strong - original, 90)), 0.10)
        self.assertGreater(float(strong.mean()), float(mild.mean()))
        self.assertTrue(np.all(strong >= original))
        self.assertLessEqual(float(strong.max()), 1.0)
        protected = cf.semantic_edit(original, np.zeros_like(mask), 1, 12, 42)
        np.testing.assert_array_equal(protected, original)

    def test_gradcam_refreshes_cached_predictions_after_retraining(self):
        gradcam = load_script("05_gradcam.py")
        rows = [dict(path=f"example-{c}", true_idx=c, pred_idx=c,
                     correct=True, confidence=0.9, normal_prob=0.9 if c == 0 else 0.1,
                     pneumonia_prob=0.1 if c == 0 else 0.9, true_class_prob=0.9)
                for c in (0, 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.csv"
            gradcam.write_selected_examples(rows, path)
            current = [{**row, "confidence": 0.95} for row in rows]
            selected = gradcam.resolve_selected_examples(current, path, 1)
            self.assertTrue(all(row["confidence"] == 0.95 for row in selected))
            self.assertTrue(all(row["confidence"] == 0.95
                                for row in gradcam.load_selected_examples(path)))

    def test_selection_excludes_misclassified_sources(self):
        rows = [dict(path=f"{c}-{i}", true_idx=c, correct=correct, confidence=0.9)
                for c in (0, 1) for i, correct in enumerate((False, True))]
        selected = cf.select_examples(rows, 2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["correct"] for row in selected))
        self.assertEqual({row["true_idx"] for row in selected}, {0, 1})

    def test_export_preserves_native_detail_with_zero_edit(self):
        native = np.random.default_rng(0).random((127, 193, 3), dtype=np.float32)
        original = np.full((224, 224, 3), 0.4, dtype=np.float32)
        np.testing.assert_array_equal(export.transfer_edit(native, original, original), native)

    def test_export_applies_grayscale_delta_and_clips(self):
        native = np.full((125, 201, 3), 0.95, dtype=np.float32)
        original = np.full((224, 224, 3), 0.4, dtype=np.float32)
        result = export.transfer_edit(native, original, original + 0.1)
        self.assertEqual(result.shape, native.shape)
        np.testing.assert_allclose(result, 1)

    def test_optimizer_returns_scored_snapshot_and_grayscale_change(self):
        for module in (cf, load_script("07_gradcam_guided_counterfactual.py")):
            with self.subTest(module=module.__name__):
                model = BrightnessClassifier()
                rgb = np.full((16, 16, 3), 0.4, dtype=np.float32)
                input_tensor = cf.display_to_tensor(rgb, torch.device("cpu"))
                delta, _, prob, _ = module.generate_counterfactual(
                    model, input_tensor, 0, 1, torch.ones_like(input_tensor),
                    torch.device("cpu"), 4, 0.1, 0.04, 0.3, 0.7, 1.0, 4,
                )
                actual = torch.softmax(model(module.clamp_normalized(input_tensor + delta)), dim=1)[0, 1]
                self.assertAlmostEqual(prob, actual.item(), places=6)
                display = cf.tensor_to_display(input_tensor + delta)
                np.testing.assert_allclose(display[:, :, 0], display[:, :, 1], atol=1e-6)
                np.testing.assert_allclose(display[:, :, 0], display[:, :, 2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
