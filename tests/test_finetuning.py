import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights

from feedwatch.data import CLASS_IDS
from feedwatch.finetuning import (FineTunedClassifier, augment_clips, load_candidate,
                                 normalize_pixels, pixel_transform, scaled_optimizer_step, validation_gate)
from feedwatch.visual import preprocess_bgr, state_hash

ROOT = Path(__file__).resolve().parents[1]
HEAD = {'embedding_dim': 576, 'gru_hidden_size': 32, 'mlp_hidden_size': 101, 'dropout': .3}


class FineTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = json.loads((ROOT / 'config/finetune_v4.json').read_text())

    def test_cached_pixels_preserve_existing_preprocessing(self):
        bgr = np.random.default_rng(42).integers(0, 256, (300, 520, 3), dtype=np.uint8)
        config = {'crop_xyxy': [.05, .1, .95, .95]}
        reference = preprocess_bgr(bgr, config, MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
        pixels = preprocess_bgr(bgr, config, pixel_transform())
        actual = normalize_pixels(pixels[None, None].expand(1, 16, 3, 224, 224))
        torch.testing.assert_close(actual[0, 0], reference, atol=0, rtol=0)

    def test_augmentation_is_clip_consistent_reproducible_and_variable(self):
        pixels = torch.randint(0, 256, (2, 1, 3, 32, 32), dtype=torch.uint8).expand(2, 16, 3, 32, 32)
        first = augment_clips(pixels, self.cfg['augmentation'], torch.Generator().manual_seed(42))
        again = augment_clips(pixels, self.cfg['augmentation'], torch.Generator().manual_seed(42))
        other = augment_clips(pixels, self.cfg['augmentation'], torch.Generator().manual_seed(43))
        torch.testing.assert_close(first, again, atol=0, rtol=0)
        for frame in range(1, 16):
            torch.testing.assert_close(first[:, frame], first[:, 0], atol=0, rtol=0)
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(0 <= first.min() <= first.max() <= 1)

    def test_frames_keep_order_without_synthetic_flicker(self):
        pixels = torch.arange(16, dtype=torch.uint8).view(1, 16, 1, 1, 1).expand(1, 16, 3, 32, 32) * 10
        actual = augment_clips(pixels, self.cfg['augmentation'], torch.Generator().manual_seed(42))
        self.assertTrue(torch.all(actual.mean((2, 3, 4)).diff() > 0))

    def test_only_last_blocks_and_head_learn_batchnorm_statistics_stay_fixed(self):
        torch.manual_seed(42)
        model = FineTunedClassifier(HEAD, torch.zeros(576), torch.ones(576)).train()
        prefix = state_hash(model.features[:9])
        suffix = state_hash(model.features[9:])
        running = {k: v.clone() for k, v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        loss = torch.nn.functional.cross_entropy(model(torch.randint(0, 256, (1, 16, 3, 224, 224), dtype=torch.uint8)), torch.tensor([2]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.features[:9].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.features[9:].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))
        optimizer.step()
        self.assertEqual(prefix, state_hash(model.features[:9]))
        self.assertNotEqual(suffix, state_hash(model.features[9:]))
        for key, value in running.items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_checkpoint_restores_exact_predictions(self):
        model = FineTunedClassifier(HEAD, torch.zeros(576), torch.ones(576)).eval()
        pixels = torch.randint(0, 256, (1, 16, 3, 224, 224), dtype=torch.uint8)
        with torch.inference_mode():
            expected = model(pixels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'format': 'feedwatch-partial-finetune-v1', 'class_mapping': CLASS_IDS,
                        'head_config': HEAD, 'first_block': 9, 'state_dict': model.state_dict()}, path)
            actual, _ = load_candidate(path, torch.device('cpu'))
            with torch.inference_mode():
                torch.testing.assert_close(actual(pixels), expected, rtol=0, atol=0)

    def test_amp_overflow_skips_update_reduces_scale_and_recovers(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        original = state_hash(model)
        loss = model(torch.ones(1, 2)).sum()
        scaler.scale(loss).backward()
        model.weight.grad.fill_(float('inf'))
        self.assertFalse(scaled_optimizer_step(model, optimizer, scaler, 1.))
        self.assertEqual(original, state_hash(model))
        self.assertEqual(scaler.get_scale(), 64.)
        scaler.scale(model(torch.ones(1, 2)).sum()).backward()
        self.assertTrue(scaled_optimizer_step(model, optimizer, scaler, 1.))
        self.assertNotEqual(original, state_hash(model))

    def test_gate_rejects_improvement_at_expense_of_another_class(self):
        baseline = {'macro_f1': .5, 'classification_report': {n: {'recall': .5} for n in CLASS_IDS}}
        candidate = {'macro_f1': .6, 'classification_report': {n: {'recall': .65} for n in CLASS_IDS}}
        self.assertTrue(validation_gate([baseline]*3, [candidate]*3, self.cfg)['passed'])
        candidate['classification_report']['strong']['recall'] = .4
        self.assertFalse(validation_gate([baseline]*3, [candidate]*3, self.cfg)['passed'])

    def test_gate_requires_every_seed_and_fixed_operational_seed(self):
        baseline = {'macro_f1': .5, 'classification_report': {n: {'recall': .5} for n in CLASS_IDS}}
        candidate = {'macro_f1': .6, 'classification_report': {n: {'recall': .65} for n in CLASS_IDS}}
        weak = copy.deepcopy(candidate)
        weak['macro_f1'] = .51
        self.assertFalse(validation_gate([baseline]*3, [weak, candidate, candidate], self.cfg)['passed'])
        with self.assertRaises(ValueError):
            validation_gate([baseline]*3, [candidate]*2, self.cfg)


if __name__ == '__main__':
    unittest.main()
