"""Partial MobileNet fine-tuning; the original frozen extractor stays unchanged."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, PILToTensor

from .data import CLASS_IDS
from .temporal_heads import TemporalHead


def stream_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_transform():
    """Same PIL geometry as the existing extractor, before float normalization."""
    preset = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    return Compose([Resize(preset.resize_size, interpolation=preset.interpolation, antialias=True),
                    CenterCrop(preset.crop_size), PILToTensor()])


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 5 or pixels.shape[1:] != (16, 3, 224, 224):
        raise ValueError('Expected RGB clips [B,16,3,224,224]')
    x = pixels.float() / 255 if pixels.dtype == torch.uint8 else pixels
    mean = x.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
    std = x.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def augment_clips(pixels: torch.Tensor, settings: dict, generator: torch.Generator) -> torch.Tensor:
    """Sample once per clip, including contrast reference, to avoid artificial flicker."""
    if pixels.dtype != torch.uint8 or pixels.ndim != 5:
        raise ValueError('Augmentation requires uint8 RGB clips')
    x = pixels.float() / 255
    b, t, c, h, w = x.shape
    def uniform(key):
        low, high = settings[key]
        return torch.rand(b, generator=generator, device=x.device) * (high - low) + low
    side = uniform('crop_side_range')
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = side
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device, generator=generator) * 2 - 1) * (1 - side[:, None])
    theta = theta.repeat_interleave(t, 0)
    flat = x.flatten(0, 1)
    grid = nn.functional.affine_grid(theta, flat.shape, align_corners=False)
    x = nn.functional.grid_sample(flat, grid, mode='bilinear', padding_mode='border', align_corners=False).view(b, t, c, h, w)
    x = (x * uniform('brightness_range').view(b, 1, 1, 1, 1)).clamp(0, 1)
    gray_weights = x.new_tensor([.2989, .587, .114]).view(1, 1, 3, 1, 1)
    gray = (x * gray_weights).sum(2, keepdim=True)
    reference = gray.mean((1, 3, 4), keepdim=True)
    x = (reference + (x - reference) * uniform('contrast_range').view(b, 1, 1, 1, 1)).clamp(0, 1)
    gray = (x * gray_weights).sum(2, keepdim=True)
    return (gray + (x - gray) * uniform('saturation_range').view(b, 1, 1, 1, 1)).clamp(0, 1)


class FineTunedClassifier(nn.Module):
    def __init__(self, head_config: dict, mean: torch.Tensor, std: torch.Tensor, first_block: int = 9):
        super().__init__()
        if first_block not in range(1, 13) or mean.shape != (576,) or std.shape != (576,) or (std <= 0).any():
            raise ValueError('Invalid fine-tuning configuration')
        self.first_block = first_block
        self.head_config = head_config
        encoder = mobilenet_v3_small(weights=None)
        self.features = encoder.features
        self.pool = encoder.avgpool
        self.head = TemporalHead('gru', head_config)
        self.register_buffer('mean', mean.clone().float())
        self.register_buffer('std', std.clone().float())
        for index, block in enumerate(self.features):
            block.requires_grad_(index >= first_block)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for index, block in enumerate(self.features):
            if index < self.first_block:
                block.eval()
        # Small, correlated frame batches should not replace ImageNet running stats.
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, pixels: torch.Tensor, encoder_amp: bool = False):
        x = normalize_pixels(pixels)
        batch, frames = x.shape[:2]
        with torch.autocast(device_type=x.device.type, enabled=encoder_amp, dtype=torch.float16):
            x = x.flatten(0, 1)
            with torch.no_grad():
                for index in range(self.first_block):
                    x = self.features[index](x)
            for index in range(self.first_block, len(self.features)):
                x = self.features[index](x)
            sequence = self.pool(x).flatten(1).view(batch, frames, 576)
        return self.head((sequence.float() - self.mean) / self.std)


def initialize_candidate(weights: Path, head_path: Path, first_block: int, device: torch.device):
    # Local files only. Do not implicitly download weights or datasets.
    if not stream_hash(weights).startswith('047dcff4'):
        raise ValueError('Unexpected MobileNet weights')
    original = torch.load(head_path, weights_only=True, map_location='cpu')
    if original['class_mapping'] != CLASS_IDS or original['extractor_trained'] or original['kind'] != 'gru':
        raise ValueError('Expected original frozen-encoder GRU')
    model = FineTunedClassifier(original['head_config'], original['mean'], original['std'], first_block)
    state = torch.load(weights, weights_only=True, map_location='cpu')
    model.features.load_state_dict({k.removeprefix('features.'): v for k, v in state.items() if k.startswith('features.')})
    model.head.load_state_dict(original['state_dict'])
    return model.to(device), original


def load_candidate(path: Path, device: torch.device):
    bundle = torch.load(path, weights_only=True, map_location='cpu')
    if bundle['format'] != 'feedwatch-partial-finetune-v1' or bundle['class_mapping'] != CLASS_IDS:
        raise ValueError('Incompatible fine-tuned checkpoint')
    state = bundle['state_dict']
    model = FineTunedClassifier(bundle['head_config'], state['mean'], state['std'], bundle['first_block'])
    model.load_state_dict(state)
    return model.to(device).eval(), bundle


def scaled_optimizer_step(model, optimizer, scaler, max_grad_norm):
    """Let AMP back off its scale on overflow without touching model weights."""
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
    if finite:
        nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        scaler.step(optimizer)
    elif not scaler.is_enabled():
        raise FloatingPointError('Non-finite gradients without mixed precision')
    # unscale_ records overflows; update lowers the scale even when step is skipped.
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return finite


def validation_gate(baselines: list[dict], candidates: list[dict], cfg: dict) -> dict:
    if len(baselines) != len(cfg['seeds']) or len(candidates) != len(baselines):
        raise ValueError('Every prespecified seed is required')
    delta = float(np.mean([c['macro_f1'] - b['macro_f1'] for b, c in zip(baselines, candidates)]))
    recalls = {name: float(np.mean([c['classification_report'][name]['recall'] - b['classification_report'][name]['recall']
                                   for b, c in zip(baselines, candidates)])) for name in CLASS_IDS}
    g = cfg['validation_gate']
    i = cfg['seeds'].index(cfg['operational_seed'])
    checks = {'mean_macro_f1_gain': delta >= g['mean_macro_f1_gain'],
              'medium_recall_gain': recalls['medium'] >= g['mean_medium_recall_gain'],
              'other_classes_preserved': all(recalls[n] >= -g['max_other_mean_recall_drop'] for n in CLASS_IDS if n != 'medium'),
              'consistent_seeds': sum(c['macro_f1'] > b['macro_f1'] for b, c in zip(baselines, candidates)) >= g['minimum_improved_seeds'],
              'operational_seed_gain': candidates[i]['macro_f1'] - baselines[i]['macro_f1'] >= g['operational_seed_macro_f1_gain']}
    return {'passed': all(checks.values()), 'checks': checks, 'mean_macro_f1_delta': delta, 'mean_recall_deltas': recalls}
