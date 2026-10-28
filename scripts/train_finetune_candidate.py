"""Train/evaluate a bounded partial-fine-tuning experiment using local AV-FFIA clips."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from feedwatch.data import CLASS_IDS, file_hash, video_path, write_json, utc_now
from feedwatch.evaluation import metrics, save_evaluation
from feedwatch.experiment import load_train_val
from feedwatch.finetuning import (augment_clips, initialize_candidate, load_candidate,
                                 pixel_transform, scaled_optimizer_step, stream_hash, validation_gate)
from feedwatch.temporal_heads import EarlyStopping
from feedwatch.visual import decode_visual, state_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/experiments/finetune-v4'


class PixelDataset(Dataset):
    def __init__(self, path, rows):
        self.path, self.rows, self.array = str(path), rows, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r', allow_pickle=False)
        return torch.from_numpy(self.array[index].copy()), int(self.rows.iloc[index].class_id)


def deterministic_setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def prepare(cfg, rows, extraction, workers=4):
    folder = OUT / 'pixels'
    folder.mkdir(parents=True, exist_ok=True)
    transform = pixel_transform()
    data_root = (ROOT / 'video_6.19-6.22').resolve()
    for split, frame in rows.items():
        target = folder / f'{split}.npy'
        identity = {'rows_sha256': file_hash(OUT / f'used_{split}.csv'), 'extraction': extraction,
                    'geometry': 'official PIL resize + center crop, uint8 RGB; no normalization or augmentation',
                    'shape': [len(frame), 16, 3, 224, 224]}
        info = folder / f'{split}.json'
        if info.exists():
            saved = json.loads(info.read_text())
            if saved['identity'] != identity or stream_hash(target) != saved['cache_sha256']:
                raise ValueError('Pixel cache mismatch')
            ledger = folder / f'{split}-sources.json'
            if file_hash(ledger) != saved['sources_sha256']:
                raise ValueError('Source ledger changed')
            sources = json.loads(ledger.read_text())
            if [entry['relative_path'] for entry in sources] != frame.relative_path.tolist():
                raise ValueError('Cached source order changed')
            def check_source(entry):
                if stream_hash(video_path(data_root, entry['relative_path'])) != entry['sha256']:
                    raise ValueError('Source video changed since decoding: ' + entry['relative_path'])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(check_source, sources))
            print(f'Cache verified: {split}', flush=True)
            continue
        temporary = target.with_suffix('.partial.npy')
        cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=tuple(identity['shape']))
        def decode(path):
            source = video_path(data_root, path)
            before = (source.stat().st_size, source.stat().st_mtime_ns)
            decoded = decode_visual(data_root, path, extraction, transform)
            if not decoded['ok']:
                raise ValueError(f"{path}: {decoded['error']}")
            digest = stream_hash(source)
            if before != (source.stat().st_size, source.stat().st_mtime_ns):
                raise ValueError('Video changed during decoding')
            return decoded['frames'].numpy(), {'relative_path': path, 'sha256': digest, **decoded['details']}
        records = []
        started = time.perf_counter()
        # Bound pending decoded clips: executor.map would buffer an entire dataset on Windows.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = frame.relative_path.tolist()
            for offset in range(0, len(paths), 16):
                for i, (pixels, record) in enumerate(pool.map(decode, paths[offset:offset + 16]), start=offset):
                    cache[i] = pixels
                    records.append(record)
                if offset % 256 == 0:
                    print(f'Decoded {split} {len(records)}/{len(frame)} ({time.perf_counter()-started:.0f}s)', flush=True)
        cache.flush()
        del cache
        temporary.replace(target)
        write_json(folder / f'{split}-sources.json', records)
        write_json(info, {'identity': identity, 'cache_sha256': stream_hash(target),
                         'sources_sha256': file_hash(folder / f'{split}-sources.json'),
                         'seconds': time.perf_counter()-started, 'complete': True})
    train_hashes = {x['sha256'] for x in json.loads((folder / 'train-sources.json').read_text())}
    val_hashes = {x['sha256'] for x in json.loads((folder / 'val-sources.json').read_text())}
    if train_hashes & val_hashes:
        raise ValueError('Identical videos cross train/validation boundary')


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    for pixels, target in loader:
        logits.append(model(pixels.to(device, non_blocking=True)).cpu())
        labels.append(target)
    logits, labels = torch.cat(logits), torch.cat(labels)
    return logits.softmax(1).numpy(), metrics(labels.numpy(), logits.argmax(1).numpy()), nn.functional.cross_entropy(logits, labels).item()


def pilot(cfg, rows, extraction, device):
    model, _ = initialize_candidate(ROOT / 'artifacts/pretrained/mobilenet_v3_small-047dcff4.pth',
                                   ROOT / 'artifacts/experiments/temporal-controlled/gru/seed_42/best_head.pt', 9, device)
    data_root = (ROOT / 'video_6.19-6.22').resolve()
    clips = []
    for path in rows['train'].relative_path.iloc[:cfg['training']['batch_size']]:
        decoded = decode_visual(data_root, path, extraction, pixel_transform())
        if not decoded['ok']:
            raise ValueError(decoded['error'])
        clips.append(decoded['frames'])
    pixels = torch.stack(clips).to(device)
    generator = torch.Generator(device=device).manual_seed(20261028)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    seconds = []
    for _ in range(6):
        started = time.perf_counter()
        x = augment_clips(pixels, cfg['augmentation'], generator)
        model.zero_grad(set_to_none=True)
        logits = model(x, encoder_amp=True)
        logits.square().mean().backward()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
    report = {'batch_size': len(pixels), 'measured_step_seconds': seconds,
              'steady_step_seconds': float(np.mean(seconds[2:])),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'trainable_encoder_parameters': sum(p.numel() for p in model.features.parameters() if p.requires_grad),
              'trainable_head_parameters': sum(p.numel() for p in model.head.parameters()),
              'device': torch.cuda.get_device_name(), 'validation_examined': False, 'weights_saved': False}
    write_json(OUT / 'resource-pilot.json', report)
    print(json.dumps(report), flush=True)


def train_seed(seed, cfg, rows, device):
    folder = OUT / f'seed_{seed}'
    folder.mkdir(exist_ok=True)
    if (folder / 'results.json').exists():
        return json.loads((folder / 'results.json').read_text())
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    sampler_generator = torch.Generator().manual_seed(seed)
    tc = cfg['training']
    datasets = {s: PixelDataset(OUT / f'pixels/{s}.npy', r) for s, r in rows.items()}
    train_loader = DataLoader(datasets['train'], batch_size=tc['batch_size'], shuffle=True,
                              generator=sampler_generator, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(datasets['val'], batch_size=tc['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    original_path = ROOT / f'artifacts/experiments/temporal-controlled/gru/seed_{seed}/best_head.pt'
    model, original = initialize_candidate(ROOT / 'artifacts/pretrained/mobilenet_v3_small-047dcff4.pth',
                                           original_path, cfg['first_trainable_block'], device)
    frozen_initial = state_hash(model.features[:cfg['first_trainable_block']])
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.features.parameters() if p.requires_grad], 'lr': tc['encoder_learning_rate']},
        {'params': model.head.parameters(), 'lr': tc['head_learning_rate']}], weight_decay=tc['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=tc['encoder_amp'])
    stopping = EarlyStopping(tc['patience'])
    history = []
    started = time.perf_counter()
    provenance = {'protocol_sha256': file_hash(OUT / 'protocol.json'), 'initial_head_sha256': file_hash(original_path),
                  'train_rows_sha256': file_hash(OUT / 'used_train.csv'), 'val_rows_sha256': file_hash(OUT / 'used_val.csv')}
    def save_best(epoch, validation):
        torch.save({'format': 'feedwatch-partial-finetune-v1', 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'head_config': original['head_config'], 'first_block': cfg['first_trainable_block'],
                    'class_mapping': CLASS_IDS, 'seed': seed, 'selected_epoch': epoch, 'validation': validation,
                    'provenance': provenance, 'extractor_trained': epoch > 0}, folder / 'best_model.pt')
    initial_prob, initial_metrics, initial_loss = evaluate(model, val_loader, device)
    reference = pd.read_csv(ROOT / f'artifacts/experiments/temporal-controlled/gru/seed_{seed}/val/predictions.csv')
    reference = reference.set_index('relative_path').loc[rows['val'].relative_path]
    np.testing.assert_array_equal(initial_prob.argmax(1), reference.predicted_id)
    np.testing.assert_allclose(initial_prob, reference[[f'prob_{n}' for n in CLASS_IDS]], atol=2e-4, rtol=1e-4)
    write_json(folder / 'initial-parity.json', {'predictions_equal': True, 'validation': initial_metrics,
               'max_probability_difference': float(np.abs(initial_prob-reference[[f'prob_{n}' for n in CLASS_IDS]].to_numpy()).max())})
    stopping.step(initial_metrics['macro_f1'])
    best_epoch = 0
    save_best(0, initial_metrics)
    history.append({'epoch': 0, 'train_loss_augmented': None, 'val_loss': initial_loss, 'val_macro_f1': initial_metrics['macro_f1'],
                    'val_accuracy': initial_metrics['accuracy'], 'val_medium_recall': initial_metrics['classification_report']['medium']['recall'], 'seconds': 0})
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, tc['max_epochs'] + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, total_count, skipped_steps, completed_steps = 0., 0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (pixels, targets) in enumerate(train_loader):
            pixels, targets = pixels.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            augmented = augment_clips(pixels, cfg['augmentation'], generator)
            loss = nn.functional.cross_entropy(model(augmented, encoder_amp=tc['encoder_amp']), targets,
                                                label_smoothing=tc['label_smoothing'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite forward loss')
            # Correct final accumulation group: it may contain fewer than 32 clips.
            start = (step // tc['accumulation_steps']) * tc['accumulation_steps'] * tc['batch_size']
            group_examples = min(tc['accumulation_steps'] * tc['batch_size'], len(datasets['train']) - start)
            scaler.scale(loss * len(targets) / group_examples).backward()
            total_loss += loss.item() * len(targets)
            total_count += len(targets)
            if (step + 1) % tc['accumulation_steps'] == 0 or step + 1 == len(train_loader):
                updated = scaled_optimizer_step(model, optimizer, scaler, tc['max_grad_norm'])
                completed_steps += int(updated)
                skipped_steps += int(not updated)
        if not completed_steps:
            raise FloatingPointError('No optimizer updates completed this epoch')
        _, validation, val_loss = evaluate(model, val_loader, device)
        improved, stop = stopping.step(validation['macro_f1'])
        if improved:
            best_epoch = epoch
            save_best(epoch, validation)
        history.append({'epoch': epoch, 'train_loss_augmented': total_loss / total_count, 'val_loss': val_loss,
                        'val_macro_f1': validation['macro_f1'], 'val_accuracy': validation['accuracy'],
                        'val_medium_recall': validation['classification_report']['medium']['recall'],
                        'amp_skipped_steps': skipped_steps, 'optimizer_steps': completed_steps,
                        'seconds': time.perf_counter() - epoch_started})
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(f"seed={seed} epoch={epoch} val_f1={validation['macro_f1']:.4f} medium_recall={history[-1]['val_medium_recall']:.4f} best_epoch={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
        if stop:
            break
    if state_hash(model.features[:cfg['first_trainable_block']]) != frozen_initial:
        raise ValueError('Frozen prefix changed')
    del model
    model, bundle = load_candidate(folder / 'best_model.pt', device)
    probabilities, replay, _ = evaluate(model, val_loader, device)
    if abs(replay['macro_f1'] - stopping.best) > 1e-12:
        raise ValueError('Reloaded model does not reproduce selection metric')
    validation = save_evaluation(rows['val'], probabilities, folder / 'val', 'validation_development')
    result = {'seed': seed, 'selected_epoch': best_epoch, 'epochs_run': epoch,
              'baseline': initial_metrics, 'validation': validation, 'checkpoint_sha256': file_hash(folder / 'best_model.pt'),
              'frozen_prefix_unchanged': True, 'checkpoint_replay_verified': True,
              'training_seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'test_evaluated': False}
    write_json(folder / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'prepare', 'train', 'all'], default='all')
    parser.add_argument('--decode-workers', type=int, default=4)
    args = parser.parse_args()
    if args.decode_workers < 1 or args.decode_workers > (os.cpu_count() or 1):
        raise ValueError('Invalid number of decoding workers')
    cfg = json.loads((ROOT / 'config/finetune_v4.json').read_text())
    deterministic_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment configuration requires a local CUDA GPU')
    device = torch.device('cuda')
    rows, provenance = load_train_val(ROOT / 'artifacts/manifest.csv', ROOT / 'artifacts/splits')
    rows = {s: r.sort_values('relative_path').reset_index(drop=True) for s, r in rows.items()}
    for split in rows:
        expected_days = cfg['train_days' if split == 'train' else 'validation_days']
        if len(rows[split]) != cfg['expected_counts'][split] or set(rows[split].date) != set(expected_days):
            raise ValueError('Training/validation partition differs from protocol')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != cfg:
        raise ValueError('Do not alter an existing protocol')
    write_json(protocol, cfg)
    for split, frame in rows.items():
        frame.to_csv(OUT / f'used_{split}.csv', index=False)
    extraction = json.loads((ROOT / 'config/visual.json').read_text())['extraction']
    if args.stage == 'pilot':
        pilot(cfg, rows, extraction, device)
        return
    prepare(cfg, rows, extraction, args.decode_workers)
    if args.stage == 'prepare':
        return
    run_record = OUT / 'run.json'
    if run_record.exists():
        previous = json.loads(run_record.read_text())
        current_code = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [Path(__file__), ROOT / 'src/feedwatch/finetuning.py']}
        if previous['code_sha256'] != current_code:
            if any(OUT.glob('seed_*/results.json')):
                raise ValueError('Completed seeds cannot be mixed across code revisions')
            write_json(OUT / ('run-before-' + str(time.time_ns()) + '.json'), previous)
    if not any(OUT.glob('seed_*/results.json')):
        write_json(run_record, {'started_at': utc_now(), 'data': provenance, 'device': torch.cuda.get_device_name(),
                   'torch': str(torch.__version__), 'protocol_sha256': file_hash(protocol),
                   'code_sha256': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [Path(__file__), ROOT / 'src/feedwatch/finetuning.py']}})
    results = [train_seed(seed, cfg, rows, device) for seed in cfg['seeds']]
    gate = validation_gate([r['baseline'] for r in results], [r['validation'] for r in results], cfg)
    write_json(OUT / 'results.json', {'experiment': cfg['experiment'], 'runs': results, 'validation_gate': gate,
               'audit_required': gate['passed'], 'audit_evaluated': False, 'promoted': False,
               'decision': 'Await prespecified retrospective audits' if gate['passed'] else 'Keep original operational model',
               'new_data_downloaded': False, 'finished_at': utc_now()})
    print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    main()
