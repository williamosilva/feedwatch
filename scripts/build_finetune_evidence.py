"""Verify saved fine-tuning results and render the recorded learning curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from feedwatch.data import file_hash, write_json
from feedwatch.evaluation import metrics
from feedwatch.finetuning import validation_gate

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'artifacts/experiments/finetune-v4'
DIST = ROOT / 'sites/feedwatch/dist/learning'


def main():
    result = json.loads((EXPERIMENT / 'results.json').read_text())
    if result['audit_required'] and not result['audit_evaluated']:
        raise ValueError('Complete the prespecified audits before publishing a decision')
    cfg = json.loads((EXPERIMENT / 'protocol.json').read_text())
    initial = torch.load(ROOT / 'artifacts/pretrained/mobilenet_v3_small-047dcff4.pth', weights_only=True, map_location='cpu')
    histories, checks, sources = {}, [], {}
    for run in result['runs']:
        seed = run['seed']
        folder = EXPERIMENT / f'seed_{seed}'
        history = pd.read_csv(folder / 'history.csv')
        checkpoint = folder / 'best_model.pt'
        if file_hash(checkpoint) != run['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after evaluation')
        bundle = torch.load(checkpoint, weights_only=True, map_location='cpu')
        old_head = ROOT / f'artifacts/experiments/temporal-controlled/gru/seed_{seed}/best_head.pt'
        if file_hash(old_head) != bundle['provenance']['initial_head_sha256']:
            raise ValueError('Original trained head changed during the experiment')
        if int(history.loc[history.val_macro_f1.idxmax(), 'epoch']) != run['selected_epoch']:
            raise ValueError('Selected epoch mismatch')
        if abs(history.loc[history.epoch.eq(run['selected_epoch']), 'val_macro_f1'].iloc[0] - run['validation']['macro_f1']) > 1e-12:
            raise ValueError('History does not reproduce reported score')
        predicted = pd.read_csv(folder / 'val/predictions.csv')
        measured = metrics(predicted.class_id.to_numpy(), predicted.predicted_id.to_numpy())
        if measured['confusion_matrix'] != run['validation']['confusion_matrix']:
            raise ValueError('Published predictions do not reproduce confusion matrix')
        changed, max_change = [], 0.
        for key, value in bundle['state_dict'].items():
            if not key.startswith('features.'):
                continue
            same = torch.equal(value, initial[key])
            block = int(key.split('.')[1])
            if block < cfg['first_trainable_block'] or 'running_' in key or 'num_batches_tracked' in key:
                if not same:
                    raise ValueError(f'Frozen parameter/buffer changed: {key}')
            elif not same:
                changed.append(key)
                max_change = max(max_change, float((value - initial[key]).abs().max()))
        if run['selected_epoch'] > 0 and not changed:
            raise ValueError('Fine-tuning did not change the saved encoder')
        checks.append({'seed': seed, 'original_head_unchanged': True,
                       'frozen_prefix_unchanged': True, 'batchnorm_statistics_unchanged': True,
                       'changed_encoder_tensors': len(changed), 'maximum_encoder_parameter_change': max_change,
                       'saved_predictions_reproduce_metrics': True})
        histories[str(seed)] = json.loads(history.to_json(orient='records'))
        for path in (folder / 'history.csv', folder / 'val/predictions.csv', checkpoint):
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    gate = validation_gate([r['baseline'] for r in result['runs']], [r['validation'] for r in result['runs']], cfg)
    if gate != result['validation_gate']:
        raise ValueError('Promotion gate mismatch')
    summary = {}
    for kind in ('baseline', 'validation'):
        values = [run[kind] for run in result['runs']]
        summary[kind] = {'mean_macro_f1': float(np.mean([v['macro_f1'] for v in values])),
                         'mean_accuracy': float(np.mean([v['accuracy'] for v in values])),
                         'mean_recalls': {name: float(np.mean([v['classification_report'][name]['recall'] for v in values]))
                                          for name in ('none', 'weak', 'medium', 'strong')}}
    payload = {'schema': 1, 'protocol': cfg, 'results': result, 'comparison': summary, 'histories': histories,
               'verification': checks, 'source_sha256': sources,
               'scope': 'Development validation on one recording day; does not measure generalization to other farms or ration savings.'}
    DIST.mkdir(parents=True, exist_ok=True)
    write_json(DIST / 'finetune-evidence.json', payload)
    write_json(EXPERIMENT / 'verification.json', {'checks': checks, 'gate_recomputed': True, 'comparison': summary})
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'text.color': '#263746'})
    colors = ['#173b59', '#497b65', '#9a6839']
    for language in ('en', 'pt'):
        for mobile, size in ((False, (10, 4.2)), (True, (4.4, 3.9))):
            fig, ax = plt.subplots(figsize=size)
            for run, color in zip(result['runs'], colors):
                frame = pd.DataFrame(histories[str(run['seed'])])
                label = f"Seed {run['seed']}" if language == 'en' else f"Inicialização {run['seed']}"
                ax.plot(frame.epoch, frame.val_macro_f1, label=label, color=color, linewidth=1.8)
                selected = frame.loc[frame.epoch.eq(run['selected_epoch'])].iloc[0]
                ax.scatter([selected.epoch], [selected.val_macro_f1], color=color, s=35, zorder=3)
            ax.set(xlabel='Fine-tuning epoch' if language == 'en' else 'Época do ajuste',
                   ylabel='Validation macro-F1' if language == 'en' else 'Macro-F1 na validação', ylim=(0, 1))
            ax.grid(axis='y', color='#e5eaf0')
            ax.set_axisbelow(True)
            ax.legend(frameon=False, loc='lower right', fontsize=9 if mobile else 11)
            fig.tight_layout()
            name = 'finetune-learning' + ('-mobile' if mobile else '') + ('.en' if language == 'en' else '')
            fig.savefig(DIST / f'{name}.svg', facecolor='white')
            if not mobile:
                fig.savefig(EXPERIMENT / f'{name}.png', dpi=170, facecolor='white')
            plt.close(fig)
    pd.concat([pd.DataFrame(history).assign(seed=int(seed)) for seed, history in histories.items()], ignore_index=True).to_csv(DIST / 'finetune-history.csv', index=False)
    print(json.dumps({'comparison': summary, 'gate': gate, 'verification_passed': True}), flush=True)


if __name__ == '__main__':
    main()
