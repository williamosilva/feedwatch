"""Load the unchanged operational model from a relocatable, hash-checked bundle."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import CLASS_IDS, file_hash
from .final_evaluation import inference_setup
from .temporal_heads import load_temporal_checkpoint
from .video_inference import OperationalPredictor
from .visual import load_extractor, state_hash


def load_predictor(protocol=Path('artifacts/experiments/final-test'), device='auto'):
    manifest=Path('artifacts/runtime.json')
    if not manifest.exists():return OperationalPredictor(protocol,device)
    record=json.loads(manifest.read_text(encoding='utf-8'))
    root=manifest.resolve().parent.parent
    def verified(key):
        spec=record[key];path=(root/spec['path']).resolve()
        if root not in path.parents or file_hash(path)!=spec['sha256']:
            raise ValueError('Arquivo do pacote de execução inválido: '+key)
        return path
    head_path,weights_path=verified('head'),verified('extractor')
    from . import visual
    if file_hash(Path(visual.__file__))!=record['visual_source_sha256']:
        raise ValueError('Pré-processamento difere do pacote operacional')
    predictor=OperationalPredictor.__new__(OperationalPredictor)
    predictor.device=inference_setup(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    predictor.extraction=record['extraction']
    predictor.extractor,predictor.transform,weights=load_extractor(weights_path.parent,predictor.device)
    predictor.head,predictor.bundle=load_temporal_checkpoint(head_path,predictor.device)
    if predictor.bundle['kind']!='gru' or predictor.bundle['seed']!=42:
        raise ValueError('O pacote deve preservar a GRU operacional seed 42')
    if predictor.extraction!=record['expected_extraction']:
        raise ValueError('Parâmetros de extração incompatíveis')
    predictor.mean,predictor.std=predictor.bundle['mean'].numpy(),predictor.bundle['std'].numpy()
    if predictor.mean.shape!=(576,) or predictor.std.shape!=(576,) or not np.isfinite(predictor.mean).all() or not np.isfinite(predictor.std).all() or (predictor.std<=0).any():
        raise ValueError('Normalização de treino inválida')
    predictor.head.requires_grad_(False).eval()
    predictor.initial_extractor=state_hash(predictor.extractor)
    predictor.initial_head=state_hash(predictor.head)
    predictor.identification={'checkpoint':record['head']['path'],'checkpoint_sha256':file_hash(head_path),
        'seed':42,'kind':'gru','runtime_device':str(predictor.device),'class_mapping':CLASS_IDS,
        'bundle_version':record['version'],'scores':'softmax outputs, not calibrated confidence','training':False,
        'extraction_config_id':predictor.bundle['provenance']['extraction_config_id']}
    return predictor
