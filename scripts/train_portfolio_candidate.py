"""One preregistered weighted GRU experiment; train/validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feedwatch.data import CLASS_IDS, file_hash, write_json
from feedwatch.evaluation import save_evaluation
from feedwatch.temporal_data import load_sequences
from feedwatch.temporal_heads import TemporalHead, EarlyStopping, evaluate_head, load_temporal_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/experiments/portfolio-candidate-v1'


def main():
    if (OUTPUT / 'results.json').exists():
        print('Completed experiment retained:', OUTPUT)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'config/portfolio_training.json'
    cfg = json.loads(config_path.read_text())
    protocol = OUTPUT / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != cfg: raise ValueError('Protocol changed')
    write_json(protocol, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    array, rows, audit, _ = load_sequences(ROOT/'video_6.19-6.22',ROOT/'artifacts/manifest.csv',
        ROOT/'artifacts/splits',ROOT/'artifacts/experiments/visual-full',ROOT/'artifacts/visual-cache',
        OUTPUT/'repair-cache',OUTPUT,None,42,device)
    indices = {k: np.flatnonzero(rows.split.eq(k)) for k in ('train','val')}
    means = array[indices['train']].mean(1)
    mean, std = means.mean(0), np.maximum(means.std(0),1e-6)
    x = {k: torch.from_numpy((array[idx]-mean)/std).to(device) for k,idx in indices.items()}
    y = {k: rows.iloc[idx].class_id.to_numpy() for k,idx in indices.items()}
    metadata = {k: rows.iloc[idx].reset_index(drop=True) for k,idx in indices.items()}
    weights = len(y['train'])/(4*np.bincount(y['train'],minlength=4))
    weights[CLASS_IDS['medium']] *= cfg['medium_multiplier']
    weights /= weights.mean()
    tensor_weights = torch.tensor(weights,dtype=torch.float32,device=device)
    targets = torch.from_numpy(y['train']).long().to(device)
    heads = json.loads((ROOT/'config/temporal.json').read_text())['heads']
    provenance = {'extraction_config_id':audit['config_id'],'train_sha256':file_hash(OUTPUT/'used_train.csv'),
        'protocol_sha256':file_hash(protocol),'test_evaluated':False,'new_test_used_for_selection':False}
    runs=[]
    for seed in cfg['seeds']:
        folder = OUTPUT / f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        model = TemporalHead('gru',heads).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        stopping=EarlyStopping(cfg['patience'])
        history=[];started=time.perf_counter()
        for epoch in range(1,cfg['max_epochs']+1):
            model.train()
            for batch in torch.randperm(len(targets),generator=generator,device=device).split(cfg['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x['train'][batch]),targets[batch],weight=tensor_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            _,train_metrics,train_loss=evaluate_head(model,x['train'],y['train'],256)
            _,val_metrics,val_loss=evaluate_head(model,x['val'],y['val'],256)
            improved,stop=stopping.step(val_metrics['macro_f1'])
            history.append({'epoch':epoch,'train_macro_f1':train_metrics['macro_f1'],'val_macro_f1':val_metrics['macro_f1'],
                            'val_medium_recall':val_metrics['classification_report']['medium']['recall'],'train_loss':train_loss,'val_loss':val_loss})
            if improved:
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'kind':'gru','seed':seed,'head_config':heads,'training_config':cfg,
                    'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'class_mapping':CLASS_IDS,
                    'selected_epoch':epoch,'validation_macro_f1':val_metrics['macro_f1'],
                    'provenance':provenance,'extractor_trained':False,'test_evaluated':False,
                    'loss_weights':torch.tensor(weights,dtype=torch.float32)},folder/'best_head.pt')
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            if epoch==1 or epoch%5==0 or stop:
                print(f'seed={seed} epoch={epoch} val_f1={val_metrics["macro_f1"]:.4f} medium_recall={val_metrics["classification_report"]["medium"]["recall"]:.4f}',flush=True)
            if stop:break
        model,bundle=load_temporal_checkpoint(folder/'best_head.pt',device)
        prob,measured,_=evaluate_head(model,x['val'],y['val'],256)
        assert abs(measured['macro_f1']-bundle['validation_macro_f1'])<1e-12
        result=save_evaluation(metadata['val'],prob,folder/'val','validation_development')
        reloaded,_=load_temporal_checkpoint(folder/'best_head.pt',device)
        again,_,_=evaluate_head(reloaded,x['val'],y['val'],256)
        np.testing.assert_allclose(prob,again,atol=1e-7)
        runs.append({'seed':seed,'selected_epoch':bundle['selected_epoch'],'seconds':time.perf_counter()-started,
            'validation':result,'checkpoint_sha256':file_hash(folder/'best_head.pt'),'checkpoint_replay_verified':True})
        write_json(folder/'results.json',runs[-1])
    write_json(OUTPUT/'results.json',{'runs':runs,'loss_weights':weights.tolist(),'protocol':cfg,
        'test_evaluated':False,'new_test_evaluated':False,'operational_candidate_seed':42,
        'mean_validation_macro_f1':float(np.mean([r['validation']['macro_f1'] for r in runs]))})
    print('Training complete; candidate frozen before independent evaluation.',flush=True)


if __name__=='__main__':main()
