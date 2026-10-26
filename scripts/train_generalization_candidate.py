"""Bounded, preregistered regularization comparison using train/validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feedwatch.data import file_hash, write_json
from feedwatch.evaluation import save_evaluation
from feedwatch.temporal_data import load_sequences
from feedwatch.temporal_heads import TemporalHead, EarlyStopping, evaluate_head, load_temporal_checkpoint

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/generalization-v2'


def main():
    cfg_path=ROOT/'config/generalization_v2.json'
    cfg=json.loads(cfg_path.read_text())
    if (OUT/'results.json').exists():
        print('Completed comparison retained.');return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=cfg:raise ValueError('Protocol changed')
    write_json(protocol,cfg)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    array,rows,audit,_=load_sequences(ROOT/'video_6.19-6.22',ROOT/'artifacts/manifest.csv',ROOT/'artifacts/splits',
        ROOT/'artifacts/experiments/visual-full',ROOT/'artifacts/visual-cache',OUT/'repair-cache',OUT,None,42,device)
    indices={k:np.flatnonzero(rows.split.eq(k)) for k in ('train','val')}
    means=array[indices['train']].mean(1)
    mean,std=means.mean(0),np.maximum(means.std(0),1e-6)
    x={k:torch.from_numpy((array[idx]-mean)/std).to(device) for k,idx in indices.items()}
    y={k:rows.iloc[idx].class_id.to_numpy() for k,idx in indices.items()}
    metadata={k:rows.iloc[idx].reset_index(drop=True) for k,idx in indices.items()}
    target=torch.from_numpy(y['train']).long().to(device)
    baseline=[json.loads((ROOT/f'artifacts/experiments/temporal-controlled/gru/seed_{s}/results.json').read_text())['validation'] for s in cfg['seeds']]
    runs=[]
    for settings in cfg['configurations']:
        for seed in cfg['seeds']:
            folder=OUT/settings['name']/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'results.json').exists():
                runs.append(json.loads((folder/'results.json').read_text()));continue
            torch.manual_seed(seed)
            generator=torch.Generator(device=device).manual_seed(seed)
            heads={**json.loads((ROOT/'config/temporal.json').read_text())['heads'],
                   'gru_hidden_size':settings['hidden_size'],'dropout':settings['dropout']}
            model=TemporalHead('gru',heads).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=settings['weight_decay'])
            stopping=EarlyStopping(cfg['patience']);history=[];started=time.perf_counter()
            for epoch in range(1,cfg['max_epochs']+1):
                model.train()
                for batch in torch.randperm(len(target),generator=generator,device=device).split(cfg['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(model(x['train'][batch]),target[batch],label_smoothing=settings['label_smoothing'])
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                _,tm,tl=evaluate_head(model,x['train'],y['train'],256)
                _,vm,vl=evaluate_head(model,x['val'],y['val'],256)
                improved,stop=stopping.step(vm['macro_f1'])
                history.append({'epoch':epoch,'train_accuracy':tm['accuracy'],'val_accuracy':vm['accuracy'],
                    'train_macro_f1':tm['macro_f1'],'val_macro_f1':vm['macro_f1'],'train_loss':tl,'val_loss':vl})
                if improved:
                    torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'kind':'gru',
                        'seed':seed,'head_config':heads,'training_config':cfg,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                        'class_mapping':{'none':0,'weak':1,'medium':2,'strong':3},'selected_epoch':epoch,
                        'validation_macro_f1':vm['macro_f1'],'extractor_trained':False,'test_evaluated':False,
                        'provenance':{'protocol_sha256':file_hash(protocol),'train_sha256':file_hash(OUT/'used_train.csv'),
                            'validation_sha256':file_hash(OUT/'used_val.csv'),'extraction_config_id':audit['config_id']}},folder/'best_head.pt')
                if stop:break
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            model,bundle=load_temporal_checkpoint(folder/'best_head.pt',device)
            prob,_,_=evaluate_head(model,x['val'],y['val'],256)
            validation=save_evaluation(metadata['val'],prob,folder/'val','validation_development')
            _,training,_=evaluate_head(model,x['train'],y['train'],256)
            run={'configuration':settings['name'],'seed':seed,'selected_epoch':bundle['selected_epoch'],
                 'epochs_run':len(history),'seconds':time.perf_counter()-started,'train':training,
                 'validation':validation,'checkpoint_sha256':file_hash(folder/'best_head.pt')}
            write_json(folder/'results.json',run);runs.append(run)
            print(f"{settings['name']} seed={seed} epoch={bundle['selected_epoch']} train={training['accuracy']:.4f} val={validation['accuracy']:.4f} macro_f1={validation['macro_f1']:.4f}",flush=True)
    def average(items,metric):return float(np.mean([r[metric] for r in items]))
    def recall(items,label):return float(np.mean([r['classification_report'][label]['recall'] for r in items]))
    configurations=[]
    for settings in cfg['configurations']:
        subset=[r['validation'] for r in runs if r['configuration']==settings['name']]
        configurations.append({'name':settings['name'],'mean_macro_f1':average(subset,'macro_f1'),
            'mean_accuracy':average(subset,'accuracy'),'recalls':{n:recall(subset,n) for n in ('none','weak','medium','strong')}})
    selected=max(configurations,key=lambda c:c['mean_macro_f1'])
    deltas={'macro_f1':selected['mean_macro_f1']-average(baseline,'macro_f1'),
        'recall':{n:selected['recalls'][n]-recall(baseline,n) for n in selected['recalls']}}
    gate=cfg['validation_gate']
    checks={'macro_f1_gain':deltas['macro_f1']>=gate['mean_macro_f1_gain'],
        'medium_recall_gain':deltas['recall']['medium']>=gate['mean_medium_recall_gain'],
        'other_recalls':all(v>=-gate['max_other_mean_recall_drop'] for n,v in deltas['recall'].items() if n!='medium')}
    write_json(OUT/'results.json',{'configurations':configurations,'selected_configuration':selected['name'],
        'candidate_seed':42,'baseline_mean_macro_f1':average(baseline,'macro_f1'),'deltas':deltas,
        'validation_gate_checks':checks,'eligible_for_new_holdout':all(checks.values()),'runs':runs,
        'test_evaluated':False,'operational_model_replaced':False,'protocol_sha256':file_hash(protocol)})
    print(json.dumps({'selected':selected['name'],'validation_gate':checks,'eligible_for_new_holdout':all(checks.values())}),flush=True)


if __name__=='__main__':main()
