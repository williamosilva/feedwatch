"""Evaluate three fixed models on previously unused AV-FFIA days, without tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feedwatch.data import CLASS_IDS, file_hash, parse_relative_path, write_json
from feedwatch.evaluation import metrics, save_evaluation
from feedwatch.motion import FEATURE_NAMES, extract_video
from feedwatch.temporal_heads import load_temporal_checkpoint
from feedwatch.video_inference import OperationalPredictor
from feedwatch.visual import decode_visual

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/independent-evaluation/results-v1'


def main():
    if (OUTPUT/'results.json').exists():
        print('Independent evaluation already completed and retained.');return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/independent-evaluation'
    selection=json.loads((source/'selection.json').read_text())
    candidate_path=ROOT/'artifacts/experiments/portfolio-candidate-v1/seed_42/best_head.pt'
    training=ROOT/'artifacts/experiments/portfolio-candidate-v1/results.json'
    if not training.exists():raise ValueError('Candidate training must finish before the new test is opened')
    rf_path=ROOT/'artifacts/experiments/motion-full/random_forest/model.pkl'
    old_path=ROOT/'artifacts/experiments/temporal-controlled/gru/seed_42/best_head.pt'
    protocol={'selection_sha256':file_hash(source/'selection.json'),'candidate_sha256':file_hash(candidate_path),
        'reference_sha256':file_hash(old_path),'random_forest_sha256':file_hash(rf_path),
        'operational_candidate_seed':42,'old_test_2022_6_22_evaluated':False,
        'candidate_frozen_before_test':True,'thresholds':json.loads((ROOT/'config/portfolio_training.json').read_text())['promotion']}
    seal=OUTPUT/'protocol.json'
    if seal.exists() and json.loads(seal.read_text())!=protocol:raise ValueError('Frozen evaluation inputs changed')
    write_json(seal,protocol)
    predictor=OperationalPredictor(device_name='auto')
    candidate,bundle=load_temporal_checkpoint(candidate_path,predictor.device)
    candidate.requires_grad_(False).eval()
    with rf_path.open('rb') as stream:rf=pickle.load(stream)
    motion_cfg=json.loads((ROOT/'config/motion.json').read_text())['extraction']
    rows=[];probabilities={key:[] for key in ('operational','candidate','random_forest')};failures=[]
    deadline=time.monotonic()+3600
    records=OUTPUT/'observations.jsonl'
    if records.exists():raise ValueError('Partial evaluation exists; preserve it and inspect before retrying')
    for index,member in enumerate(selection['members'],1):
        relative=member['relative_path'];path=source/'videos'/relative
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Acquisition did not finish within one hour')
            time.sleep(1)
        row=parse_relative_path(relative)
        visual=decode_visual(source/'videos',relative,predictor.extraction,predictor.transform)
        motion=extract_video(source/'videos',relative,motion_cfg)
        if not visual['ok'] or not motion['ok']:
            failures.append({'relative_path':relative,'visual':visual.get('error'),'motion':motion.get('error')})
            continue
        from feedwatch.visual import encode_frames
        embeddings=encode_frames(predictor.extractor,visual['frames'],predictor.device)
        original=predictor.predict_embeddings(embeddings)
        with torch.inference_mode():
            tensor=torch.from_numpy((embeddings-bundle['mean'].numpy())/bundle['std'].numpy()).unsqueeze(0).to(predictor.device)
            proposed=candidate(tensor).softmax(1).cpu().numpy()[0]
        forest=rf['estimator'].predict_proba(pd.DataFrame([[motion['features'][n] for n in FEATURE_NAMES]],columns=FEATURE_NAMES))[0]
        values={'operational':np.asarray([original['scores'][n] for n in CLASS_IDS]),'candidate':proposed,'random_forest':forest}
        rows.append(row)
        for name,prob in values.items():probabilities[name].append(prob)
        with records.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'relative_path':relative,'source_sha256':file_hash(path),
                'scores':{name:prob.tolist() for name,prob in values.items()}})+'\n')
        if index%20==0 or index==len(selection['members']):print(f'Independent inputs evaluated {index}/{len(selection["members"])}; no tuning',flush=True)
    predictor.verify_frozen()
    frame=pd.DataFrame(rows)
    results={name:save_evaluation(frame,np.stack(prob),OUTPUT/name,'independent_2022_6_27_to_29') for name,prob in probabilities.items()}
    by_day={}
    for day,indices in frame.groupby('date').groups.items():
        by_day[day]={name:metrics(frame.loc[indices].class_id.to_numpy(),np.stack(prob)[list(indices)].argmax(1)) for name,prob in probabilities.items()}
    old,new,forest=results['operational'],results['candidate'],results['random_forest']
    rules=protocol['thresholds']
    deltas={'macro_f1':new['macro_f1']-old['macro_f1'],
            'medium_recall':new['classification_report']['medium']['recall']-old['classification_report']['medium']['recall'],
            'recall_by_class':{n:new['classification_report'][n]['recall']-old['classification_report'][n]['recall'] for n in CLASS_IDS}}
    checks={'macro_f1_gain':deltas['macro_f1']>=rules['macro_f1_gain'],
            'medium_recall_gain':deltas['medium_recall']>=rules['medium_recall_gain'],
            'other_class_recall_preserved':all(v>=-rules['max_other_recall_drop'] for n,v in deltas['recall_by_class'].items() if n!='medium'),
            'matches_random_forest':new['macro_f1']>=forest['macro_f1'],
            'complete_coverage':len(rows)==len(selection['members'])}
    result={'protocol':protocol,'requested':len(selection['members']),'evaluated':len(rows),'failures':failures,
            'models':results,'by_day':by_day,'deltas':deltas,'promotion_checks':checks,
            'candidate_eligible_for_promotion':all(checks.values()),'old_test_reevaluated':False,
            'limitations':['Small stratified sample; adjacent clips within days are correlated.',
                'Labels are published dataset labels, not newly independently adjudicated.',
                'New days from the same research setting do not establish generalization to other farms.']}
    write_json(OUTPUT/'results.json',result)
    lines=['# Avaliação independente do candidato','',f'{len(rows)} / {len(selection["members"])} clipes, dias 27–29/06/2022. Teste antigo não reavaliado.',
           '', '| Modelo | Acurácia | Macro-F1 | Recall média |','|---|---:|---:|---:|']
    for name,m in results.items():lines.append(f'| {name} | {m["accuracy"]:.4f} | {m["macro_f1"]:.4f} | {m["classification_report"]["medium"]["recall"]:.4f} |')
    lines+=['', 'Promoção: '+('elegível pelos critérios fixados' if all(checks.values()) else 'reprovada; manter o modelo operacional anterior'),
            '', 'Ver results.json para critérios, métricas por dia/classe, cobertura, hashes e limitações.',
            '', 'O teste usa rótulos publicados, não revisão independente por especialista. A amostra é do mesmo contexto de pesquisa.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'models':{n:{'macro_f1':m['macro_f1'],'medium_recall':m['classification_report']['medium']['recall']} for n,m in results.items()},'checks':checks}),flush=True)


if __name__=='__main__':main()
