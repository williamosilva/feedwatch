"""Render learning curves and audit saved, labeled evaluations without rerunning test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feedwatch.data import file_hash, write_json
from feedwatch.evaluation import metrics
from learning_svg_locale import write_english_svg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/learning-review-v3'
DIST=ROOT/'sites/feedwatch/dist/learning'
LABELS=['Não observada','Baixa','Moderada','Alta']


def bootstrap(frame,draws=2000):
    """Resample days, then complete groups within each sampled day; preserve correlation."""
    rng=np.random.default_rng(20261023)
    clusters={day:[group.index.to_numpy() for _,group in rows.groupby('group',sort=True)]
              for day,rows in frame.groupby('date',sort=True)}
    days=sorted(clusters)
    true,pred=frame.class_id.to_numpy(),frame.predicted_id.to_numpy()
    values=[]
    for _ in range(draws):
        indices=[]
        for day_index in rng.integers(0,len(days),len(days)):
            groups=clusters[days[day_index]]
            indices.extend(groups[i] for i in rng.integers(0,len(groups),len(groups)))
        idx=np.concatenate(indices)
        values.append(float(np.mean(true[idx]==pred[idx])))
    return {'accuracy_interval_95':np.quantile(values,[.025,.975]).tolist(),'replicates':draws,
        'seed':20261023,'days':len(days),'date_group_clusters':sum(map(len,clusters.values())),
        'resampling':'days, then whole date/group clusters; conditional diagnostic with only three observed days; not a guarantee for other farms'}


def save_figure(fig,name):
    for suffix in ('svg','png'):
        fig.savefig(OUT/f'{name}.{suffix}',dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    write_english_svg(OUT/f'{name}.svg')


def main():
    OUT.mkdir(parents=True,exist_ok=True);DIST.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'artifacts/experiments/temporal-controlled/gru/seed_42'
    history=pd.read_csv(folder/'history.csv')
    original=json.loads((folder/'results.json').read_text())
    epoch=original['selected_epoch']
    assert int(history.loc[history.val_macro_f1.idxmax(),'epoch'])==epoch
    for split,column in (('train','train_accuracy'),('validation','val_accuracy')):
        assert abs(float(history.loc[history.epoch.eq(epoch),column].iloc[0])-original[split]['accuracy'])<1e-10
    new=ROOT/'artifacts/independent-evaluation/results-v1/operational'
    predictions=pd.read_csv(new/'predictions.csv').reset_index(drop=True)
    recorded=json.loads((new/'metrics.json').read_text())
    measured=metrics(predictions.class_id.to_numpy(),predictions.predicted_id.to_numpy())
    assert measured['confusion_matrix']==recorded['confusion_matrix']
    assert abs(measured['accuracy']-recorded['accuracy'])<1e-12
    regularization=json.loads((ROOT/'artifacts/experiments/generalization-v2/results.json').read_text())
    audit=bootstrap(predictions)
    confidence=predictions[['prob_none','prob_weak','prob_medium','prob_strong']].max(axis=1).to_numpy()
    correct=predictions.class_id.eq(predictions.predicted_id).to_numpy()
    ece=0.
    for i in range(10):
        mask=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else (confidence<=1))
        if mask.any():ece+=mask.mean()*abs(float(confidence[mask].mean())-float(correct[mask].mean()))
    payload={'operational_model':'MobileNetV3 Small frozen + GRU seed 42','trained_component':'GRU classification head',
        'training_dataset':'AV-FFIA subset; 3898 train clips from 2022_6_19 and 2022_6_20; 1705 validation clips from 2022_6_21',
        'detector':'Fishial pretrained separately; no detector accuracy measured in this review',
        'selected_epoch':epoch,'epochs_run':len(history),'train_accuracy':original['train']['accuracy'],
        'validation_accuracy':original['validation']['accuracy'],'validation_macro_f1':original['validation']['macro_f1'],
        'history':history.to_dict('records'),'independent_metrics':measured,'independent_uncertainty':audit,
        'calibration_diagnostic':{'mean_max_score':float(confidence.mean()),'accuracy':float(correct.mean()),'ece_10_bins':float(ece),
            'interpretation':'Uncalibrated score is not probability of being correct; no temperature or threshold selected on this set.'},
        'regularization_comparison':{k:regularization[k] for k in ('configurations','selected_configuration','baseline_mean_macro_f1',
            'deltas','validation_gate_checks','eligible_for_new_holdout','test_evaluated','operational_model_replaced')},
        'sources':{str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in
            (folder/'history.csv',folder/'results.json',new/'predictions.csv',new/'metrics.json',
             ROOT/'config/generalization_v2.json',ROOT/'artifacts/experiments/generalization-v2/results.json')},
        'historical_test_reopened':False,'previous_independent_predictions_recomputed':False,
        'new_heldout_days_evaluated':False,'limits':'Only three independent evaluation days in the same research setting. No human video-wide ground truth or detector boxes were added.'}
    payload['regularization_comparison']['baseline_mean_accuracy']=float(np.mean([
        json.loads((ROOT/f'artifacts/experiments/temporal-controlled/gru/seed_{s}/results.json').read_text())['validation']['accuracy']
        for s in (42,43,44)]))
    write_json(OUT/'evidence.json',payload)
    history.to_csv(OUT/'operational-history.csv',index=False)
    pd.DataFrame(measured['confusion_matrix'],index=LABELS,columns=LABELS).to_csv(OUT/'confusion-matrix.csv',index_label='Referência')
    plt.rcParams.update({'font.family':'Arial','font.size':11,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263746','text.color':'#263746'})
    fig,ax=plt.subplots(figsize=(10,4.2))
    ax.plot(history.epoch,100*history.train_accuracy,label='Vídeos de treino',color='#173b59',linewidth=2)
    ax.plot(history.epoch,100*history.val_accuracy,label='Vídeos de validação (outro dia)',color='#497b65',linewidth=2,linestyle='--')
    ax.axvline(epoch,color='#8695a2',linestyle=':',linewidth=1.2,label=f'Versão escolhida: etapa {epoch}')
    for key,color in (('train_accuracy','#173b59'),('val_accuracy','#497b65')):
        value=100*float(history.loc[history.epoch.eq(epoch),key].iloc[0])
        ax.scatter([epoch],[value],color=color,s=36,zorder=5)
        ax.annotate(f'{value:.1f}%'.replace('.',','),(epoch,value),xytext=(8,-17 if key=='train_accuracy' else 9),textcoords='offset points',color=color,fontweight='bold')
    ax.set(xlabel='Etapa de treinamento (época)',ylabel='Classificações corretas (%)',ylim=(0,104),xlim=(1,len(history)))
    ax.grid(axis='y',color='#e5eaf0');ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False,fontsize=10)
    fig.tight_layout();save_figure(fig,'operational-learning')
    fig,ax=plt.subplots(figsize=(9,3.4))
    recalls=[100*measured['classification_report'][name]['recall'] for name in ('none','weak','medium','strong')]
    bars=ax.barh(LABELS,recalls,color=['#708ba1','#52758f','#173b59','#365b78'],height=.58)
    ax.invert_yaxis();ax.set(xlim=(0,108),xlabel='Vídeos do nível corretamente identificados (%)')
    ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color='#e5eaf0');ax.set_axisbelow(True)
    for bar,value in zip(bars,recalls):ax.text(value+2,bar.get_y()+bar.get_height()/2,f'{value:.1f}%'.replace('.',','),va='center')
    fig.tight_layout();save_figure(fig,'class-recall')
    fig,ax=plt.subplots(figsize=(4.4,3.9))
    ax.plot(history.epoch,100*history.train_accuracy,label='Treino',color='#173b59',linewidth=2)
    ax.plot(history.epoch,100*history.val_accuracy,label='Validação',color='#497b65',linewidth=2,linestyle='--')
    ax.axvline(epoch,color='#8695a2',linestyle=':',linewidth=1.2,label=f'Etapa {epoch}')
    for key,color in (('train_accuracy','#173b59'),('val_accuracy','#497b65')):
        value=100*float(history.loc[history.epoch.eq(epoch),key].iloc[0])
        ax.scatter([epoch],[value],color=color,s=24,zorder=5)
        ax.annotate(f'{value:.1f}%'.replace('.',','),(epoch,value),xytext=(-45,-17 if key=='train_accuracy' else 9),textcoords='offset points',color=color,fontweight='bold',fontsize=10)
    ax.set(xlabel='Etapa de treinamento',ylabel='Acerto (%)',ylim=(0,104),xlim=(1,len(history)),xticks=[1,15,30,47])
    ax.grid(axis='y',color='#e5eaf0');ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False,fontsize=10)
    fig.tight_layout();save_figure(fig,'operational-learning-mobile')
    fig,ax=plt.subplots(figsize=(4.4,3.2))
    bars=ax.barh(LABELS,recalls,color=['#708ba1','#52758f','#173b59','#365b78'],height=.58)
    ax.invert_yaxis();ax.set(xlim=(0,125),xlabel='Identificações corretas (%)',xticks=[0,50,100])
    ax.grid(axis='x',color='#e5eaf0');ax.set_axisbelow(True)
    for bar,value in zip(bars,recalls):ax.text(value+3,bar.get_y()+bar.get_height()/2,f'{value:.1f}%'.replace('.',','),va='center',fontsize=10)
    fig.tight_layout();save_figure(fig,'class-recall-mobile')
    for p in OUT.iterdir():
        if p.suffix in ('.svg','.csv','.json'):(DIST/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'selected_epoch':epoch,'train_accuracy':payload['train_accuracy'],
        'validation_accuracy':payload['validation_accuracy'],'bootstrap':audit,
        'regularization_eligible':regularization['eligible_for_new_holdout']}),flush=True)


if __name__=='__main__':main()
