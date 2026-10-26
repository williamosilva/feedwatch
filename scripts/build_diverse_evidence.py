"""Publish recorded training curves and paired evaluations without rerunning a test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from feedwatch.data import file_hash
from learning_svg_locale import write_english_svg

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'artifacts/experiments/diverse-v3'
DIST=ROOT/'sites/feedwatch/dist/learning'


def main():
    validation_path=EXPERIMENT/'validation-results.json'
    if not validation_path.exists():
        print(json.dumps({'status':'training_not_completed'}))
        return
    validation=json.loads(validation_path.read_text(encoding='utf-8'))
    protocol=json.loads((EXPERIMENT/'protocol.json').read_text(encoding='utf-8'))
    run=next(r for r in validation['runs'] if r['seed']==42)
    history=pd.read_csv(EXPERIMENT/'seed_42/history.csv')
    assert file_hash(EXPERIMENT/'seed_42/best_head.pt')==run['checkpoint_sha256']
    epoch=run['selected_epoch']
    assert epoch==int(history.loc[history.val_mean_day_macro_f1.idxmax(),'epoch'])
    point=history.loc[history.epoch.eq(epoch)].iloc[0]
    assert abs(float(point.train_accuracy)-run['training']['accuracy'])<1e-10
    assert abs(float(point.val_accuracy)-run['validation']['accuracy'])<1e-10
    heldout_path=EXPERIMENT/'heldout-results.json'
    heldout=json.loads(heldout_path.read_text(encoding='utf-8')) if heldout_path.exists() else None
    manifest_path=ROOT/'config/operational_model.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    deployed=bool(manifest and manifest.get('head',{}).get('sha256')==run['checkpoint_sha256'])
    if deployed:
        assert heldout and heldout['eligible_for_promotion']
        assert manifest['promotion']['heldout_sha256']==file_hash(heldout_path)
    counts={name:len(pd.read_csv(EXPERIMENT/f'expanded_{name}.csv')) for name in ('train','val')}
    payload={'schema':1,'operational_model_replaced':deployed,'protocol':protocol,
             'counts':counts,'seed_42':run,'comparison':validation['comparison'],
             'heldout':heldout,'history':history.to_dict('records'),
             'source_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in
                 (validation_path,EXPERIMENT/'protocol.json',EXPERIMENT/'seed_42/history.csv',
                  EXPERIMENT/'seed_42/best_head.pt')},
             'scope':'Recording days from the same research site; no completed human session validation or measured feed savings.'}
    DIST.mkdir(parents=True,exist_ok=True)
    (DIST/'diverse-evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    history.to_csv(DIST/'diverse-history.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':11,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'text.color':'#263746'})
    for mobile,size in ((False,(10,4.2)),(True,(4.4,3.9))):
        fig,ax=plt.subplots(figsize=size)
        ax.plot(history.epoch,100*history.train_accuracy,label='Treino',color='#173b59',linewidth=2)
        ax.plot(history.epoch,100*history.val_accuracy,label='Validação',color='#497b65',linewidth=2,linestyle='--')
        ax.axvline(epoch,color='#8695a2',linestyle=':',label=f'Etapa escolhida: {epoch}')
        ax.set(xlabel='Etapa de treinamento',ylabel='Classificações corretas (%)',ylim=(0,104))
        ax.grid(axis='y',color='#e5eaf0');ax.set_axisbelow(True);ax.legend(frameon=False,loc='lower right')
        fig.tight_layout()
        figure_path=DIST/('diverse-learning-mobile.svg' if mobile else 'diverse-learning.svg')
        fig.savefig(figure_path,facecolor='white')
        plt.close(fig)
        write_english_svg(figure_path)
    print(json.dumps({'selected_epoch':epoch,'counts':counts,'operational_model_replaced':deployed,
                      'heldout_evaluated':heldout is not None}))


if __name__=='__main__':main()
