"""Export scientific records and user documentation."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/'sites/feedwatch/dist'
EVIDENCE={
    'config/generalization_v2.json':'generalization-protocol.json',
    'config/portfolio_training.json':'training-protocol.json',
    'artifacts/experiments/generalization-v2/results.json':'generalization-results.json',
    'artifacts/independent-evaluation/selection.json':'independent-selection.json',
    'artifacts/independent-evaluation/acquisition.json':'independent-acquisition.json',
    'artifacts/independent-evaluation/results-v1/results.json':'independent-results.json',
    'artifacts/experiments/final-test/models/gru_seed_42/metrics.json':'historical-metrics.json',
    'config/diverse_v3.json':'diverse-training-protocol.json',
    'artifacts/experiments/diverse-v3/validation-results.json':'diverse-validation-results.json',
    'artifacts/improvements-v9/detection-evidence.json':'detection-regression.json',
    'artifacts/tracking-continuity/summary.json':'tracking-continuity.json',
    'artifacts/tracking-continuity/references-v1.json':'tracking-references.json',
    'config/finetune_v4.json':'finetune-protocol.json',
    'artifacts/experiments/finetune-v4/results.json':'finetune-results.json',
    'artifacts/experiments/finetune-v4/verification.json':'finetune-verification.json',
}
DOCUMENTS=('methodology.md','evaluation.md','finetuning.md','session-review-template.csv')


def portable_text(text):
    return text.replace(str(ROOT)+'\\','').replace(ROOT.as_posix()+'/','').replace(str(Path.home())+'\\','<user>/')


def portable_value(value):
    if isinstance(value,dict):return {key:portable_value(item) for key,item in value.items()}
    if isinstance(value,list):return [portable_value(item) for item in value]
    return portable_text(value) if isinstance(value,str) else value


def main():
    evidence=DIST/'evidence';evidence.mkdir(parents=True,exist_ok=True)
    repository_evidence=ROOT/'evidence';repository_evidence.mkdir(exist_ok=True)
    for source,name in EVIDENCE.items():
        value=json.loads((ROOT/source).read_text(encoding='utf-8-sig'))
        text=json.dumps(portable_value(value),ensure_ascii=False,indent=2)
        (evidence/name).write_text(text,encoding='utf-8')
        (repository_evidence/name).write_text(text,encoding='utf-8')
    for name in DOCUMENTS:
        text=(ROOT/'docs'/name).read_text(encoding='utf-8')
        text=text.replace('../evidence/','evidence/').replace('../sites/feedwatch/dist/','')
        (DIST/name).write_text(text,encoding='utf-8')
    for name in ('LICENSE','THIRD_PARTY.md'):shutil.copy2(ROOT/name,DIST/name)
    print(json.dumps({'scientific_records':len(EVIDENCE),'documents':len(DOCUMENTS)}))


if __name__=='__main__':main()
