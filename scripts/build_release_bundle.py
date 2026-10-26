"""Build source and a portable operational bundle, without shipping the raw dataset."""
import json
import shutil
import zipfile
from pathlib import Path

from feedwatch.data import file_hash
from feedwatch.portfolio_presentation import DIST
from prepare_public_repository import public_files, write_checkout

ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/DIST


def main():
    DIST.mkdir(parents=True,exist_ok=True)
    old=ROOT/'artifacts/experiments/temporal-controlled/gru/seed_42/best_head.pt'
    extractor=ROOT/'artifacts/pretrained/mobilenet_v3_small-047dcff4.pth'
    extraction=json.loads((ROOT/'config/visual.json').read_text())['extraction']
    runtime={'version':'feedwatch-operational-42-portable-v1','head':{'path':'artifacts/runtime/gru_seed_42.pt','sha256':file_hash(old)},
        'extractor':{'path':'artifacts/pretrained/'+extractor.name,'sha256':file_hash(extractor)},
        'visual_source_sha256':file_hash(ROOT/'src/feedwatch/visual.py'),'extraction':extraction,'expected_extraction':extraction,
        'example':'artifacts/runtime/aquario-exemplo.mp4',
        'example_credit':'Richard Muller / Pexels 6472322, Pexels License; external exploratory video, not feeding ground truth.'}
    hist_root=ROOT/'artifacts/experiments/final-test/models/gru_seed_42'
    splits=json.loads((ROOT/'artifacts/splits/validation.json').read_text())
    with zipfile.ZipFile(DIST/'feedwatch-runtime.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('artifacts/runtime.json',json.dumps(runtime,indent=2))
        archive.write(old,runtime['head']['path'])
        archive.write(extractor,runtime['extractor']['path'])
        archive.write(ROOT/'artifacts/external-examples/pexels-20261020/originals/pexels-6472322-fullhd.mp4',runtime['example'])
        archive.writestr('artifacts/splits/validation.json',json.dumps({'valid':splits['valid'],'composition':splits['composition']}))
        archive.write(hist_root/'metrics.json','artifacts/experiments/final-test/models/gru_seed_42/metrics.json')
        archive.writestr('artifacts/experiments/final-test/models/gru_seed_42/identification.json',json.dumps({'id':'gru_seed_42','family':'gru','seed':42}))
        archive.write(ROOT/'THIRD_PARTY.md','THIRD_PARTY-RUNTIME.md')
    instructions='''# Reproduzir o FeedWatch

Código AGPL-3.0. Baixe `feedwatch-source.zip` e `feedwatch-runtime.zip` e extraia ambos na mesma pasta. O pacote de execução usa caminhos relativos e preserva a GRU seed 42; não contém o dataset completo.

No Windows, com Python 3.12, abra PowerShell nessa pasta:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -e ".[local]"
.\\.venv\\Scripts\\python.exe -m feedwatch prepare-fish-detector
.\\.venv\\Scripts\\python.exe -m feedwatch serve
```

Para usar GPU, consulte a instalação do Torch/TorchVision no README. O detector é baixado da distribuição Fishial original; o comando verifica o hash configurado. O processamento também funciona em CPU, mas pode levar mais tempo. O pacote inclui um vídeo de exemplo para analisar na aplicação local.

A demonstração web reproduz resultados já calculados. O teste histórico e a avaliação adicional usam conjuntos diferentes de vídeos. Os detalhes estão no relatório de avaliação e nos créditos. Os hashes dos downloads estão em `downloads.json`.
'''
    (DIST/'reproducao.md').write_text(instructions,encoding='utf-8')
    files=public_files()
    with zipfile.ZipFile(DIST/'feedwatch-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(files.items()):archive.writestr(name,content)
    directory=write_checkout(files)
    hashes={p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in (DIST/'feedwatch-source.zip',DIST/'feedwatch-runtime.zip')}
    (DIST/'downloads.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps({'downloads':hashes,'public_repository':str(directory),'source_files':len(files)}),flush=True)


if __name__=='__main__':main()
