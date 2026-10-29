"""Create a source distribution without full videos, weights or machine-local state."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from export_public_evidence import DOCUMENTS, EVIDENCE

ROOT=Path(__file__).resolve().parents[1]
STATIC_EXTENSIONS={'.html','.css','.js','.svg','.json','.csv','.md','.txt','.jpg'}
STATIC_DOCUMENTS={'methodology.md','evaluation.md','finetuning.md','THIRD_PARTY.md','reproducao.md'}
ROOT_FILES=('README.md','README.pt-BR.md','LICENSE','THIRD_PARTY.md','pyproject.toml','.gitignore','.gitattributes',
            'requirements-lock.txt','requirements-local-lock.txt','requirements-dynamic-lock.txt')
README_ASSETS=('feedwatch-demo.gif','feedwatch-demo-pt.gif',
               'feedwatch-interface-en.png','feedwatch-interface-pt.png')
PUBLIC_SCRIPTS=(
    'acquire_independent_sample.py','benchmark_fish_detection.py','benchmark_fish_tracking.py',
    'benchmark_tracking_continuity.py',
    'build_diverse_evidence.py','build_learning_evidence.py','build_finetune_evidence.py','build_portfolio.py',
    'build_release_bundle.py','check_public_assets.py','evaluate_independent_candidate.py',
    'evaluate_sessions.py','export_public_evidence.py','prepare_official_continuous.py',
    'learning_svg_locale.py',
    'prepare_public_repository.py','retrack_saved_video.py','serve_portfolio.py','train_diverse_candidate.py',
    'train_generalization_candidate.py','train_portfolio_candidate.py','train_finetune_candidate.py',
)
PUBLIC_IGNORE='''/.venv/
.*/
!.github/
node_modules/
coverage/
test-results/
.DS_Store
Thumbs.db
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
/build/
/dist/
.env
.env.*
/video_*/
/data/
/datasets/
/artifacts/
/weights/
/checkpoints/
/runs/
*.mp4
*.avi
*.mov
*.mkv
*.webm
*.m4v
*.mpeg
*.mpg
*.zip
*.7z
*.tar*
*.pt
*.pth
*.ckpt
*.onnx
*.npy
*.npz
*.pkl
'''


def public_files():
    files={}
    for directory in ('src/feedwatch','config','tests'):
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in {'.py','.css','.js','.json','.toml'}:continue
            files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for name in PUBLIC_SCRIPTS:
        files['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
    for name in ROOT_FILES:files[name]=(ROOT/name).read_bytes()
    files['.gitignore']=PUBLIC_IGNORE.encode('utf-8')
    for name in DOCUMENTS:files['docs/'+name]=(ROOT/'docs'/name).read_bytes()
    for name in README_ASSETS:files['docs/assets/'+name]=(ROOT/'docs/assets'/name).read_bytes()
    dist=ROOT/'sites/feedwatch/dist'
    for name in EVIDENCE.values():files['evidence/'+name]=(dist/'evidence'/name).read_bytes()
    for path in dist.rglob('*'):
        if not path.is_file() or path.suffix not in STATIC_EXTENSIONS or path.name=='downloads.json':continue
        if 'media' in path.relative_to(dist).parts:continue
        if path.suffix=='.md' and path.name not in STATIC_DOCUMENTS:continue
        if path.parent==dist/'evidence' and path.name not in EVIDENCE.values():continue
        relative=path.relative_to(ROOT).as_posix()
        if path.name=='data.json':
            data=json.loads(path.read_text(encoding='utf-8'))
            files['evidence/examples.json']=json.dumps(data,ensure_ascii=False).encode('utf-8')
            data['cases']=[]
            data['message']='Os vídeos da demonstração não acompanham o código. Na aplicação local, selecione seu vídeo ou use o exemplo do pacote de execução.'
            files[relative]=json.dumps(data,ensure_ascii=False).encode('utf-8')
        elif path.name=='index.html':
            text=path.read_text(encoding='utf-8')
            text=re.sub(r'<a\b[^>]*href="feedwatch-source\.zip"[^>]*>.*?</a>',
                        '<a href="../../../README.md">README</a>',text,flags=re.DOTALL)
            text=text.replace('href="feedwatch-runtime.zip" download',
                              'href="https://github.com/williamosilva/feedwatch/releases/download/v0.1.0/feedwatch-runtime.zip"')
            files[relative]=text.encode('utf-8')
        elif path.name=='THIRD_PARTY.md':
            files[relative]=(ROOT/'THIRD_PARTY.md').read_bytes()
        elif path.name=='reproducao.md':
            files[relative]=('''# Executar o FeedWatch

Siga as instruções do [README em inglês](../../../README.md) ou do [README em português](../../../README.pt-BR.md).

O [pacote de execução](https://github.com/williamosilva/feedwatch/releases/download/v0.1.0/feedwatch-runtime.zip) contém o classificador treinado, o extrator e um vídeo de exemplo. Extraia o pacote na raiz do projeto. Os detectores Fishial são baixados pelo comando indicado no README.

Não é necessário baixar os datasets nem treinar novamente para analisar um vídeo. O código está sob AGPL-3.0; os componentes externos mantêm seus próprios termos, descritos em [THIRD_PARTY.md](THIRD_PARTY.md).
''').encode('utf-8')
        else:files[relative]=path.read_bytes()
    return files


def write_checkout(files):
    base=(ROOT/'artifacts/public-repository').resolve()
    base.mkdir(parents=True,exist_ok=True)
    directory=base/('FeedWatch-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    directory.mkdir()
    for name,content in files.items():
        target=directory/name
        if not target.resolve().is_relative_to(directory):raise ValueError('Invalid source path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    (base/'latest.json').write_text(json.dumps({'directory':str(directory),'files':len(files)},indent=2),encoding='utf-8')
    return directory


def main():
    files=public_files()
    directory=write_checkout(files)
    print(json.dumps({'directory':str(directory),'files':len(files)}))


if __name__=='__main__':main()
