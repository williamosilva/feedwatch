"""Package real saved predictions and licensed media for a buildless portfolio."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from feedwatch.data import file_hash
from feedwatch.portfolio_presentation import ASSETS, DIST, HEADER, WORKSPACE
from feedwatch.video_export import hidden_process, verify_export
from feedwatch.video_overview import summarize_video
from feedwatch.learning_presentation import learning_html
from feedwatch.i18n import i18n_javascript, localize_html

ROOT=Path(__file__).resolve().parents[1]
NAMES={'none':'Não observada','weak':'Baixa','medium':'Moderada','strong':'Alta'}


def encode(source,target,filters):
    if target.exists():return
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-i',str(source),
        '-vf',filters,'-an','-c:v','libx264','-crf','23','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
        check=True,timeout=180,**hidden_process())


def main():
    DIST.mkdir(parents=True,exist_ok=True)
    media=DIST/'media';media.mkdir(exist_ok=True)
    for name in ('portfolio.css','portfolio.js'):shutil.copy2(ASSETS/name,DIST/('styles.css' if name.endswith('css') else 'app.js'))
    cases=[]
    for identifier,title,runid,original,credit in [
        ('aquario','Aquário','20261026T010808-detector-v2-aquario','pexels-6472322-fullhd.mp4','Richard Muller · Pexels 6472322 · licença Pexels'),
        ('carpas','Carpas — exemplo com falhas','20261026T011136-detector-v3-carpas','pexels-3625177-fullhd.mp4','lam loi · Pexels 3625177 · licença Pexels')]:
        directory=ROOT/'artifacts/local-app-runs'/runid
        run=json.loads((directory/'run.json').read_text(encoding='utf-8'))
        raw=ROOT/'artifacts/external-examples/pexels-20261020/originals'/original
        origin=media/f'{identifier}-original.mp4'
        if not origin.exists():shutil.copy2(raw,origin)
        target=media/f'{identifier}-resultado-{file_hash(directory/"annotated.mp4")[:12]}.mp4'
        encode(directory/'annotated.mp4',target,'crop=1920:1080:0:0')
        frames=[{'count':r.get('detected_count')} for r in (json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines())]
        poster=media/f'{identifier}.jpg'
        cap=cv2.VideoCapture(str(target));cap.set(cv2.CAP_PROP_POS_MSEC,3000);ok,img=cap.read();cap.release()
        if ok:cv2.imwrite(str(poster),img)
        cases.append({'id':identifier,'title':title,'original':origin.relative_to(DIST).as_posix(),'annotated':target.relative_to(DIST).as_posix(),
            'poster':poster.relative_to(DIST).as_posix(),'windows':run['windows'],'fps':run['metadata']['fps'],
            'duration':run['metadata']['duration_seconds'],'frames':frames,
            'scope':'Este vídeo foi gravado em um ambiente diferente do AV-FFIA. Não há uma classificação de referência para medir o acerto da atividade alimentar neste exemplo.',
            'credit':credit,
            'processing_note':'Filmagem integral; rodapé da saída técnica removido para exibição em 16:9.',
            'source_sha256':file_hash(raw),'result_sha256':file_hash(target)})
    catalog=json.loads((ROOT/'artifacts/demos/validation-mvp/demo_report.json').read_text())['examples']
    for label in NAMES:
        entry=next(e for e in catalog if e['class_name']==label)
        directory=ROOT/'artifacts/demos/validation-mvp'/entry['folder']
        analysis=json.loads((directory/'analysis.json').read_text())
        original=ROOT/'video_6.19-6.22'/entry['relative_path']
        target=media/f'dataset-{label}.mp4'
        encode(original,target,'scale=1920:1080')
        poster=media/f'dataset-{label}.jpg'
        cap=cv2.VideoCapture(str(target));ok,img=cap.read();cap.release()
        if ok:cv2.imwrite(str(poster),img)
        cases.append({'id':'dataset-'+label,'title':'AV-FFIA — referência: '+NAMES[label].lower()+' (2 s)','reference_class':label,
            'original':target.relative_to(DIST).as_posix(),'annotated':target.relative_to(DIST).as_posix(),'poster':poster.relative_to(DIST).as_posix(),
            'windows':[{**w,'available_at_seconds':w['end_seconds']} for w in analysis['windows']],'fps':25.,'duration':2.,'frames':[],
            'scope':'A categoria de referência é a publicada no AV-FFIA. Este clipe fez parte da validação do projeto; a previsão do modelo pode discordar da referência.',
            'credit':'Meng Cui e colaboradores · AV-FFIA · CC BY 4.0 · Clipe da validação.',
            'source_clip':entry['relative_path'],'selection_note':'Seleção histórica, seed 42.',
            'source_sha256':file_hash(original),'result_sha256':file_hash(target)})
    for case in cases:case['overview']=summarize_video(case['windows'],case['duration'],case['frames'])
    data={'cases':cases,'model':'MobileNetV3 Small congelada + GRU seed 42','historical':{'accuracy':.6055962691538974,'macro_f1':.5902772087968453,'medium_recall':.18309859154929578}}
    (DIST/'data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    with (DIST/'app.js').open('a',encoding='utf-8') as out:
        out.write("\nfetch('data.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error();return r.json()}).then(data=>mountFeedWatch(document.querySelector('[data-fw-root]'),data)).catch(()=>{const target=document.querySelector('[data-fw-root]');const showError=()=>{target.textContent=window.FeedWatchI18n?.t('Não foi possível carregar a demonstração. Recarregue a página.')||'The demonstration could not be loaded. Reload the page.';};showError();window.FeedWatchI18n?.subscribe(showError);});\n")
    body=HEADER
    body+='<main><section id="demo" data-fw-root>'+WORKSPACE+'</section>'
    body+='''<section id="project-section" class="fw-section"><h2>Pesquisa de referência</h2><p>A ração é um dos principais custos da piscicultura. A <a href="https://www.fisheries.noaa.gov/insight/feeds-aquaculture">NOAA descreve o uso de câmeras para acompanhar o consumo</a> e evitar o fornecimento em excesso. Esse é o problema que motivou o FeedWatch.</p><p>Para treinar o modelo, usei parte do <a href="https://arxiv.org/html/2309.05058v3#S5">AV-FFIA, de Meng Cui e colaboradores</a>. O conjunto contém vídeos classificados por intensidade alimentar, com orientação de um técnico. O estudo U-FFIA usa esses dados para pesquisar o reconhecimento da atividade alimentar.</p><details class="fw-details"><summary>Modelos e processamento</summary><p>A cada 2 segundos, uma MobileNetV3 Small pré-treinada extrai características de 16 imagens. Seus pesos permanecem fixos. A GRU treinada no projeto recebe essas características e classifica a atividade alimentar. Os dois detectores Fishial localizam os peixes, e um rastreador acompanha seus deslocamentos. A detecção e a classificação alimentar são análises separadas.</p><p>Nos arquivos técnicos, os níveis não observada, baixa, moderada e alta aparecem como ausente, fraca, média e forte, respectivamente.</p></details></section>
<section class="fw-section" id="results-section"><h2>Resultados dos testes</h2><p>O modelo acertou a categoria em aproximadamente <strong>61 de cada 100 clipes</strong> no teste histórico. Em uma avaliação adicional com vídeos de outros três dias, acertou cerca de <strong>54 de cada 100</strong>. São amostras diferentes; esses percentuais não indicam o acerto de cada vídeo da demonstração.</p><p>O menor acerto foi na categoria moderada. Ainda faltam avaliações com outras câmeras, espécies e ambientes. Os testes realizados não permitem definir doses de ração, comprovar economia ou estimar a população de um tanque.</p><details class="fw-details"><summary>Dados e métricas dos testes</summary><p>A avaliação histórica utiliza 7.104 clipes separados por dia de gravação: 3.898 para treino, 1.705 para validação e 1.501 para teste. Os clipes de teste foram gravados em 22/06/2022.</p><div class="fw-table-scroll"><table class="fw-metrics"><caption>Teste histórico — GRU em uso, seed 42</caption><thead><tr><th scope="col">Acerto da categoria</th><th scope="col">Macro-F1</th><th scope="col">Identificação da categoria moderada</th></tr></thead><tbody><tr><td>60,56%</td><td>0,5903</td><td>18,31%</td></tr></tbody></table></div><p>Acurácia é o percentual de classificações corretas. Macro-F1 mede o desempenho dando o mesmo peso às quatro categorias; quanto mais próximo de 1, melhor. A última coluna mostra quantos clipes da categoria moderada foram reconhecidos corretamente, medida chamada recall nos relatórios.</p>'''
    independent=ROOT/'artifacts/independent-evaluation/results-v1/results.json'
    if independent.exists():
        evaluation=json.loads(independent.read_text())
        labels={'operational':'GRU operacional','candidate':'GRU retreinada','random_forest':'Random Forest'}
        rows=''.join('<tr><td>'+labels[key]+'</td><td>'+f'{100*m["accuracy"]:.2f}%'.replace('.',',')+'</td><td>'+f'{m["macro_f1"]:.4f}'.replace('.',',')+'</td><td>'+f'{100*m["classification_report"]["medium"]["recall"]:.2f}%'.replace('.',',')+'</td></tr>' for key,m in evaluation['models'].items())
        body+='<div class="fw-table-scroll"><table class="fw-metrics"><caption>Avaliação adicional — 224 clipes de 27 a 29/06/2022</caption><thead><tr><th scope="col">Modelo</th><th scope="col">Acerto da categoria</th><th scope="col">Macro-F1</th><th scope="col">Identificação da categoria moderada</th></tr></thead><tbody>'+rows+'</tbody></table></div><p>Também testei um retreinamento com maior peso para a categoria moderada. O resultado não atingiu os critérios para substituir o modelo em uso. Os 224 clipes foram selecionados antes de executar os modelos, e a avaliação usou as categorias publicadas no conjunto de pesquisa. Como os clipes são do mesmo contexto de gravação e podem ser semelhantes entre si, o resultado não representa o desempenho em outras fazendas.</p><p><a href="evaluation.md">Relatório da avaliação adicional</a>.</p>'
    body+='</details></section>'+learning_html()
    body+='''
<section id="tool-title" class="fw-section"><h2>Analisar um vídeo próprio</h2><p>Esta página reproduz exemplos já analisados. Para enviar outro vídeo, instale o FeedWatch no seu computador. A aplicação aceita gravações de 2 a 60 segundos, com até 100 MB. As instruções incluem a instalação do Python e dos modelos necessários.</p><div class="fw-document-links"><a href="reproducao.md">Instalação</a><a href="feedwatch-source.zip" download>Baixar código AGPL-3.0</a><a href="feedwatch-runtime.zip" download>Baixar pacote de execução</a><a href="evaluation.md">Relatório de avaliação</a><a href="methodology.md">Cálculo do panorama</a></div></section>
<section class="fw-section" id="credits"><h2>Fontes e créditos</h2><p>Dados de treinamento: AV-FFIA, de Meng Cui e colaboradores, sob CC BY 4.0. Vídeos do aquário e das carpas: Richard Muller e lam loi, no Pexels. Detector de peixes: Fishial. Os autores dessas fontes não têm vínculo com o FeedWatch nem endossam o projeto.</p><div class="fw-document-links"><a href="https://zenodo.org/records/11059975">Dados AV-FFIA</a><a href="https://github.com/FishMaster93/U-FFIA">Código da pesquisa U-FFIA</a><a href="https://www.pexels.com/video/fishes-swimming-in-aquarium-6472322/">Vídeo do aquário</a><a href="https://www.pexels.com/video/feeding-koi-fish-with-processed-fish-foods-3625177/">Vídeo das carpas</a><a href="THIRD_PARTY.md">Créditos e licenças</a></div></section></main><footer class="fw-footer"><span translate="no">FeedWatch</span><span>Projeto experimental · Código aberto AGPL-3.0</span></footer>'''
    page='<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#ffffff"><meta name="description" content="FeedWatch analisa a atividade alimentar de peixes em vídeo com um modelo treinado no AV-FFIA. Demonstração, resultados dos testes e código aberto."><title>FeedWatch — Análise da alimentação de peixes por vídeo</title><link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="styles.css"><style>body{margin:0;background:#fff}body>.fw-page{max-width:1200px;margin:auto;padding:0 32px}@media(max-width:600px){body>.fw-page{padding:0 20px}}</style><script src="app.js" defer></script></head><body><div class="fw fw-page">'+body+'</div></body></html>'
    page=page.replace('<script src="app.js" defer></script>','<script src="i18n.js" defer></script><script src="app.js" defer></script>')
    page=localize_html(page)
    (DIST/'i18n.js').write_text(i18n_javascript(),encoding='utf-8')
    page=page.replace('href="styles.css"','href="styles.css?v='+file_hash(DIST/'styles.css')[:12]+'"').replace('src="app.js"','src="app.js?v='+file_hash(DIST/'app.js')[:12]+'"').replace('src="i18n.js"','src="i18n.js?v='+file_hash(DIST/'i18n.js')[:12]+'"')
    (DIST/'index.html').write_text(page,encoding='utf-8')
    (DIST/'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="3" fill="#173b59"/><text x="8" y="24" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#fff">F</text></svg>',encoding='utf-8')
    print(f'Built {len(cases)} real examples at {DIST}',flush=True)


if __name__=='__main__':main()
