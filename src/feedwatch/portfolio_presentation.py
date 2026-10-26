"""Shared presentation for the local application and the static portfolio.

Media and predictions are prepared from real saved runs, never invented in the UI.
"""
import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote
from .video_overview import summarize_video
from .i18n import i18n_javascript, localize_html

ASSETS=Path(__file__).parent/'assets'
DIST=Path('sites/feedwatch/dist')

WORKSPACE='''<div class="fw">
<div class="fw-demo-heading"><h2 data-demo-heading>Demonstração</h2><p data-demo-intro>Escolha um vídeo para ver a classificação da atividade alimentar e as marcações nos peixes. Os exemplos já estão analisados.</p></div>
<section class="fw-workspace" aria-label="Demonstração interativa">
<div class="fw-toolbar"><span class="fw-presentation-brand" translate="no">FeedWatch</span><div class="fw-source"><label for="fw-source">Vídeo de exemplo</label><select id="fw-source" name="example" data-source></select></div><div class="fw-toolbar-actions"><div class="fw-mode" aria-label="Marcações no vídeo"><button type="button" data-mode="original" aria-pressed="false">Sem marcações</button><button type="button" data-mode="annotated" aria-pressed="true">Com marcações</button></div><button class="fw-present" type="button" data-present aria-pressed="false" aria-label="Abrir demonstração em tela cheia" hidden>Modo apresentação</button></div></div>
<p class="fw-playback-heading">Vídeo e classificação</p>
<div class="fw-main"><div class="fw-screen"><video controls playsinline muted preload="metadata"></video><span class="fw-watermark" data-mode-label>Peixes identificados pelo sistema</span><button class="fw-start" type="button" data-start>Reproduzir vídeo</button></div>
<aside class="fw-insight" aria-label="Resultado no momento do vídeo"><div class="fw-current-result" role="status" aria-live="polite" aria-atomic="true"><p class="fw-label">Atividade do período</p><p class="fw-prediction" data-prediction>Inicie o vídeo</p><p class="fw-window" data-window>Resultado após 2 s</p></div><div class="fw-counter"><p class="fw-label" data-count-label>Peixes detectados</p><span class="fw-number" data-count>—</span></div></aside></div>
<div class="fw-timeline"><div class="fw-timeline-heading"><span>Atividade ao longo do vídeo</span><span data-clock>0,0 s</span></div><div class="fw-segments" data-segments></div><label class="fw-seek-label"><span>Posição do vídeo</span><input class="fw-seek" type="range" name="video-position" data-seek min="0" max="12" step="0.001" value="0" aria-label="Posição do vídeo em segundos"></label></div><p class="fw-footnote"><span data-coverage></span>. Cada parte corresponde a 2 segundos. Selecione uma parte para consultar seu resultado.</p>
<p class="fw-reference" data-reference hidden></p><p class="fw-presentation-status" data-presentation-status role="status" hidden></p></section>
<section class="fw-overview" aria-labelledby="fw-overview-title" data-overview>
<div class="fw-overview-heading"><h2 id="fw-overview-title">Panorama do vídeo</h2><p data-overview-intro>Resumo de todos os períodos analisados.</p></div>
<div class="fw-overview-grid"><div><p class="fw-label">Atividade estimada predominante</p><p class="fw-overview-value" data-overview-prediction>—</p><p data-overview-conclusion></p><p class="fw-overview-coverage" data-overview-coverage></p></div><div><h3>Tempo em cada nível</h3><div class="fw-distribution" data-distribution></div><p class="fw-overview-note">Percentuais do tempo classificado, não da chance de acerto.</p></div></div>
<div class="fw-evolution"><h3>Início, meio e final</h3><ol class="fw-phases" data-phases></ol><p class="fw-evolution-conclusion" data-evolution></p></div>
</section>
<details class="fw-details fw-scenario" data-scenario><summary>Comparar quantidades de ração</summary><p>Informe quanto de ração você costuma fornecer e quanto deseja simular para o mesmo período. O cálculo mostra a diferença de quantidade e de custo. As duas quantidades são definidas por você.</p>
<form class="fw-scenario-form" data-scenario-form><div class="fw-scenario-fields"><label>Quantidade habitual (kg)<input type="number" min="0" step="any" inputmode="decimal" name="usual-kg" data-usual required></label><label>Quantidade do cenário (kg)<input type="number" min="0" step="any" inputmode="decimal" name="scenario-kg" data-proposed required></label><label>Preço por kg (R$), opcional<input type="number" min="0" step="any" inputmode="decimal" name="price-per-kg" data-unit-price></label></div><button class="fw-present" type="submit">Comparar cenário</button><p class="fw-scenario-output" data-scenario-output role="status" aria-live="polite" hidden></p></form>
<p class="fw-scenario-note">O modelo não indica a quantidade adequada de ração. A diferença de custo é uma simulação.</p></details>
<details class="fw-details fw-result-details"><summary>Detalhes do resultado</summary><div class="fw-result-explanations"><div><h3>Atividade neste período</h3><p data-meaning>Inicie o vídeo para ver a classificação de cada período.</p></div><div><h3>Contagem na imagem</h3><p data-count-note>Quantidade na imagem atual. Não representa o total de peixes do tanque.</p></div></div><div class="fw-overview-detections"><h3>Contagem média por imagem</h3><p class="fw-overview-count" data-overview-count>—</p><p data-overview-count-note></p></div><p class="fw-overview-changes" data-overview-changes></p><p>O panorama soma o tempo classificado em cada nível. A predominância do vídeo exige pelo menos 80% da duração classificada. Para comparar início, meio e final, o vídeo é dividido em três partes iguais. Cada parte precisa ter pelo menos 4 segundos, dois resultados e 80% do tempo classificado. Um nível só predomina se ocupar mais da metade do tempo classificado. Esses critérios permitem exibir a comparação, mas ainda não foram validados como indicadores de manejo.</p><p data-scope></p><p class="fw-analysis-note" data-badge>Exemplo já analisado</p></details>
<p class="fw-credits" data-credit></p>
<section class="fw-guide" id="method-section" aria-labelledby="fw-guide-title"><h2 id="fw-guide-title">Leitura dos resultados</h2><p>O modelo classifica a atividade alimentar em quatro níveis. A classificação é uma estimativa a partir das imagens; ela não mede a fome dos peixes ou a quantidade de alimento necessária.</p>
<dl class="fw-levels"><div data-level="none"><dt>Não observada</dt><dd>Sem sinais de atividade alimentar reconhecidos pelo modelo.</dd></div><div data-level="weak"><dt>Baixa</dt><dd>Pouca atividade alimentar estimada no período.</dd></div><div data-level="medium"><dt>Moderada</dt><dd>Atividade alimentar estimada em nível intermediário.</dd></div><div data-level="strong"><dt>Alta</dt><dd>Atividade alimentar estimada como intensa.</dd></div></dl>
<div class="fw-reading-notes"><div><h3>Classificação a cada 2 segundos</h3><p>O resultado aparece ao fim de cada período de 2 segundos. Por isso, o nível mostrado pode mudar durante a reprodução. Se sobrar menos de 2 segundos no fim do vídeo, essa parte fica sem classificação.</p></div><div><h3>Marcações nos peixes</h3><p>Os retângulos indicam onde o detector encontrou peixes. As linhas mostram seus deslocamentos, e os números ajudam a acompanhá-los enquanto estão visíveis. Um peixe pode receber outro número ao reaparecer. Essas marcações não confirmam que ele está comendo.</p></div></div>
</section></div>'''

HEADER='''<div class="fw"><a class="fw-skip" href="#demo">Pular para a demonstração</a><header class="fw-header"><a class="fw-brand" href="#"><span translate="no">FeedWatch</span><span class="fw-brand-context">Aquicultura</span></a><nav class="fw-nav" aria-label="Navegação principal"><a href="#demo">Demonstração</a><a href="#method-section">Interpretação</a><a href="#project-section">Pesquisa</a><a href="#learning-section">Treinamento</a><a href="#tool-title">Instalação</a></nav></header><div class="fw-hero"><p class="fw-kicker">Projeto experimental</p><h1>Análise da alimentação de peixes por vídeo</h1><p class="fw-lead">Treinei um modelo com vídeos do AV-FFIA para classificar a atividade alimentar dos peixes. O FeedWatch analisa gravações em períodos de dois segundos e reúne os resultados em um resumo do vídeo.</p><p class="fw-intro-note">O objetivo é avaliar ajustes no fornecimento de ração para reduzir desperdícios. A economia ainda precisa ser medida em testes de manejo.</p><div class="fw-document-links"><a href="https://arxiv.org/html/2309.05058v3#S5">Estudo AV-FFIA/U-FFIA</a><a href="#tool-title">Analisar um vídeo próprio</a></div></div></div>'''

LANGUAGE_SWITCH='''<label class="fw-language" for="fw-language"><span>Idioma</span><select id="fw-language" name="language" data-language-select aria-label="Idioma"><option value="en" translate="no">English</option><option value="pt" translate="no">Português</option></select></label>'''
HEADER=localize_html(HEADER.replace('</nav></header>','</nav>'+LANGUAGE_SWITCH+'</header>'))
WORKSPACE=localize_html(WORKSPACE)


def local_url(path):
    return '/gradio_api/file='+quote(str(Path(path).resolve()),safe='')


def demo_payload():
    path=DIST/'data.json'
    if not path.exists():return {'cases':[],'message':'A demonstração está sendo preparada. Você já pode analisar um vídeo abaixo.'}
    data=json.loads(path.read_text(encoding='utf-8'))
    for case in data['cases']:
        for key in ('original','annotated','poster'):
            if case.get(key) and not case[key].startswith(('https://','http://')):case[key]=local_url(DIST/case[key])
    return data


def result_payload(result):
    details=result['details'];meta=details['metadata']
    frames=[]
    path=Path(result['directory'])/'frames.jsonl'
    if path.exists():
        frames=[{'count':row.get('detected_count')} for row in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())]
    return {'live':True,'cases':[{'id':'seu-video','title':'Seu vídeo · análise concluída',
        'original':local_url(details['local_app']['input_copy']),'annotated':local_url(result['video']),
        'poster':'','windows':deepcopy(details['windows']),'fps':meta['fps'],'duration':meta['duration_seconds'],
        'frames':frames,'overview':summarize_video(details['windows'],meta['duration_seconds'],frames),
        'scope':'A atividade alimentar e a identificação de peixes são analisadas separadamente. O sistema pode errar.',
        'credit':'Seu arquivo, analisado neste computador. Os resultados dos testes do projeto não medem o acerto desta gravação.'}]}


def widget_js():
    return i18n_javascript()+'\n'+(ASSETS/'portfolio.js').read_text(encoding='utf-8')+'\nmountFeedWatch(element, props.value); watch("value", () => mountFeedWatch(element, props.value));'


def widget_css():
    return (ASSETS/'portfolio.css').read_text(encoding='utf-8')


def font_head():
    return ''
