"""Plain-language presentation of recorded training and evaluation evidence."""
import json
from html import escape
from .portfolio_presentation import DIST,local_url
from .i18n import localize_html


def _baseline_html(local=False):
    path=DIST/'learning/evidence.json'
    if not path.exists():return '<section class="fw fw-section" id="learning-section"><h2>Treinamento do modelo</h2><p>Os registros de treinamento não estão incluídos nesta instalação. Eles estão disponíveis no relatório do projeto e na demonstração pública.</p></section>'
    data=json.loads(path.read_text(encoding='utf-8'))
    def url(name):return escape(local_url(DIST/name) if local else name,quote=True)
    def percent(value):return f'{100*value:.1f}%'.replace('.',',')
    comparison=data['regularization_comparison']
    labels={'small_regularized':'GRU menor com regularização','strong_regularized':'GRU com regularização maior',
            'small_strong_regularized':'GRU menor com regularização maior'}
    rows=''.join('<tr><th scope="row">'+labels[c['name']]+'</th><td>'+percent(c['mean_accuracy'])+'</td><td>'+f'{c["mean_macro_f1"]:.4f}'.replace('.',',')+'</td></tr>' for c in comparison['configurations'])
    epoch=data['selected_epoch'];train=percent(data['train_accuracy']);validation=percent(data['validation_accuracy'])
    baseline=f'{comparison["baseline_mean_macro_f1"]:.4f}'.replace('.',',')
    baseline_accuracy=percent(comparison['baseline_mean_accuracy'])
    return f'''<section class="fw fw-section fw-learning" id="learning-section"><h2>Treinamento do modelo</h2>
<p>Treinei uma rede <strong>GRU</strong> para classificar a atividade alimentar com <strong>3.898 clipes de dois dias do AV-FFIA</strong>. Usei outros 1.705 clipes, de um terceiro dia, para acompanhar o desempenho durante o treinamento e escolher a versão do modelo.</p><p>As caixas nos peixes são geradas por dois detectores Fishial já pré-treinados, separados da GRU. Os vídeos de aquário e carpas desta página não fizeram parte do treinamento da GRU.</p>
<figure class="fw-learning-figure"><h3>Acerto durante o treinamento</h3><picture><source media="(max-width:600px)" srcset="{url('learning/operational-learning-mobile.svg')}"><img src="{url('learning/operational-learning.svg')}" width="1000" height="420" loading="lazy" alt="Curva da GRU operacional: na etapa {epoch}, acerto de {train} no treino e {validation} na validação de outro dia."></picture><figcaption>O modelo em uso é o da etapa {epoch}, escolhida pelo melhor Macro-F1 na validação. Nessa etapa, o acerto foi de {train} nos vídeos de treino e {validation} nos de validação. O acerto menor nos vídeos de outro dia indica dificuldade para classificar imagens que não fizeram parte do treino.</figcaption></figure>
<figure class="fw-learning-figure"><h3>Acerto por categoria</h3><picture><source media="(max-width:600px)" srcset="{url('learning/class-recall-mobile.svg')}"><img src="{url('learning/class-recall.svg')}" width="900" height="340" loading="lazy" alt="Acerto por categoria nos 224 clipes de outros três dias. A categoria moderada teve somente 8,9% de identificação correta."></picture><figcaption>Percentual de clipes de cada categoria que o modelo classificou corretamente, em uma amostra de 224 clipes de outros três dias. A categoria moderada teve o menor acerto. A precisão das caixas nos peixes não foi medida nessa avaliação.</figcaption></figure>
<details class="fw-details"><summary>Testes com outras configurações</summary><p>Testei três configurações com maior regularização para reduzir a memorização dos vídeos de treino. Cada configuração foi treinada três vezes, com inicializações diferentes, totalizando nove treinos. A comparação usou os dados de treino e validação.</p><div class="fw-table-scroll"><table class="fw-metrics"><caption>Médias das três inicializações na validação</caption><thead><tr><th scope="col">Configuração</th><th scope="col">Acerto</th><th scope="col">Macro-F1</th></tr></thead><tbody><tr><th scope="row">GRU de referência</th><td>{baseline_accuracy}</td><td>{baseline}</td></tr>{rows}</tbody></table></div><p>Nenhuma configuração atingiu os critérios definidos para substituir o modelo em uso. Os dias reservados para o próximo teste continuam separados e não foram usados nessa escolha.</p><p>Os próximos testes devem incluir mais condições de filmagem e uma revisão dos erros da categoria moderada. O panorama facilita a leitura de um vídeo inteiro, mas não aumenta o acerto do modelo.</p></details>
<div class="fw-document-links"><a href="{url('evaluation.md')}">Avaliação e próximos testes</a><a href="{url('learning/evidence.json')}">Dados e origem dos gráficos</a><a href="{url('learning/operational-history.csv')}">Histórico do treinamento (CSV)</a><a href="{url('learning/confusion-matrix.csv')}">Tabela de acertos e erros (CSV)</a></div></section>'''


def _learning_html(local=False):
    baseline=_baseline_html(local)
    path=DIST/'learning/diverse-evidence.json'
    if not path.exists():return baseline
    data=json.loads(path.read_text(encoding='utf-8'))
    def url(name):return escape(local_url(DIST/name) if local else name,quote=True)
    def pct(value):return f'{100*value:.1f}%'.replace('.',',')
    def f1(value):return f'{value:.4f}'.replace('.',',')
    run=data['seed_42'];comparison=data['comparison'];heldout=data['heldout']
    promoted=data['operational_model_replaced']
    config=data['protocol']['configuration']
    train_days=len(config['existing_days']['train'])+len(config['new_days']['train'])
    val_days=len(config['existing_days']['val'])+len(config['new_days']['val'])
    if promoted:status='O novo modelo atingiu os critérios de validação e do teste reservado e passou a ser usado na aplicação.'
    elif heldout:status='O candidato passou pela validação, mas não atingiu todos os critérios no teste reservado. O modelo anterior continua em uso.'
    else:status='O novo treino não atingiu os critérios de melhoria na validação. O modelo anterior continua em uso, e o dia reservado para teste não foi aberto.'
    rows=''.join(f'<tr><th scope="row">{label}</th><td>{pct(metrics["accuracy"])}</td><td>{f1(metrics["macro_f1"])}</td><td>{pct(metrics["recalls"]["medium"])}</td></tr>'
                 for label,metrics in (('Modelo anterior',comparison['baseline']),('Novo treino',comparison['candidate'])))
    extra=''
    if heldout:
        heldout_rows=''.join(f'<tr><th scope="row">{label}</th><td>{pct(metrics["accuracy"])}</td><td>{f1(metrics["macro_f1"])}</td><td>{pct(metrics["classification_report"]["medium"]["recall"])}</td></tr>'
                            for label,metrics in (('Modelo anterior',heldout['baseline']),('Novo treino',heldout['candidate'])))
        extra=f'<h3>Teste no dia reservado</h3><p>Comparação dos dois modelos nos mesmos {heldout["candidate"]["examples"]} clipes de 18/06/2022, após a seleção na validação.</p><div class="fw-table-scroll"><table class="fw-metrics"><thead><tr><th scope="col">Modelo</th><th scope="col">Acerto</th><th scope="col">Macro-F1</th><th scope="col">Identificação da moderada</th></tr></thead><tbody>{heldout_rows}</tbody></table></div>'
    current=f'''<section class="fw fw-section fw-learning" id="learning-section"><h2>Treinamento com mais dias de gravação</h2>
<p>Ampliei o treino para {format(data['counts']['train'], ',').replace(',', '.')} clipes de {train_days} dias e a validação para {format(data['counts']['val'], ',').replace(',', '.')} clipes de {val_days} dias do AV-FFIA. A GRU manteve a mesma arquitetura, com amostragem equilibrada entre dia, sessão e categoria. O extrator MobileNetV3 Small permaneceu congelado.</p>
<p>{status}</p><figure class="fw-learning-figure"><h3>Acerto durante o novo treinamento</h3><picture><source media="(max-width:600px)" srcset="{url('learning/diverse-learning-mobile.svg')}"><img src="{url('learning/diverse-learning.svg')}" width="1000" height="420" loading="lazy" alt="Acerto por etapa do novo treino: treino e validação, inicialização 42."></picture><figcaption>Na inicialização 42, a etapa escolhida foi {run['selected_epoch']}, pelo melhor Macro-F1 médio entre os dias de validação. Nessa etapa, o acerto foi de {pct(run['training']['accuracy'])} no treino e {pct(run['validation']['accuracy'])} na validação.</figcaption></figure>
<details class="fw-details"><summary>Comparação entre os treinamentos</summary><p>Os dois modelos foram avaliados nos mesmos vídeos de validação, com três inicializações cada. A tabela dá o mesmo peso a cada dia de gravação e a cada inicialização.</p><div class="fw-table-scroll"><table class="fw-metrics"><caption>Validação — dias 17 e 21/06/2022</caption><thead><tr><th scope="col">Modelo</th><th scope="col">Acerto médio</th><th scope="col">Macro-F1 médio</th><th scope="col">Identificação da moderada</th></tr></thead><tbody>{rows}</tbody></table></div>{extra}<p>Esses vídeos são do mesmo local de pesquisa. O resultado não mede desempenho em outras fazendas nem economia de ração.</p></details>
<div class="fw-document-links"><a href="{url('learning/diverse-evidence.json')}">Dados do novo treinamento</a><a href="{url('learning/diverse-history.csv')}">Histórico do novo treino (CSV)</a><a href="{url('evaluation.md')}">Relatório de avaliação</a></div></section>'''
    if promoted:
        baseline=baseline.replace('id="learning-section"','id="initial-learning-section"').replace('Treinamento do modelo','Treinamento inicial').replace('O modelo em uso é','O modelo inicial é')
        return current+'<details class="fw-details"><summary>Consultar o treinamento inicial</summary>'+baseline+'</details>'
    current=current.replace('id="learning-section"','id="diverse-learning-section"')
    return baseline+'<details class="fw-details"><summary>Treino adicional com mais dias de gravação</summary>'+current+'</details>'


def _finetune_html(local=False):
    path=DIST/'learning/finetune-evidence.json'
    if not path.exists():return ''
    data=json.loads(path.read_text(encoding='utf-8'))
    def url(name):return escape(local_url(DIST/name) if local else name,quote=True)
    comparison=data['comparison'];results=data['results']
    status=('O ajuste foi aprovado nas comparações e passou a ser usado na aplicação.' if results['promoted'] else
            'O ajuste não atingiu todos os critérios de substituição. O modelo anterior continua em uso.')
    def f1(value):return f'{value:.4f}'.replace('.',',')
    def percent(value):return f'{100*value:.1f}%'.replace('.',',')
    old,new=comparison['baseline'],comparison['validation']
    rows=f'<tr><th scope="row">Macro-F1</th><td>{f1(old["mean_macro_f1"])}</td><td>{f1(new["mean_macro_f1"])}</td></tr>'
    rows+=''.join(f'<tr><th scope="row">{label}</th><td>{percent(old["mean_recalls"][name])}</td><td>{percent(new["mean_recalls"][name])}</td></tr>'
                  for name,label in (('none','Não observada'),('weak','Baixa'),('medium','Moderada'),('strong','Alta')))
    return f'''<details class="fw-details" id="finetune-experiment"><summary>Ajuste das camadas de imagem</summary>
<section class="fw fw-section fw-learning"><h2>Ajuste da MobileNet</h2>
<p>Treinei as últimas camadas da MobileNet junto com a GRU, usando os mesmos vídeos e pequenas variações de iluminação, cor e recorte. A comparação foi repetida com três inicializações.</p>
<p>{status}</p><div class="fw-table-scroll"><table class="fw-metrics"><caption>Validação de 21/06/2022: média das três execuções</caption>
<thead><tr><th scope="col">Medida</th><th scope="col">Modelo anterior</th><th scope="col">Após o ajuste</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>O Macro-F1 combina o desempenho nas quatro categorias. Vai de zero a um; quanto maior, melhor. Os percentuais mostram quantos exemplos de cada categoria foram reconhecidos corretamente.</p>
<figure class="fw-learning-figure"><picture><source media="(max-width:600px)" srcset="{url('learning/finetune-learning-mobile.svg')}"><img src="{url('learning/finetune-learning.svg')}" width="1000" height="420" loading="lazy" alt="Desempenho na validação durante os três ajustes da MobileNet."></picture>
<figcaption>Cada linha representa um treinamento. Os pontos marcam as etapas escolhidas; a etapa zero é o modelo antes do ajuste. A validação usa 1.705 clipes de outro dia.</figcaption></figure>
<div class="fw-document-links"><a href="{url('finetuning.md')}">Método do ajuste</a><a href="{url('learning/finetune-evidence.json')}">Resultados do ajuste</a><a href="{url('learning/finetune-history.csv')}">Histórico do ajuste (CSV)</a></div></section></details>'''


def learning_html(local=False):
    return localize_html(_learning_html(local)+_finetune_html(local))
