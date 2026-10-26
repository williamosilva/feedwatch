"""Thin localhost Gradio wrapper around the existing CLI video pipeline."""

import json
import shutil
import threading
import time
import uuid
import zipfile
from html import escape
from datetime import datetime, timezone
from pathlib import Path

from .data import file_hash, write_json
from .local_example import DIRECTORY, EXAMPLE, prepare_example
from .local_summary import plain_summary
from .simple_video import SimpleRuntime, run_simple
from .video_inference import inspect_video, open_capture, plan_windows
from .research_presentation import (CSS, REPORTS, historical_data, historical_table,
                                    light_theme, portuguese_i18n, report_links, result_sections)
from .portfolio_presentation import (HEADER, WORKSPACE, DIST, demo_payload, result_payload,
                                     widget_css, widget_js, font_head)
from .learning_presentation import learning_html

LOCAL_TRANSLATIONS=json.loads((Path(__file__).parent/'assets/local-translations.json').read_text(encoding='utf-8'))


def local_text(source):
    """English is the initial presentation language; technical records stay unchanged."""
    return LOCAL_TRANSLATIONS.get(source,source)


def language_pair(english,portuguese):
    return ('<div class="fw-local-copy" data-fw-local-language="en" translate="no">'+english+'</div>'
            '<div class="fw-local-copy" data-fw-local-language="pt" translate="no" hidden style="display:none">'+portuguese+'</div>')


def localized_markdown(source,english=None):
    from markdown_it import MarkdownIt
    renderer=MarkdownIt('commonmark',{'html':False})
    return language_pair('<div class="prose">'+renderer.render(local_text(source) if english is None else english)+'</div>',
                         '<div class="prose">'+renderer.render(source)+'</div>')


def localized_status(source,**values):
    safe={key:escape(str(value)) for key,value in values.items()}
    return language_pair(local_text(source).format(**safe),source.format(**safe))


def localized_error(error):
    source=str(error)
    if source in LOCAL_TRANSLATIONS:return localized_status(source)
    return '<code translate="no">'+escape(source)+'</code>'


def local_i18n_javascript():
    """Translate native Gradio copy without remounting controls or model sessions."""
    return r'''() => {
      const catalog = __CATALOG__;
      const reverse = new Map(Object.entries(catalog).map(([pt,en]) => [en,pt]));
      const normalize = (value) => value.replace(/\s+/g,' ').trim();
      const canonical = (value) => reverse.get(value) || value;
      const texts = new WeakMap(), attributes = new WeakMap();
      let queued = false, api = null;
      const root = document.querySelector('.gradio-container');
      if (!root) return;
      const skip = '#brand,#demo,#learning-evidence,#technical-data,script,style,pre,code,textarea,[translate="no"],[data-fw-i18n],.file-preview,.file-name,.filename,.orig-name';
      function translated(source) {
        const original = normalize(source), key = canonical(original);
        let value = api.language === 'pt' ? key : api.t(key);
        if (value === key && catalog[key] && api.language !== 'pt') value = catalog[key];
        if (value === original || (value === key && key === original)) {
          for (const [pt,en] of Object.entries(catalog).filter(([pt]) => pt.endsWith(':'))) {
            if (original.startsWith(pt)) value = (api.language === 'pt' ? pt : en) + original.slice(pt.length);
            else if (original.startsWith(en)) value = (api.language === 'pt' ? pt : en) + original.slice(en.length);
          }
        }
        if (value === original) return source;
        return source.slice(0,source.length-source.trimStart().length)+value+source.slice(source.trimEnd().length);
      }
      function refresh() {
        if (!api) return;
        root.querySelectorAll('[data-fw-local-language]').forEach(element => {
          const visible = element.dataset.fwLocalLanguage === api.language;
          element.hidden = !visible;
          const display = visible ? '' : 'none';
          if (element.style.display !== display) element.style.display = display;
        });
        const walker = document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT|NodeFilter.SHOW_TEXT,{
          acceptNode(node) {
            if (node.nodeType === Node.ELEMENT_NODE && node.matches(skip)) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
          }
        });
        for (let node=walker.nextNode();node;node=walker.nextNode()) {
          if (node.nodeType === Node.TEXT_NODE) {
            if (!node.data.trim()) continue;
            let saved = texts.get(node);
            if (!saved || saved.last !== node.data) saved = {source:node.data,last:node.data};
            const next = translated(saved.source);
            saved.last = next; texts.set(node,saved);
            if (node.data !== next) node.data = next;
          } else {
            let saved = attributes.get(node);
            if (!saved) {saved={};attributes.set(node,saved);}
            for (const key of ['aria-label','title','placeholder']) {
              if (!node.hasAttribute(key)) continue;
              const value = node.getAttribute(key);
              let record = saved[key];
              if (!record || record.last !== value) record = {source:value,last:value};
              const next = translated(record.source); record.last=next;saved[key]=record;
              if (value !== next) node.setAttribute(key,next);
            }
          }
        }
        const upload = root.querySelector('#video-input .upload-container');
        if (upload) {
          const label = translated('Selecionar ou arrastar vídeo');
          if (upload.getAttribute('aria-label') !== label) upload.setAttribute('aria-label',label);
        }
      }
      function schedule() {
        if (queued) return;
        queued=true;requestAnimationFrame(() => {queued=false;refresh();});
      }
      const observer = new MutationObserver(schedule);
      observer.observe(root,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['aria-label','title','placeholder']});
      let attempts = 0;
      function connect() {
        if (!window.FeedWatchI18n) {
          if (++attempts < 100) setTimeout(connect,100);
          return;
        }
        api=window.FeedWatchI18n;
        const unsubscribe=api.subscribe(schedule);
        window.FeedWatchLocalI18n={refresh:schedule,disconnect(){observer.disconnect();unsubscribe();}};
        schedule();
      }
      connect();
    }'''.replace('__CATALOG__',json.dumps(LOCAL_TRANSLATIONS,ensure_ascii=True))

PHASES={'prepare':'Preparando vídeo','food':'Analisando imagens',
        'render':'Renderizando resultado','verify':'Verificando vídeo',
        'replay':'Conferindo previsões','card':'Finalizando vídeo',
        'save':'Salvando arquivos'}


class LocalService:
    def __init__(self,output_root=Path('artifacts/local-app-runs'),config=Path('config/simple_video.json'),
                 protocol=Path('artifacts/experiments/final-test'),device='auto',runtime_factory=SimpleRuntime):
        self.output_root=Path(output_root).resolve()
        self.config,self.protocol,self.device=Path(config),Path(protocol),device
        self.runtime_factory=runtime_factory
        self.runtime=None
        self.lock=threading.Lock()
        self.model_load_count=0

    def analyze(self,path,diagnostic=False,show_speed=False,progress=None):
        # Also protects direct Python/API callers, independently of Gradio's queue.
        with self.lock:
            started=time.perf_counter()
            identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
            output=self.output_root/identifier
            events=[];last_sent=0.
            def advance(phase,completed,total):
                nonlocal last_sent
                elapsed=time.perf_counter()-started
                events.append({'phase':phase,'completed':completed,'total':total,'elapsed_seconds':elapsed})
                if progress is not None and (completed in (0,total) or elapsed-last_sent>=.15):
                    progress((completed,total) if total else None,desc=local_text(PHASES[phase]),unit='frames' if phase in ('food','render','verify','replay') else 'steps')
                    last_sent=elapsed
            try:
                advance('prepare',0,3)
                if not path:raise ValueError('Selecione um vídeo ou use o vídeo de exemplo.')
                source=Path(path).resolve()
                if not source.is_file():raise ValueError('O arquivo de vídeo não está disponível.')
                if source.stat().st_size>100*1024*1024:raise ValueError('Use um vídeo de até 100 MB.')
                if '2022_6_22' in source.parts:raise ValueError('O dia reservado para teste não é usado nesta ferramenta de demonstração.')
                cap=open_capture(source)
                try:meta=inspect_video(cap)
                finally:cap.release()
                if meta['duration_seconds']>60:raise ValueError('Use um vídeo de até 60 segundos.')
                if meta['width']*meta['height']>3840*2160:raise ValueError('Use um vídeo de até 3840 × 2160 pixels.')
                if not any(w['status']=='pending' for w in plan_windows(meta,'sequential')):
                    raise ValueError('O vídeo precisa de ao menos uma janela completa e válida de dois segundos, com 16 frames distintos.')
                advance('prepare',1,3)
                # Retain the exact upload independently of Gradio's temporary cache.
                staged=self.output_root.parent/'local-app-inputs'/identifier/('input'+source.suffix.lower())
                staged.parent.mkdir(parents=True)
                shutil.copy2(source,staged)
                signature=file_hash(source)
                if signature!=file_hash(staged):raise ValueError('O arquivo mudou durante a cópia da entrada.')
                advance('prepare',2,3)
                load_started=time.perf_counter()
                runtime_reused=self.runtime is not None
                if self.runtime is None:
                    self.runtime=self.runtime_factory(self.config,self.protocol,self.device)
                    self.model_load_count+=1
                load_seconds=time.perf_counter()-load_started
                advance('prepare',3,3)
                record=DIRECTORY/'source.json'
                provenance=(record if record.exists() and json.loads(record.read_text(encoding='utf-8'))['trimmed_sha256']==signature else None)
                command=(f'.\\.venv\\Scripts\\python.exe -m feedwatch annotate-simple --video "{staged}" '
                         f'--output-dir "{output}-replay" --config "{self.config}" --device {self.device}'
                         +(' --diagnostic' if diagnostic else '')+(' --show-speed' if show_speed else '')
                         +(f' --source-record "{provenance}"' if provenance else ''))
                result=run_simple(staged,output,self.config,self.protocol,self.device,source_record=provenance,
                                  command=command,runtime=self.runtime,progress=advance,
                                  diagnostic=diagnostic,show_speed=show_speed)
                if not result.get('output_video'):raise ValueError(result.get('error','Não foi possível concluir a exportação.'))
                advance('save',0,1)
                summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
                explanation=plain_summary(summary,result)
                result['local_app']={'model_load_count':self.model_load_count,'model_load_seconds_this_request':load_seconds,
                    'runtime_reused':runtime_reused,
                    'models_loaded_once_per_process':True,'gpu_execution_serialized':True,'input_copy':str(staged),
                    'source_upload_name':source.name,'input_sha256':signature,'output_directory':str(output),
                    'diagnostic':bool(diagnostic),'show_speed':bool(show_speed)}
                result['local_app']['request_processing_seconds']=time.perf_counter()-started
                result['local_app']['request_effective_input_fps']=result['rendered_frames']/result['local_app']['request_processing_seconds']
                write_json(output/'run.json',result)
                report=explanation+('\n\n---\n\nTempo de processamento: '
                    f'{result["local_app"]["request_processing_seconds"]:.2f} s; '
                    f'{result["local_app"]["request_effective_input_fps"]:.2f} frames/s de processamento; '
                    f'{meta["fps"]:g} FPS de reprodução. Isso não comprova inferência em tempo real. '
                    'A previsão alimentar aparece apenas depois do fim do trecho de origem; a última janela termina no fim do vídeo e consta no resumo.\n\n'
                    'Identificação dos modelos, configuração e resultados completos: arquivos do pacote técnico.\n\n'
                    f'Comando equivalente (pasta nova):\n\n```powershell\n{command}\n```\n')
                (output/'report.md').write_text(report,encoding='utf-8')
                details={key:result.get(key) for key in ('status','metadata','windows','visual_errors','food_model','detector',
                         'dynamic_config','timings','hardware','export_verification','reload_verification','source_note','local_app')}
                details['coverage']=summary
                details['rendered_frames']=result['rendered_frames']
                write_json(output/'details.json',details)
                # Gradio/orjson requires string object keys (checkpoint class maps use int keys).
                details=json.loads((output/'details.json').read_text(encoding='utf-8'))
                advance('save',1,1)
                write_json(output/'progress.json',events)
                with zipfile.ZipFile(output/'technical-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
                    for name in ('run.json','analysis.json','config.json','source.json','summary.json','summary.md',
                                 'report.md','details.json','windows.csv','frames.csv','frames.jsonl','progress.json'):
                        if (output/name).exists():archive.write(output/name,name)
                return {'video':str(output/'annotated.mp4'),'report':str(output/'report.md'),
                        'summary':explanation,'details':details,'archive':str(output/'technical-results.zip'),
                        'directory':str(output),'seconds':result['local_app']['request_processing_seconds']}
            except Exception as error:
                output.mkdir(parents=True,exist_ok=True)
                write_json(output/'failure.json',{'status':'unavailable','message':str(error),
                    'elapsed_seconds':time.perf_counter()-started,'predicted_class':None,'progress':events})
                raise


def build_app(service=None):
    import gradio as gr
    service=service or LocalService()
    with gr.Blocks(title='FeedWatch',analytics_enabled=False) as app:
        gr.HTML(HEADER,elem_id='brand',apply_default_css=False,css_template=widget_css(),head=font_head())
        viewer=gr.HTML(value=demo_payload(),html_template=WORKSPACE,css_template=widget_css(),
                       js_on_load=widget_js(),apply_default_css=False,elem_id='demo')
        gr.HTML(localized_markdown('## Analisar um vídeo próprio\n\nSelecione uma gravação de 2 a 60 segundos, '
                    'com até 100 MB. O resultado aparecerá na área de vídeo acima.'),elem_id='tool-title')
        video_input=gr.File(label=local_text('Selecionar vídeo'),file_types=['video'],type='filepath',elem_id='video-input')
        with gr.Row(elem_id='input-actions'):
            analyze=gr.Button(local_text('Analisar vídeo'),variant='primary')
            example=gr.Button(local_text('Usar vídeo de exemplo'),variant='secondary')
        gr.HTML(localized_markdown('O processamento é realizado neste computador.'))
        status=gr.HTML(localized_status('Selecione um vídeo ou use o exemplo.'),html_template='<div role="status" aria-live="polite">${value}</div>',elem_id='status')
        with gr.Accordion(local_text('Opções avançadas'),open=False):
            speed=gr.Checkbox(label=local_text('Mostrar deslocamento aparente em pixels/s'),value=False)
            diagnostic=gr.Checkbox(label=local_text('Modo diagnóstico: recorte da GRU e movimento da imagem'),value=False)
        progress_display=gr.Markdown('',elem_id='progress')
        with gr.Column(visible=False,elem_id='results') as result_group:
            gr.HTML(localized_markdown('## Resultado da análise'),elem_id='result-title')
            result_video=gr.Video(label=local_text('Vídeo anotado'),visible=False,interactive=False,format='mp4',buttons=[],elem_id='result-video')
            with gr.Row(elem_id='downloads'):
                download_video=gr.DownloadButton(local_text('Baixar vídeo'),interactive=False,variant='secondary')
                download_report=gr.DownloadButton(local_text('Baixar relatório'),interactive=False,variant='secondary')
            with gr.Accordion(local_text('Ver detalhes da análise'),open=False):
                explanation=gr.HTML(elem_id='explanation')
                gr.HTML(localized_markdown('Resultados por trecho, cobertura, falhas, configuração e identificação dos modelos. '
                            'Scores não são confiança calibrada. O detector não tem uma acurácia medida neste vídeo.'))
                gr.HTML(localized_markdown('### Resultados e rastreabilidade','### Results and provenance'))
                technical=gr.JSON(label=local_text('Resultados e rastreabilidade'),show_label=False,elem_id='technical-data')
                archive=gr.File(label=local_text('Baixar arquivos técnicos'),interactive=False)
        gr.HTML(localized_markdown('## Pesquisa de referência\n\nA ração é um dos principais custos da piscicultura. '
                    'A [NOAA](https://www.fisheries.noaa.gov/insight/feeds-aquaculture) descreve o uso de câmeras para acompanhar o consumo e evitar o fornecimento em excesso. '
                    'Esse é o problema que motivou o FeedWatch.\n\n'
                    'Para treinar o modelo, usei parte do [AV-FFIA, de Meng Cui e colaboradores](https://arxiv.org/html/2309.05058v3#S5). '
                    'O conjunto contém vídeos classificados por intensidade alimentar, com orientação de um técnico. '
                    'O estudo U-FFIA usa esses dados para pesquisar o reconhecimento da atividade alimentar.'),elem_id='project-section')
        with gr.Accordion(local_text('Modelos e processamento'),open=False):
            gr.HTML(localized_markdown('A análise utiliza 16 imagens a cada dois segundos. Uma MobileNetV3 Small congelada extrai '
                        'representações; uma GRU treinada no projeto estima a categoria de atividade alimentar. '
                        'Os dois detectores Fishial e o rastreador identificam e acompanham os peixes separadamente. '
                        'Nos arquivos técnicos, as categorias conservam os nomes ausente, fraca, média e forte, '
                        'apresentadas na interface como não observada, baixa, moderada e alta.'))
        with gr.Accordion(local_text('Dados, resultados e limites do experimento'),open=False):
            try:
                history=historical_data()
                gr.HTML(language_pair(historical_table(history,'en'),historical_table(history)),elem_id='experiment-overview',apply_default_css=False)
            except (ValueError,OSError,KeyError):gr.HTML(localized_markdown('Resultados históricos indisponíveis. Consulte os relatórios abaixo.'))
            gr.HTML(localized_markdown('O teste histórico não mede o acerto do vídeo enviado. O recall da classe média foi **18,31%**. '
                        'A GRU não demonstrou superioridade consistente sobre a Random Forest nem benefício claro da ordem temporal. '
                        'Um dia reservado não representa outras fazendas ou câmeras. Scores não são confiança calibrada.'))
            gr.HTML(localized_markdown(report_links(),report_links('en')))
        gr.HTML(learning_html(local=True),css_template=widget_css(),apply_default_css=False,elem_id='learning-evidence')
        gr.HTML(localized_markdown('Protótipo experimental. A classificação descreve intensidade alimentar prevista; '
                    'não determina fome, saciedade ou dosagem de ração.'),elem_id='prototype-note')

        def example_ui(progress=gr.Progress()):
            progress(None,desc=local_text('Preparando vídeo'))
            try:
                portable=Path('artifacts/runtime.json')
                if portable.exists():path=Path(json.loads(portable.read_text(encoding='utf-8'))['example'])
                else:path,_=prepare_example()
            except Exception as error:raise gr.Error(local_text('Exemplo indisponível:')+' '+local_text(str(error))) from error
            return str(path.resolve()),localized_status('Exemplo selecionado. Clique em Analisar vídeo.')

        def analyze_ui(path,diagnostic,show_speed,progress=gr.Progress()):
            # Clear an earlier result immediately, including download paths, on each request.
            yield (None,'',gr.update(value=None,interactive=False),gr.update(value=None,interactive=False),None,None,
                   localized_status('Preparando vídeo…'),gr.update(visible=False),'',{'cases':[],'message':'Analisando seu vídeo. Aguarde a conclusão abaixo.'},
                   gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False))
            try:
                result=service.analyze(path,diagnostic,show_speed,progress)
                yield (result['video'],localized_markdown(result_sections(result['details']),result_sections(result['details'],'en')),gr.update(value=result['video'],interactive=True),
                       gr.update(value=result['report'],interactive=True),result['details'],result['archive'],
                       localized_status('Análise concluída em {seconds} segundos. Veja o comparador no início da página.',seconds=f'{result["seconds"]:.1f}'),gr.update(visible=True),'',
                       result_payload(result),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True))
            except Exception as error:
                yield (None,'',gr.update(value=None,interactive=False),gr.update(value=None,interactive=False),None,None,
                       localized_status('Análise indisponível.')+localized_error(error)+localized_status('Selecione um vídeo legível com ao menos dois segundos.'),
                       gr.update(visible=False),'',{'cases':[],'message':'Análise indisponível. Selecione outro vídeo e tente novamente.'},
                       gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True))

        def reset_ui(path):
            return (None,'',gr.update(value=None,interactive=False),gr.update(value=None,interactive=False),None,None,
                    localized_status('Vídeo selecionado. Clique em Analisar vídeo.' if path else 'Selecione um vídeo ou use o exemplo.'),
                    gr.update(visible=False),'',demo_payload())

        example.click(example_ui,outputs=[video_input,status],concurrency_limit=1,concurrency_id='gpu',show_progress_on=[status])
        result_outputs=[result_video,explanation,download_video,download_report,technical,archive,status,result_group,progress_display,viewer]
        video_input.change(reset_ui,inputs=[video_input],outputs=result_outputs,queue=False)
        analyze.click(analyze_ui,inputs=[video_input,diagnostic,speed],
                      outputs=result_outputs+[video_input,analyze,example],
                      concurrency_limit=1,concurrency_id='gpu',api_name='analyze_video',show_progress_on=[progress_display])
        app.load(None,js=local_i18n_javascript())
    app.queue(default_concurrency_limit=1,max_size=8)
    return app


def launch_local(port=7860,inbrowser=True,device='auto'):
    import gradio as gr
    service=LocalService(device=device)
    app=build_app(service)
    app.launch(server_name='127.0.0.1',server_port=port,inbrowser=inbrowser,share=False,
               allowed_paths=[str(DIRECTORY.resolve()),str(DIST.resolve()),str(service.output_root),
                              str((service.output_root.parent/'local-app-inputs').resolve())]+[str(p.resolve()) for p in REPORTS.values() if p.is_file()],
               theme=light_theme(),css=CSS,footer_links=[],i18n=portuguese_i18n(),
               enable_monitoring=False,ssr_mode=False,max_file_size='100mb',run_history=False)
