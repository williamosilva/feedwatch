"""Presentation only: historical aggregates and deterministic explanations of saved outputs."""

import json
from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import quote

from .data import file_hash
from .video_inference import PORTUGUESE

REPORTS = {
    'Avaliação final no dia reservado': Path('artifacts/experiments/final-test/report.md'),
    'Características de movimento e Random Forest': Path('artifacts/experiments/motion-full/report.md'),
    'MobileNet congelada e classificador linear': Path('artifacts/experiments/visual-full/report.md'),
    'Comparação controlada entre MLP e GRU': Path('artifacts/experiments/temporal-controlled/report.md'),
}


def number(value,places=0,language='pt'):
    text=f'{value:,.{places}f}'
    return text if language=='en' else text.replace(',','_').replace('.',',').replace('_','.')


def historical_data(root=Path('artifacts')):
    files={'splits':root/'splits/validation.json',
           'metrics':root/'experiments/final-test/models/gru_seed_42/metrics.json',
           'identification':root/'experiments/final-test/models/gru_seed_42/identification.json'}
    content={key:json.loads(path.read_text(encoding='utf-8')) for key,path in files.items()}
    splits=content['splits'];metrics=content['metrics'];model=content['identification']
    if not splits['valid'] or model['family']!='gru' or model['seed']!=42 or model['id']!='gru_seed_42':
        raise ValueError('Identificação histórica incompatível com a referência operacional')
    composition=splits['composition']
    dates=[set(composition[k]['dates']) for k in ('train','val','test')]
    if any(a&b for i,a in enumerate(dates) for b in dates[i+1:]):
        raise ValueError('As datas dos conjuntos históricos se sobrepõem')
    counts={key:composition[key]['videos'] for key in ('train','val','test')}
    if metrics['examples']!=counts['test']:raise ValueError('Cobertura histórica inconsistente')
    return {'dataset':'AV-FFIA','total':sum(counts.values()),'counts':counts,
            'dates':{key:composition[key]['dates'] for key in counts},'seed':42,
            'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],
            'sources':{key:{'file':str(path),'sha256':file_hash(path)} for key,path in files.items()}}


def historical_table(data,language='pt'):
    counts=data['counts']
    def dates(key):
        if language=='en':return ' and '.join(f'{year}-{int(month):02d}-{int(day):02d}'
                         for year,month,day in (value.split('_') for value in data['dates'][key]))
        return ' e '.join(f'{int(day):02d}/{int(month):02d}/{year}'
                         for year,month,day in (value.split('_') for value in data['dates'][key]))
    if language=='en':
        rows=[('Data',f'AV-FFIA · subset of {number(data["total"],language="en")} videos'),
              ('Split',f'Training: {number(counts["train"],language="en")} · Validation: {number(counts["val"],language="en")} · Test: {number(counts["test"],language="en")}'),
              ('Separate days',f'Training: {dates("train")} · Validation: {dates("val")} · Test: {dates("test")}'),
              ('Operational model','Frozen MobileNetV3 Small + GRU · seed 42'),
              ('Historical held-out test',f'Accuracy: {number(data["accuracy"]*100,2,"en")}% · Macro-F1: {number(data["macro_f1"],4,"en")}')]
        return ('<table class="experiment-table"><caption>Feeding classification data and results</caption><tbody>'+
                ''.join(f'<tr><th scope="row">{escape(label)}</th><td>{escape(value)}</td></tr>' for label,value in rows)+
                '</tbody></table>')
    rows=[('Dados',f'AV-FFIA · subconjunto de {number(data["total"])} vídeos'),
          ('Divisão',f'Treino: {number(counts["train"])} · Validação: {number(counts["val"])} · Teste: {number(counts["test"])}'),
          ('Dias distintos',f'Treino: {dates("train")} · Validação: {dates("val")} · Teste: {dates("test")}'),
          ('Modelo operacional','MobileNetV3 Small congelada + GRU · seed 42'),
          ('Teste reservado · histórico',f'Acurácia: {number(data["accuracy"]*100,2)}% · Macro-F1: {number(data["macro_f1"],4)}')]
    return ('<table class="experiment-table"><caption>Dados e resultados da classificação alimentar</caption><tbody>'+
            ''.join(f'<tr><th scope="row">{escape(label)}</th><td>{escape(value)}</td></tr>' for label,value in rows)+
            '</tbody></table>')


def report_links(language='pt'):
    english=['Final held-out day evaluation','Motion features and Random Forest',
             'Frozen MobileNet and linear classifier','Controlled MLP and GRU comparison']
    return '\n\n'.join(f'[{english[index] if language=="en" else label} (Markdown)](/gradio_api/file={quote(str(path.resolve()),safe="")})'
                       for index,(label,path) in enumerate(REPORTS.items()) if path.is_file())


def result_sections(details,language='pt'):
    """Only reformats recorded predictions; never receives a model or modifies results."""
    summary=details['coverage']
    valid=[w for w in details['windows'] if w['status']=='ok']
    counts=Counter(w['predicted_class'] for w in valid)
    if dict(counts)!= {c:n for c,n in summary['class_distribution'].items() if n}:
        raise ValueError('Resumo e previsões salvas são inconsistentes')
    if language=='en':
        names={'none':'not observed','weak':'low','medium':'moderate','strong':'high'}
        count=len(valid)
        prediction=(f'Analyzed **{count} two-second interval'+('' if count==1 else 's')+
                    f'** in a {number(summary["duration_seconds"],1,"en")}-second video. ')
        if valid:
            prediction+='Predictions: '+'; '.join(f'**{names[c]} in {n}**' for c,n in summary['class_distribution'].items() if n)+'. '
            largest=max(counts.values());winners=[key for key,n in counts.items() if n==largest]
            if len(winners)>1:
                prediction+='The intervals were tied between '+', '.join(names[key] for key in winners)+'. '
            else:
                prediction+=f'The most frequent prediction was {names[winners[0]]} ({largest}/{count} valid intervals). '
            changes=sum(a['predicted_class']!=b['predicted_class'] for a,b in zip(valid,valid[1:])
                        if a['end_seconds']==b['start_seconds'])
            if changes:prediction+=f'The prediction changed {changes} time'+('' if changes==1 else 's')+' between consecutive intervals.'
            elif len(counts)==1:prediction+='The prediction stayed the same across valid intervals.'
        else:prediction+='Feeding classification was unavailable; failed intervals were not assigned a category.'
        meaning=('Intensity is a prediction for the **scene**, calculated for each interval. '
                 'Boxes and trails are a separate fish detection analysis; they do not describe the feeding intensity of each fish.')
        if counts.get('none') or counts.get('weak'):
            meaning+=' Not observed or low predictions do not establish that the fish stopped feeding.'
        low,high=summary['detections_per_frame_min'],summary['detections_per_frame_max']
        limits=(f'The detector produced {low} to {high} detections per image. This range is not the total fish population of the tank. '
                if low is not None else 'Fish detections were unavailable. ')
        if summary['frames_with_suspended_tracks']:limits+='Tracking was interrupted, and temporary IDs may change. '
        if summary['invalid_windows']:
            n=len(summary['invalid_windows']);limits+=f'{n} incomplete or failed interval'+(' was' if n==1 else 's were')+' left unclassified. '
        if summary['frames_with_visual_errors']:
            n=summary['frames_with_visual_errors'];limits+=f'Visual analysis failed on {n} image'+('' if n==1 else 's')+'. '
        limits+='Continuous-video analysis is exploratory. Historical metrics do not measure accuracy on this recording.'
        return f'### Model predictions\n\n{prediction}\n\n### Interpretation\n\n{meaning}\n\n### Analysis limitations\n\n{limits}'
    words={0:'zero',1:'um',2:'dois',3:'três',4:'quatro',5:'cinco',6:'seis'}
    duration=number(summary['duration_seconds'],1)
    prediction=(f'Foram analisados **{words.get(len(valid),str(len(valid)))} trechos de dois segundos** '
                f'em um vídeo de {duration} segundos. ')
    if valid:
        prediction+='Previsões: '+ '; '.join(f'**{PORTUGUESE[c].lower()} em {n}**' for c,n in summary['class_distribution'].items() if n)+'. '
        prediction+=summary['conclusion']
        changes=sum(a['predicted_class']!=b['predicted_class'] for a,b in zip(valid,valid[1:])
                    if a['end_seconds']==b['start_seconds'])
        if changes:prediction+=f' Houve {changes} mudança(s) entre trechos consecutivos.'
        elif len(counts)==1:prediction+=' A previsão se manteve nos trechos válidos.'
    else:prediction+='A classificação alimentar ficou indisponível; não foi atribuída uma classe às falhas.'
    meaning=('A intensidade é uma previsão para a **cena**, calculada a cada trecho. '
             'As caixas e trilhas são uma análise separada de detecções de peixes; não indicam a intensidade alimentar de cada indivíduo.')
    if counts.get('none') or counts.get('weak'):
        meaning+=' Previsões ausente ou fraca não demonstram que os peixes deixaram de se alimentar.'
    low,high=summary['detections_per_frame_min'],summary['detections_per_frame_max']
    limits=(f'O detector produziu de {low} a {high} detecções por imagem; essa faixa não é o total de peixes do tanque. '
            if low is not None else 'As detecções ficaram indisponíveis. ')
    if summary['frames_with_suspended_tracks']:
        limits+='Houve perdas de acompanhamento, e os IDs temporários podem mudar. '
    if summary['invalid_windows']:
        limits+=f'{len(summary["invalid_windows"])} trecho(s) incompleto(s) ou com falha ficaram sem classificação. '
    if summary['frames_with_visual_errors']:
        limits+=f'Houve falha visual em {summary["frames_with_visual_errors"]} imagens. '
    limits+='O uso em vídeo contínuo é exploratório; as métricas históricas não medem o acerto desta gravação.'
    return (f'### O que o modelo previu\n\n{prediction}\n\n'
            f'### O que isso significa\n\n{meaning}\n\n'
            f'### Limites desta análise\n\n{limits}')


CSS='''
:root {color-scheme:light!important;scrollbar-gutter:stable}
body {background:#f7f8fa!important;color:#20262e!important}
.gradio-container {width:100%!important;min-width:0!important;max-width:1060px!important;margin:auto!important;padding:clamp(24px,3vw,36px) clamp(16px,2vw,24px) 28px!important;font-family:"Segoe UI",Arial,sans-serif!important;font-size:16px!important;color:#20262e!important;background:#f7f8fa!important;color-scheme:light!important}
.gradio-container > .main {padding:0!important}
.gradio-container > .main,.gradio-container .contain,.gradio-container > .main > .wrap {min-width:0!important;max-width:100%;width:100%}
.gradio-container .column {min-width:0!important;max-width:100%}
#results {min-width:0!important;width:100%}
#results [data-testid="json"] {max-width:100%;overflow:auto!important}
.gradio-container .prose {font-size:16px;line-height:1.65;color:#20262e}
.gradio-container .prose p {margin:0 0 14px}
#brand h1 {font-family:"Segoe UI",Arial,sans-serif;font-size:42px;font-weight:650;letter-spacing:-.035em;line-height:1.15;color:#20262e;margin:0 0 8px}
#brand h2 {font-size:22px;font-weight:400;line-height:1.4;color:#52606d;margin:0 0 18px}
#introduction {max-width:920px} #method {max-width:960px}
.experiment-table {border-collapse:collapse;width:100%;font-size:15px;line-height:1.5;margin:4px 0 0;background:#fff;border:1px solid #dce2e8}
.experiment-table caption {text-align:left;font-weight:600;color:#20262e;font-size:17px;padding:0 0 12px}
.experiment-table th,.experiment-table td {text-align:left;vertical-align:top;padding:10px 14px;border:0;border-bottom:1px solid #e5e9ee}
.experiment-table th {width:25%;font-weight:600;color:#394652}
#metric-note .prose {font-size:14px;color:#52606d;line-height:1.55}
#tool-title {border-top:1px solid #dce2e8;margin-top:18px;padding-top:24px}
#tool-title h2,#result-title h2 {font-size:24px;line-height:1.3;font-weight:600;color:#20262e}
#video-input {background:#fff;border:1px solid #dce2e8;border-radius:4px;min-height:128px}
#input-actions {gap:12px} #input-actions button {min-height:46px;box-shadow:none;white-space:normal}
#status .prose {color:#52606d;font-size:14px} #progress {min-height:28px}
#result-video {border:1px solid #dce2e8;border-radius:4px;background:#fff;overflow:hidden}
#result-video video {width:100%;max-height:76vh;object-fit:contain;background:#fff}
#explanation h3 {font-size:18px;line-height:1.4;font-weight:600;margin:20px 0 6px}
#downloads button {box-shadow:none;min-height:44px}
#prototype-note {border-top:1px solid #dce2e8;padding-top:16px;margin-top:10px}
#prototype-note .prose {font-size:14px;line-height:1.55;color:#52606d}
.gradio-container a {color:#24559b;text-decoration:underline;text-underline-offset:3px}
.gradio-container button,.gradio-container input {font-family:inherit}
.gradio-container button:focus-visible,.gradio-container a:focus-visible,.gradio-container input:focus-visible {outline:2px solid #24559b!important;outline-offset:3px}
footer {display:none!important}
@media(max-width:600px) {
 #brand h1 {font-size:34px} #brand h2 {font-size:20px}
 .experiment-table th,.experiment-table td {padding:9px 10px;font-size:14px}
 .experiment-table th {width:31%} #input-actions {flex-direction:column}
 #input-actions button {width:100%}
 #result-video video {max-height:none}
}
@media(prefers-reduced-motion:reduce) {.gradio-container * {scroll-behavior:auto!important}}
'''


CSS+='''
body,.gradio-container {background:#ffffff!important;color:#263746!important}
.gradio-container {max-width:1200px!important;padding:0 clamp(20px,2vw,32px) 24px!important}
#brand,#demo {border:0!important;background:transparent!important;padding:0!important}
#brand .fw h1 {font-family:Arial,Helvetica,sans-serif!important;font-size:38px;font-weight:600;line-height:1.25;letter-spacing:-.018em;color:#263746}
#brand .fw-header {flex-wrap:wrap}
#brand .fw-hero h1 {max-width:100%}
#brand>div,#demo>div {padding:0!important}
#brand .fw-hero p {margin-bottom:0}
#tool-title {margin-top:20px;padding-top:24px}
#video-input {min-height:110px!important;max-height:150px}
#video-input .wrap {min-height:90px!important}
#status {font-size:14px;color:#526170}
#project-section {border-top:1px solid #dce3e9;margin-top:22px;padding-top:24px}
@media(max-width:850px){#brand .fw h1{font-size:33px}}
@media(max-width:600px){.gradio-container{padding:0 20px 20px!important}#brand .fw h1{font-size:28px}}
'''


def light_theme():
    import gradio as gr
    theme=gr.themes.Base(primary_hue='blue',secondary_hue='blue',neutral_hue='slate',radius_size='sm',
                         font=['Segoe UI','Arial','sans-serif'],font_mono=['Consolas','monospace'])
    theme=theme.set(body_background_fill='#ffffff',body_text_color='#263746',body_text_size='16px',
                    body_text_color_subdued='#52606d',block_background_fill='#ffffff',block_border_color='#dce2e8',
                    block_radius='4px',block_shadow='none',block_label_text_color='#394652',
                    block_label_background_fill='#ffffff',block_title_text_color='#20262e',
                    input_background_fill='#ffffff',input_border_color='#bcc6d0',
                    button_primary_background_fill='#173b59',button_primary_background_fill_hover='#102c44',
                    button_primary_text_color='#ffffff',button_primary_border_color='#173b59',
                    button_secondary_background_fill='#ffffff',button_secondary_background_fill_hover='#edf2f8',
                    button_secondary_text_color='#24559b',button_secondary_border_color='#bcc6d0',
                    button_large_text_size='16px',button_large_radius='4px',checkbox_background_color_selected='#24559b',
                    checkbox_border_color_selected='#24559b')
    values=theme.to_dict()['theme']
    # Keep Gradio's supported dark variants equal to the light variants, including browser dark preference.
    return theme.set(**{key:values[key[:-5]] for key in values if key.endswith('_dark') and key[:-5] in values})


def platform_i18n():
    import gradio as gr
    text={'upload_text.click_to_upload':'Clique para selecionar um vídeo','upload_text.drop_file':'Arraste um vídeo até aqui',
          'upload_text.drop_video':'Arraste um vídeo até aqui','common.or':'ou',
          'common.complete':'Concluído','common.clear':'Limpar','common.download':'Baixar',
          'file.uploading':'Enviando arquivo…'}
    english={'upload_text.click_to_upload':'Click to select a video','upload_text.drop_file':'Drop a video here',
             'upload_text.drop_video':'Drop a video here','common.or':'or',
             'common.complete':'Complete','common.clear':'Clear','common.download':'Download',
             'file.uploading':'Uploading file…'}
    return gr.I18n(en=english,pt=text,**{'pt-BR':text})


# Retain the existing import for scripts that used the earlier presentation helper.
portuguese_i18n=platform_i18n
