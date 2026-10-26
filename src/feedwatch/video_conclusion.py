"""Deterministic summaries of observed outputs; no external text generation or causal claims."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .data import CLASS_IDS,write_json
from .video_inference import PORTUGUESE
from .video_overlay import font_file


def periods(indices,fps):
    if not indices:return []
    result=[];start=previous=indices[0]
    for index in indices[1:]:
        if index!=previous+1:
            result.append({'start_seconds':start/fps,'end_seconds':(previous+1)/fps})
            start=index
        previous=index
    result.append({'start_seconds':start/fps,'end_seconds':(previous+1)/fps})
    return result


def build_summary(analysis,records):
    windows=analysis.get('windows',[])
    valid=[w for w in windows if w['status']=='ok']
    if any(w['end_seconds']-w['start_seconds']<2-1e-6 for w in valid) and analysis['mode']=='sequential':
        raise ValueError('Resumo não pode classificar janela incompleta')
    counts=Counter(w['predicted_class'] for w in valid)
    if set(counts)-set(CLASS_IDS):raise ValueError('Classe inválida no resumo')
    distribution={name:counts.get(name,0) for name in CLASS_IDS}
    winners=[name for name in CLASS_IDS if valid and distribution[name]==max(distribution.values())]
    if not winners:
        conclusion='Não há janelas válidas suficientes para concluir a intensidade alimentar prevista neste trecho.'
        predominant='Indisponível'
    elif len(winners)>1:
        names=' e '.join(PORTUGUESE[name] for name in winners)
        conclusion=f'Neste trecho, houve empate entre as intensidades alimentares previstas: {names.lower()}.'
        predominant='Empate: '+names
    else:
        predominant=PORTUGUESE[winners[0]]
        conclusion=f'Neste trecho, predominou a intensidade alimentar prevista {predominant.lower()} ({distribution[winners[0]]}/{len(valid)} janelas válidas).'
    metadata=analysis['metadata']
    available=[r['detected_count'] for r in records if r['detector_status']=='ok']
    no_tracking=[r['frame_index'] for r in records if r.get('confirmed_track_count',0)==0]
    errors=[r['frame_index'] for r in records if r['error']]
    return {'duration_seconds':metadata['duration_seconds'],'valid_analyzed_seconds':sum(w['end_seconds']-w['start_seconds'] for w in valid),
        'complete_valid_windows':len(valid),'window_count':len(windows),'class_distribution':distribution,
        'predominant_classes':winners,'predominant_text':predominant,'tie':len(winners)>1,'conclusion':conclusion,
        'detections_per_frame_min':min(available) if available else None,'detections_per_frame_max':max(available) if available else None,
        'detection_count_scope':'current filtered detections per frame, not total or unique fish in tank',
        'frames_without_confirmed_tracking':len(no_tracking),'seconds_without_confirmed_tracking':len(no_tracking)/metadata['fps'],
        'periods_without_confirmed_tracking':periods(no_tracking,metadata['fps']),
        'frames_with_suspended_tracks':sum(bool(r.get('suspended_track_ids')) for r in records),
        'unconfirmed_detection_observations':sum(not o.get('track_confirmed',False) for r in records for o in r['detections']),
        'available_speed_observations':sum(o['speed_source_px_per_second'] is not None for r in records for o in r['detections']),
        'displayed_speed_observations':sum(r.get('layout',{}).get('displayed_speed_labels',0) for r in records),
        'frames_with_visual_errors':len(errors),'visual_error_periods':periods(errors,metadata['fps']),
        'invalid_windows':[{'window_id':w['window_id'],'start_seconds':w['start_seconds'],'end_seconds':w['end_seconds'],'status':w['status'],'error':w['error']} for w in windows if w['status']!='ok'],
        'source_note':analysis.get('source_note',''),'template':'feedwatch-observed-video-v1','text_generation':'deterministic local Python template; no LLM/service',
        'tracking_reliability':'technical confirmation criterion, not verified individual identity or biological speed',
        'not_inferred':['hunger','satiety','health','physical_size','feeding_quantity','feeder_control','causal_relation_with_motion_or_counts']}


def render_card(summary,width,height,scale=1.):
    canvas=Image.new('RGB',(width,height),(13,32,40));draw=ImageDraw.Draw(canvas)
    x=width*.055;y=height*.045
    def line(value,fraction=.037,color=(241,247,245)):
        nonlocal y
        font=ImageFont.truetype(font_file(True),round(width*fraction*scale))
        words=value.split();text=''
        for word in words:
            candidate=(text+' '+word).strip()
            if draw.textlength(candidate,font=font)>width*.89 and text:
                draw.text((x,y),text,font=font,fill=color);y+=width*fraction*scale*1.35;text=word
            else:text=candidate
        draw.text((x,y),text,font=font,fill=color);y+=width*fraction*scale*1.5
    line('FeedWatch · resumo do trecho',.045,(255,224,60))
    line(f'{summary["duration_seconds"]:.1f} s de vídeo · {summary["complete_valid_windows"]} janelas válidas',.038)
    line('Distribuição das previsões:',.029,(187,208,218))
    d=summary['class_distribution']
    line(f'Ausente: {d["none"]} · Fraca: {d["weak"]}',.038)
    line(f'Média: {d["medium"]} · Forte: {d["strong"]}',.038)
    line('Conclusão: '+summary['predominant_text'],.043,(255,224,60))
    low,high=summary['detections_per_frame_min'],summary['detections_per_frame_max']
    line(f'Detecções por frame: {low}–{high}' if low is not None else 'Detecções indisponíveis',.033)
    line('Essa faixa não é o total de indivíduos do tanque.',.027,(187,208,218))
    line(f'Sem qualquer trilha confirmada: {summary["seconds_without_confirmed_tracking"]:.2f} s',.030)
    line(f'Suspensões parciais em {summary["frames_with_suspended_tracks"]} frames.',.027,(187,208,218))
    line(f'Janelas inválidas: {len(summary["invalid_windows"])} · frames com falha visual: {summary["frames_with_visual_errors"]}',.027)
    if summary['source_note']:line('Aplicação exploratória em condições não avaliadas.',.027,(187,208,218))
    line('Não comprova fome ou saciedade. Sem recomendação de alimentação.',.025,(187,208,218))
    if y>height*.97:
        if scale<.7:raise ValueError('Cartão excedeu a altura disponível; não cortar texto')
        return render_card(summary,width,height,scale*.9)
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def write_summary(output,summary,run):
    write_json(output/'summary.json',summary)
    table='\n'.join(f'| {name} · {PORTUGUESE[name]} | {summary["class_distribution"][name]} |' for name in CLASS_IDS)
    ranges='; '.join(f'{p["start_seconds"]:.2f}–{p["end_seconds"]:.2f} s' for p in summary['periods_without_confirmed_tracking']) or 'Nenhum'
    report=f'''# FeedWatch — resumo do vídeo contínuo

{summary['conclusion']}

- Duração do vídeo: **{summary['duration_seconds']:.2f} s**. Duração em janelas completas válidas: **{summary['valid_analyzed_seconds']:.2f} s**; **{summary['complete_valid_windows']}/{summary['window_count']} janelas válidas**.
- Predominância: **{summary['predominant_text']}**. Empate: **{'sim' if summary['tie'] else 'não'}**.
- Faixa de detecções por frame: **{summary['detections_per_frame_min']}–{summary['detections_per_frame_max']}**; não representa o total de indivíduos do tanque.
- Sem qualquer acompanhamento confirmado: **{summary['seconds_without_confirmed_tracking']:.2f} s** ({summary['frames_without_confirmed_tracking']} frames). Intervalos: {ranges}. Confirmação é um critério técnico; não prova identidade do peixe.
- Frames com alguma trilha suspensa: {summary['frames_with_suspended_tracks']}; falhas visuais: {summary['frames_with_visual_errors']}; janelas inválidas/incompletas: {len(summary['invalid_windows'])}. Detecções sem confirmação: {summary['unconfirmed_detection_observations']} observações; deslocamentos com histórico suficiente: {summary['available_speed_observations']}; efetivamente exibidos: {summary['displayed_speed_observations']} (os demais foram omitidos para evitar colisão de textos).

| Classe prevista | Janelas válidas |
|---|---:|
{table}

## Fonte e apresentação

{summary['source_note']}

Fonte: `{run['input_video']}`. Não houve concatenação de clipes para formar o trecho. O vídeo anotado mantém todos os pixels/resolução da filmagem e acrescenta um rodapé fora da imagem. O detector vê o frame completo original; a GRU mantém os 16 frames, recorte e normalização históricos, estado zero por janela. A previsão aparece somente depois do fim da janela de origem. O último intervalo completo pode aparecer somente no cartão final, depois de terminar o vídeo. O cartão é pós-análise e não entra na inferência. Modo diagnóstico opcional inclui recorte e fluxo; o vídeo principal os omite.

## Execução e arquivos

- Tempo do pipeline, incluindo exportação, verificação e cartão (carga de modelos quando necessária): **{run['total_processing_seconds']:.2f} s**; **{run['effective_input_fps']:.3f} FPS efetivo** ({run['rendered_frames']} frames reais / tempo). FPS de reprodução: **{run['metadata']['fps']:g}**. Não é uma afirmação de tempo real.
- Hardware: {run['hardware']}; dispositivo: `{run['food_model']['runtime_device']}`.
- Verificação da GRU ({run['reload_verification'].get('mode','new checkpoint instance')}): {run['reload_verification']['windows_matched']} janelas reproduzidas; maior diferença de scores {run['reload_verification']['max_score_difference']:.3g}.
- [MP4 anotado](annotated.mp4), [apresentação com cartão](presentation.mp4), [cartão](conclusion-card.png), [janelas](windows.csv), [frames detalhados](frames.jsonl), [frames CSV](frames.csv), [configuração](config.json), [execução](run.json).

Comando executado:

```powershell
{run['reproduce_command']}
```

Use uma pasta nova/vazia para repetir. Os scores não são confiança calibrada. O detector externo ainda pode perder peixes, agrupar indivíduos e confundir estruturas; filtros e aparência do vídeo não substituem um detector validado. Associações podem fragmentar: não se desenham posições previstas ou conexões sobre perdas/ambiguidades. Pixels/s são deslocamentos aparentes dos centros ao longo do histórico confirmado, não velocidades físicas. Nenhuma relação causal entre contagens, movimento e alimentação foi inferida.

Texto e cartão calculados pelo template local `{summary['template']}`, sem LLM ou serviço externo. Não houve treinamento, alteração alimentar ou reavaliação do teste. A conclusão descreve apenas as previsões deste trecho; não recomenda quantidade de ração ou acionamento de alimentador e não comprova fome, saciedade, saúde ou tamanho físico.
'''
    (output/'summary.md').write_text(report,encoding='utf-8')
