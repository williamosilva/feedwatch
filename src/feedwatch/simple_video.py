"""Refined video command, reusing the original operational inference and dynamic export."""

import json
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from .data import CLASS_IDS,file_hash,write_json
from .dynamic_video import export_dynamic,load_dynamic
from .video_conclusion import build_summary,render_card,write_summary
from .video_export import H264Writer,verify_export
from .video_inference import OperationalPredictor,analyze_video,open_capture


class SimpleRuntime:
    """One frozen food predictor and one detector, reused by serial local UI calls."""
    def __init__(self, config_path=Path('config/simple_video.json'),
                 protocol=Path('artifacts/experiments/final-test'), device='auto'):
        started=time.perf_counter()
        from .portable_runtime import load_predictor
        self.predictor=load_predictor(protocol,device)
        self.config,self.detector=load_dynamic(config_path,self.predictor)
        self.config_path=Path(config_path).resolve()
        self.config_sha256=file_hash(self.config_path)
        self.load_seconds=time.perf_counter()-started


def append_card(output,analysis,card,seconds):
    metadata=analysis['export_verification']
    count=round(seconds*metadata['fps'])
    expected={**metadata,'frame_count':metadata['frame_count']+count,
              'duration_seconds':metadata['duration_seconds']+count/metadata['fps']}
    source=output/'annotated.mp4'
    pending=output/'presentation.pending.mp4'
    writer=H264Writer(pending,source,expected,analysis['source_media']['has_audio'],output/'presentation-encoder.log')
    cap=open_capture(source)
    try:
        for index in range(metadata['frame_count']):
            ok,image=cap.read()
            if not ok:raise ValueError(f'Falha lendo anotado no frame {index}')
            writer.write(image)
        for _ in range(count):writer.write(card)
    finally:
        cap.release();writer.close()
    verified=verify_export(pending,expected)
    final=output/'presentation.mp4';pending.replace(final)
    return {'path':str(final.resolve()),'sha256':file_hash(final),'card_seconds':count/metadata['fps'],
            'verification':verified,'card_starts_at_seconds':metadata['duration_seconds']}


def run_simple(video,output,config_path=Path('config/simple_video.json'),protocol=Path('artifacts/experiments/final-test'),
               device='auto',mode='sequential',diagnostic=False,motion_only=False,source_record=None,command='',
               *,runtime=None,progress=None,show_speed=False):
    started=time.perf_counter()
    note='Aplicação exploratória: condições de captura deste vídeo não foram avaliadas nos experimentos alimentares.'
    provenance=None
    if source_record:
        provenance=json.loads(source_record.read_text(encoding='utf-8'))
        if provenance['trimmed_sha256']!=file_hash(video):raise ValueError('Fonte difere do registro de seleção')
        if provenance.get('visible_date')=='2022-06-22':raise ValueError('Não executar inferência no dia de teste reservado')
        note=(f'Fonte oficial registrada em {provenance.get("visible_date","data não verificada")}; '
            f'trecho contínuo {provenance["source_start_seconds"]:g}–{provenance["source_stop_seconds"]:g} s. '
            'Aplicação exploratória: dia, ângulo e codificação diferem das condições avaliadas.')
    if runtime is None:
        from .portable_runtime import load_predictor
        predictor=load_predictor(protocol,device)
        config,detector=load_dynamic(config_path,predictor,motion_only)
    else:
        if runtime.config_path!=Path(config_path).resolve() or runtime.config_sha256!=file_hash(config_path):
            raise ValueError('Configuração mudou após carregar modelos; reinicie a ferramenta')
        predictor,detector=runtime.predictor,runtime.detector
        config=deepcopy(runtime.config)
        if motion_only:detector=None
    config['presentation']['show_speed']=show_speed
    result=export_dynamic(video,output,predictor,config,detector,mode,simple=True,diagnostic=diagnostic,source_note=note,progress=progress)
    if not result.get('output_video'):return result
    write_json(output/'config.json',config)
    if provenance:write_json(output/'source.json',provenance)
    records=[json.loads(line) for line in (output/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=build_summary(result,records)
    # Reload and recalculate from the same raw source with the original, unchanged pipeline.
    if runtime is None:
        from .portable_runtime import load_predictor
        reloaded=load_predictor(protocol,device)
    else:reloaded=predictor
    replay_progress=(lambda phase,n,total:progress('replay',n,total)) if progress else None
    replay=analyze_video(video,reloaded,mode,progress=replay_progress)
    if len(replay['windows'])!=len(result['windows']):raise ValueError('Janelas diferem na recarga')
    max_difference,matched=0.,0
    for saved,fresh in zip(result['windows'],replay['windows']):
        if saved['status']!=fresh['status']:raise ValueError('Status difere após recarga')
        if saved['status']=='ok':
            if saved['predicted_id']!=fresh['predicted_id']:raise ValueError('Classe difere após recarga')
            values=[saved['scores'][name] for name in CLASS_IDS]
            other=[fresh['scores'][name] for name in CLASS_IDS]
            np.testing.assert_allclose(values,other,atol=1e-6,rtol=1e-5)
            max_difference=max(max_difference,float(np.max(np.abs(np.array(values)-other))))
            matched+=1
    predictor.verify_frozen();reloaded.verify_frozen()
    result['reload_verification']={'windows_matched':matched,'max_score_difference':max_difference,'raw_source_redecoded':True,
        'checkpoint_reloaded':runtime is None,'mode':'new checkpoint instance' if runtime is None else 'shared frozen instance; no model reload'}
    dims=result['export_verification']
    card=render_card(summary,dims['width'],dims['height'])
    cv2.imwrite(str(output/'conclusion-card.png'),card)
    if progress:progress('card',0,1)
    result['presentation']=append_card(output,result,card,config['presentation']['card_seconds'])
    if progress:progress('card',1,1)
    result['total_processing_seconds']=time.perf_counter()-started
    result['effective_input_fps']=result['rendered_frames']/result['total_processing_seconds']
    result['reproduce_command']=command
    write_json(output/'run.json',result)
    write_summary(output,summary,result)
    page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FeedWatch · trecho contínuo</title><style>body{margin:0;background:#0d2028;color:#fff;font:16px Segoe UI,sans-serif}main{max-width:1100px;margin:auto;padding:16px}video{width:100%;max-height:85vh}a{color:#ffe03c}</style><main><h1>FeedWatch · DURATION s de vídeo</h1><video controls playsinline preload="metadata" poster="POSTER" src="presentation.mp4"></video><p>Aplicação exploratória · IDs temporários · resumo após o trecho</p><p><a href="summary.md">Resumo</a> · <a href="annotated.mp4">Somente vídeo anotado</a> · <a href="frames.jsonl">Resultados por frame</a></p></main></html>'
    (output/'preview.html').write_text(page.replace('DURATION',f'{result["metadata"]["duration_seconds"]:g}').replace('POSTER',f'frame_{result["metadata"]["frame_count"]//2:05d}.jpg'),encoding='utf-8')
    return result
