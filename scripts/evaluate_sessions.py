"""Prepare blind review forms, then evaluate saved whole-session predictions.

Run ``python scripts/evaluate_sessions.py prepare --help`` or ``evaluate --help``.
Final evaluation requires authorized, continuous 50–60 s recordings, two blind
reviewers (one with domain knowledge) and documented disagreement consensus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from feedwatch.session_evaluation import (CONSENSUS_FIELDS, REVIEW_FIELDS, evaluate_session,
    read_csv, read_review, review_grid, summarize_sessions, validate_prediction_record, validate_session_set)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _inspect_source(video):
    import cv2
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not capture.isOpened() or not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
            raise ValueError("Cannot read video metadata")
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (round(height), round(width)):
                raise ValueError("Video resolution changed during decoding")
            decoded += 1
        if decoded != round(count):
            raise ValueError("Full decode differs from video frame count")
        return dict(fps=fps, frame_count=decoded, width=round(width), height=round(height),
                    duration_seconds=decoded / fps, full_decode_verified=True)
    finally:
        capture.release()


def _form(session, video_path):
    sid = session["session_id"]
    rows = review_grid(sid, session["duration_seconds"])
    options = '<option value="">Selecionar</option>' + ''.join(f'<option value="{k}">{v}</option>' for k, v in
        (("none", "Não observada"), ("weak", "Baixa"), ("medium", "Moderada"),
         ("strong", "Alta"), ("unobservable", "Não é possível observar")))
    titles = dict(interval="Período", whole="Vídeo inteiro", start="Início", middle="Meio", end="Final")
    cells = []
    for i, row in enumerate(rows):
        labels = options + ('<option value="ambiguous">Sem predominância clara</option>' if row['kind'] != 'interval' else '')
        cells.append(f'<tr><td><button type="button" data-start="{row["start_seconds"]}">{titles[row["kind"]]}: '
                     f'{row["start_seconds"]:.2f}–{row["end_seconds"]:.2f} s</button></td>'
                     f'<td><select aria-label="Classificação {i+1}" data-label="{i}">{labels}</select></td>'
                     f'<td><input aria-label="Observações {i+1}" data-notes="{i}"></td></tr>')
    practice = '<p class="notice">FORMULÁRIO DE PRÁTICA. Este vídeo não pode comprovar o desempenho em sessões de 50–60 segundos.</p>' if session['purpose'] == 'practice' else ''
    # This file contains only the unmarked source and blank fields. Never embed predictions.
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    video_src = video_path.as_uri() if video_path.is_absolute() else video_path.as_posix()
    return '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisão da sessão — FeedWatch</title><style>body{font:16px Arial,sans-serif;color:#173044;max-width:1100px;margin:32px auto;padding:0 20px}video{width:100%;max-height:450px;background:#111}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}input,select,button{font:inherit;padding:8px;max-width:100%}.notice{padding:15px;background:#fff1cc}label{display:inline-block;margin:12px 14px 12px 0}button{cursor:pointer}td input{box-sizing:border-box;width:100%}@media(max-width:700px){td,th{padding:7px;font-size:13px}select{width:100%}}</style>
<h1>Revisão da sessão: ''' + html.escape(sid) + '''</h1>''' + practice + '''
<p>Observe a gravação original sem consultar os resultados do sistema. Preencha os períodos e, depois, a sua leitura do vídeo inteiro e das três partes. Os dois revisores devem trabalhar separadamente.</p>
<p>Não observada: sem sinais de alimentação; baixa, moderada e alta: intensidade aparente da atividade alimentar. Use “Não é possível observar” quando a imagem não permite julgar, e registre o motivo. Ausência de visibilidade não significa ausência de alimentação.</p>
<p>A leitura do vídeo inteiro e de cada parte é uma avaliação própria. Use “Sem predominância clara” quando não houver um nível predominante. Registre dúvidas e condições que atrapalharam a observação.</p>
<video controls preload="metadata" src="''' + html.escape(video_src, quote=True) + '''"></video>
<label>Revisor <select id="role"><option>a</option><option>b</option></select></label>
<label>Nome ou código <input id="reviewer" required></label>
<label><input type="checkbox" id="domain"> Tenho conhecimento do domínio</label>
<label><input type="checkbox" id="blind"> Não consultei as previsões do modelo</label>
<p>As respostas são salvas como rascunho neste navegador, separadas por revisor A/B. Cada revisor deve usar sua própria cópia. Baixe o CSV e salve-o como review-a.csv ou review-b.csv nesta pasta. Em caso de discordância, preencham juntos o rótulo e a justificativa em consensus.csv.</p>
<table><thead><tr><th>Trecho</th><th>Atividade alimentar</th><th>Observações</th></tr></thead><tbody>''' + ''.join(cells) + '''</tbody></table>
<p><button id="download" type="button">Baixar revisão CSV</button> <span id="status" role="status"></span></p>
<script>const rows=''' + payload + ''',fields=''' + json.dumps(REVIEW_FIELDS) + ''',draftBase=''' + json.dumps('feedwatch-review:' + sid + ':' + session['source_video_sha256']) + ''';
const draftKey=()=>draftBase+':'+document.querySelector('#role').value;
const capture=()=>({reviewer:document.querySelector('#reviewer').value,domain:document.querySelector('#domain').checked,blind:document.querySelector('#blind').checked,labels:rows.map((r,i)=>document.querySelector(`[data-label="${i}"]`).value),notes:rows.map((r,i)=>document.querySelector(`[data-notes="${i}"]`).value)});
const saveDraft=()=>{try{localStorage.setItem(draftKey(),JSON.stringify(capture()));document.querySelector('#status').textContent='Rascunho salvo neste navegador.';}catch{document.querySelector('#status').textContent='O navegador não permite salvar o rascunho. Baixe o CSV antes de fechar.';}};
const loadDraft=()=>{let d={};try{d=JSON.parse(localStorage.getItem(draftKey())||'{}');}catch{}document.querySelector('#reviewer').value=d.reviewer||'';document.querySelector('#domain').checked=!!d.domain;document.querySelector('#blind').checked=!!d.blind;rows.forEach((r,i)=>{document.querySelector(`[data-label="${i}"]`).value=d.labels?.[i]||'';document.querySelector(`[data-notes="${i}"]`).value=d.notes?.[i]||'';});document.querySelector('#status').textContent='';};
document.querySelectorAll('input,select:not(#role)').forEach(e=>e.addEventListener('input',saveDraft));document.querySelector('#role').addEventListener('change',loadDraft);loadDraft();
document.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>{document.querySelector('video').currentTime=Number(b.dataset.start);document.querySelector('video').play();});
document.querySelector('#download').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim(),blind=document.querySelector('#blind').checked,domain=document.querySelector('#domain').checked;if(!reviewer||!blind){document.querySelector('#status').textContent='Informe o revisor e confirme a revisão sem previsões.';return;}const values=rows.map((r,i)=>({...r,reviewer_id:reviewer,domain_expertise:domain,blinded:blind,label:document.querySelector(`[data-label="${i}"]`).value,notes:document.querySelector(`[data-notes="${i}"]`).value}));if(values.some(r=>!r.label)){document.querySelector('#status').textContent='Preencha todas as classificações.';return;}const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',csv='\ufeff'+[fields.map(quote).join(','),...values.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\\r\\n');const a=document.createElement('a'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='review-'+document.querySelector('#role').value+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.querySelector('#status').textContent='CSV gerado. Salve uma cópia antes de fechar a página.';};</script></html>'''


def prepare(args):
    output = args.output_dir.resolve()
    video = args.video.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing reviews are never overwritten")
    date.fromisoformat(args.date)
    metadata = _inspect_source(video)
    session = dict(session_id=args.session_id, camera_id=args.camera_id, recorded_date=args.date,
        duration_seconds=metadata["duration_seconds"], source_video=str(video), source_video_sha256=_hash(video),
        source_url=args.source_url, authorization=args.authorization, continuity_note=args.continuity_note,
        continuous=args.continuous, selection_locked_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_basis=args.selection_basis, predictions_unseen_at_selection=args.predictions_unseen,
        split=args.split, purpose="practice" if args.practice else "evaluation",
        review_a="review-a.csv", review_b="review-b.csv", consensus="consensus.csv", predictions="analysis.json",
        video_verification=metadata,
        continuity_verification="Declared by preparer; file metadata and hashing do not prove absence of cuts.")
    validate_session_set([session], allow_practice=args.practice)
    output.mkdir(parents=True, exist_ok=True)
    video_ref = video
    if args.copy_video:
        video_ref = Path('source' + video.suffix.lower())
        shutil.copyfile(video, output / video_ref)
        if _hash(output / video_ref) != session['source_video_sha256']:
            raise ValueError("Copied review source differs from original")
        session['source_video'] = video_ref.as_posix()
    rows = review_grid(args.session_id, metadata['duration_seconds'])
    for role in ('a', 'b'):
        _write_csv(output / f'review-{role}.csv', [dict(r, reviewer_id='', domain_expertise='', blinded='') for r in rows], REVIEW_FIELDS)
    _write_csv(output / 'consensus.csv', rows, CONSENSUS_FIELDS)
    _write_json(output / 'session.json', session)
    (output / 'review.html').write_text(_form(session, video_ref), encoding='utf-8')
    print(json.dumps(dict(manifest=str(output / 'session.json'), review_form=str(output / 'review.html'),
                         eligible_final=not args.practice, human_labels_filled=0), ensure_ascii=False))


def evaluate(args):
    sessions = [json.loads(p.read_text(encoding="utf-8")) for p in args.manifest]
    validate_session_set(sessions, allow_practice=args.allow_practice)
    results, provenance = [], []
    for manifest, session in zip(args.manifest, sessions):
        root = manifest.resolve().parent
        source = Path(session["source_video"])
        if not source.is_absolute():
            source = root / source
        if _hash(source) != session["source_video_sha256"]:
            raise ValueError("Source video changed after session selection")
        paths = {k: root / session[k] for k in ('review_a', 'review_b', 'consensus', 'predictions')}
        prediction_record = json.loads(paths['predictions'].read_text(encoding='utf-8'))
        windows, identity = validate_prediction_record(prediction_record, session)
        a, b = read_review(paths['review_a']), read_review(paths['review_b'])
        result = evaluate_session(session, windows, a, b, read_csv(paths['consensus']))
        result['model_identification'] = identity
        results.append(result)
        provenance.append(dict(session_id=session['session_id'], source_video_sha256=session['source_video_sha256'],
                               manifest_sha256=_hash(manifest), input_sha256={k: _hash(p) for k, p in paths.items()}))
    if len({r['model_identification']['checkpoint_sha256'] for r in results}) > 1:
        raise ValueError("Evaluate different model checkpoints in separate reports")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve evaluation records")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Human feeding-activity agreement on continuous recordings; not feed dosage, hunger or savings.",
                  inference="Review and continuity declarations are recorded; not independently certified.",
                  independence="Split leakage is checked only among the supplied manifests. Training-data independence must also be documented.",
                  practice_only=all(s.get('purpose') == 'practice' for s in sessions),
                  provenance=provenance, sessions=results, summary=summarize_sessions(results))
    _write_json(output / 'results.json', report)
    print(json.dumps(dict(results=str(output / 'results.json'), sessions=len(results),
                         practice_only=report['practice_only']), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare', help='Create blank, blind forms without reading any model output')
    for name in ('video', 'output-dir'):
        prepare_parser.add_argument('--' + name, required=True, type=Path)
    for name in ('session-id', 'camera-id', 'date', 'source-url', 'authorization', 'continuity-note', 'selection-basis'):
        prepare_parser.add_argument('--' + name, required=True)
    prepare_parser.add_argument('--continuous', action='store_true', help='Explicitly attest a continuous, uncut recording')
    prepare_parser.add_argument('--predictions-unseen', action='store_true', help='Attest selection before inspecting predictions')
    prepare_parser.add_argument('--split', choices=('development', 'final'), default='final')
    prepare_parser.add_argument('--practice', action='store_true', help='Practice only: allow other durations; never final validation')
    prepare_parser.add_argument('--copy-video', action='store_true', help='Copy the unmarked source and use a relative HTML video URL for an independent reviewer')
    prepare_parser.set_defaults(run=prepare)
    evaluation = sub.add_parser('evaluate', help='Import two blind CSV reviews, explicit consensus and predictions')
    evaluation.add_argument('--manifest', type=Path, nargs='+', required=True)
    evaluation.add_argument('--output-dir', type=Path, required=True)
    evaluation.add_argument('--allow-practice', action='store_true', help='Report practice sessions with an ineligible flag')
    evaluation.set_defaults(run=evaluate)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
