"""Real validation footage montage. Cuts and independent tracker lifetimes are explicit."""

import html
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from .data import CLASS_IDS, data_root, file_hash, read_manifest, write_json
from .dynamic_overlay import transition_image
from .dynamic_report import write_dynamic_report
from .dynamic_video import export_dynamic, hardware, load_dynamic
from .temporal_data import validate_chronology
from .video_demo import select_validation
from .video_export import H264Writer, hidden_process, prepare_output, verify_export
from .video_inference import OperationalPredictor
from .visual import cache_signature, load_embedding_cache


def make_montage(output, entries, metadata):
    # No inference occurs across clips or title cards. Output clips already have isolated state.
    transitions = output/"transitions"
    transitions.mkdir()
    segments, timeline, current_frame = [],[],0
    fps = metadata["fps"]
    card_frames = max(1,round(.4*fps))
    for number, entry in enumerate(entries,1):
        card = transitions/f"{number:02d}.mp4"
        expected = {**metadata,"frame_count":card_frames,"duration_seconds":card_frames/fps}
        writer = H264Writer(card,card,expected,False,transitions/f"{number:02d}.log")
        try:
            image = transition_image(metadata["width"],metadata["height"],number,len(entries),
                entry["relative_path"],entry["class_name"])
            for _ in range(card_frames):
                writer.write(image)
        finally:
            writer.close()
        segments.append(card.resolve())
        timeline.append({"kind":"transition","clip_number":number,"start_frame":current_frame,
            "end_frame_exclusive":current_frame+card_frames,"start_seconds":current_frame/fps,
            "end_seconds":(current_frame+card_frames)/fps,"food_class":None,"detected_count":None,
            "relative_path":entry["relative_path"],"tracker_reset":True})
        current_frame += card_frames
        path = (output/entry["folder"]/"annotated.mp4").resolve()
        analysis = json.loads((path.parent/"analysis.json").read_text(encoding="utf-8"))
        meta = analysis["metadata"]
        if any(meta[key] != metadata[key] for key in ("width","height","fps")) or analysis["source_media"]["has_audio"]:
            raise ValueError("Montagem atual exige clipes de mesma resolução/FPS e sem áudio; não alterar contratos silenciosamente")
        segments.append(path)
        timeline.append({"kind":"real_clip","clip_number":number,"start_frame":current_frame,
            "end_frame_exclusive":current_frame+meta["frame_count"],"start_seconds":current_frame/fps,
            "end_seconds":(current_frame+meta["frame_count"])/fps,"relative_path":entry["relative_path"],
            "food_mode":"clip (offline complete analysis)","food_class":entry["predicted_class"],
            "dataset_label":entry["class_name"],"tracker_reset":True})
        current_frame += meta["frame_count"]
    listing = output/"concat.txt"
    # FFmpeg concat quoting; absolute paths are constructed here, never shell-expanded.
    listing.write_text("".join("file '"+path.as_posix().replace("'","'\\''")+"'\n" for path in segments),encoding="utf-8")
    pending = output/"montage.pending.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","warning","-n","-f","concat","-safe","0",
        "-i",str(listing),"-map","0:v:0","-c","copy","-an","-movflags","+faststart",str(pending)]
    result = subprocess.run(command,capture_output=True,timeout=60,**hidden_process())
    (output/"montage-encoder.log").write_bytes(result.stderr)
    if result.returncode:
        raise ValueError("FFmpeg não concluiu montagem; consulte montage-encoder.log")
    expected = {**metadata,"frame_count":current_frame,"duration_seconds":current_frame/fps}
    verified = verify_export(pending,expected)
    final = output/"feedwatch-real-montage.mp4"
    pending.replace(final)
    pd.DataFrame(timeline).to_csv(output/"montage-timeline.csv",index=False)
    return {"path":str(final.resolve()),"sha256":file_hash(final),"verification":verified,
            "independent_clips":len(entries),"transition_frames_each":card_frames,"ffmpeg_command":command,
            "timeline":"montage-timeline.csv","inference_across_cuts":False,"audio":"none in selected sources"}


def write_page(output, entries, detector_enabled=True):
    rows = "".join(f'<tr><td>{n:02d}</td><td>{html.escape(e["relative_path"])}</td><td>{e["class_name"]}</td>'
        f'<td>{e["predicted_class"]}</td><td><a href="{e["folder"]}/annotated.mp4">Vídeo</a> · '
        f'<a href="{e["folder"]}/frames.jsonl">Frames</a> · <a href="{e["folder"]}/windows.csv">Janelas</a></td></tr>'
        for n,e in enumerate(entries,1))
    document = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>FeedWatch · vídeo real</title><style>
body{margin:0;background:#eef3f3;color:#17343e;font:16px/1.55 "Segoe UI",Arial,sans-serif}main{max-width:1200px;margin:auto;padding:28px 24px}h1{font-size:30px;margin-bottom:8px}p{max-width:1000px}video{display:block;width:100%;background:#0c1e25;border-radius:6px}a{color:#126176}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #c8d5d9}td:nth-child(2){overflow-wrap:anywhere}.scroll{overflow-x:auto}.note{border-left:3px solid #28787e;padding-left:16px}a:focus-visible,video:focus-visible{outline:3px solid #865d26;outline-offset:4px}</style><main><h1>FeedWatch · peixes reais em movimento</h1>
<p>Montagem de 12 clipes independentes de validação, seed 42, sem seleção por acerto. As transições identificam cada corte. IDs, trilhas, fluxo e gráfico reiniciam em cada clipe.</p>
<video controls playsinline preload="metadata" poster="04_weak/frame_00025.jpg"><source src="feedwatch-real-montage.mp4" type="video/mp4"></video>
<p><a href="feedwatch-real-montage.mp4">Abrir MP4</a> · <a href="report.md">Relatório</a> · <a href="demo_report.json">Medições</a> · <a href="verification.json">Reprodução da GRU</a> · <a href="montage-timeline.csv">Linha do tempo</a></p>
<p class="note"><strong>Quatro origens distintas.</strong> A GRU seed 42 classifica a cena inteira no recorte histórico, em modo <em>Análise do clipe completo</em> offline. O Fishial pré-treinado propõe caixas de peixes no frame completo. O rastreador associa essas caixas com IDs temporários, trilhas curtas e deslocamento aparente em pixels/s. O fluxo óptico produz mapa, vetores azuis e gráfico de movimento da imagem; água e reflexos também contribuem.</p>
<p>Caixas podem falhar, agrupar indivíduos ou conter falsos positivos. A contagem é de detecções, não um censo de peixes. IDs não são identidades permanentes; deslocamentos incluem instabilidade das caixas. Scores estão nos registros e não são confiança calibrada. A classe alimentar não é atribuída a cada peixe.</p>
<p>Intensidade alimentar observada — protótipo experimental. Não comprova fome, saciedade nem segurança para controlar um alimentador. Reprodução a 25 FPS não significa inferência em tempo real. Não houve nova avaliação do teste.</p>
<div class="scroll"><table><thead><tr><th>Clipe</th><th>Fonte real</th><th>Rótulo do dataset</th><th>Previsão da GRU</th><th>Arquivos</th></tr></thead><tbody>ROWS</tbody></table></div></main></html>'''.replace("ROWS",rows)
    if not detector_enabled:
        document = document.replace("Quatro origens distintas.", "Componentes independentes.")
        document = document.replace("IDs, trilhas, fluxo e gráfico reiniciam em cada clipe.",
            "Fluxo e gráfico reiniciam em cada clipe. Detector e rastreador estão desligados.")
        document = document.replace("O Fishial pré-treinado propõe caixas de peixes no frame completo. O rastreador associa essas caixas com IDs temporários, trilhas curtas e deslocamento aparente em pixels/s.",
            "Detector desligado explicitamente: esta exportação não produz caixas, contagens ou IDs de rastreamento.")
        document = document.replace("Caixas podem falhar, agrupar indivíduos ou conter falsos positivos. A contagem é de detecções, não um censo de peixes. IDs não são identidades permanentes; deslocamentos incluem instabilidade das caixas.",
            "O movimento da imagem não permite contar ou identificar peixes.")
    (output/"index.html").write_text(document,encoding="utf-8")


def generate_dynamic_demo(root, splits, output, config_path, seed=42, device="auto",
                          protocol=Path("artifacts/experiments/final-test"),motion_only=False):
    if seed != 42:
        raise ValueError("A montagem controlada reutiliza a seleção anterior de seed 42")
    started = time.perf_counter()
    root = data_root(root)
    predictor = OperationalPredictor(protocol,device)
    prepare_output(output,predictor)
    rows = select_validation(read_manifest(splits/"val.csv"),seed)
    rows.to_csv(output/"selection.csv",index=False)
    write_json(output/"selection_protocol.json",{"seed":seed,"split":"val 2022_6_21","per_class":3,
        "prediction_or_detection_filter":False,"method":"same select_validation as prior MVP",
        "val_sha256":file_hash(splits/"val.csv"),"continuous_source":"known 5603 train/val clips each 2 s; no continuous source found",
        "config_sha256":file_hash(config_path)})
    config,detector = load_dynamic(config_path,predictor,motion_only)
    write_json(output/"config.json",config)
    old = pd.read_csv(Path(predictor.identification["checkpoint"]).parent/"val"/"predictions.csv").set_index("relative_path")
    entries = []
    for number,row in enumerate(rows.to_dict("records"),1):
        folder = f'{number:02d}_{row["class_name"]}'
        result = export_dynamic(root/row["relative_path"],output/folder,predictor,config,detector,"clip",
            f'Clipe {number:02d}/12 · independente · IDs locais')
        if not result.get("output_video"):
            raise ValueError(f"Demonstração interrompida, sem ocultar falha: {row['relative_path']}: {result.get('error')}")
        prediction,previous = result["windows"][0],old.loc[row["relative_path"]]
        if previous.class_name != row["class_name"] or int(previous.class_id) != row["class_id"]:
            raise ValueError("Rótulo/ID/caminho históricos incompatíveis")
        scores = np.array([prediction["scores"][name] for name in CLASS_IDS])
        expected = np.array([previous[f"prob_{name}"] for name in CLASS_IDS])
        if prediction["predicted_id"] != int(previous.predicted_id):
            raise ValueError("GRU diverge da inferência existente")
        np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
        prior_path = Path("artifacts/demos/validation-mvp")/folder/"analysis.json"
        prior_match = None
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if Path(prior["input_video"]).resolve() != (root/row["relative_path"]).resolve():
                raise ValueError("MVP histórico e seleção atual divergem")
            np.testing.assert_allclose(scores,[prior["windows"][0]["scores"][name] for name in CLASS_IDS],atol=1e-6,rtol=1e-5)
            prior_match = prediction["predicted_id"] == prior["windows"][0]["predicted_id"]
        entries.append({**row,"folder":folder,"predicted_class":prediction["predicted_class"],
            "predicted_id":prediction["predicted_id"],"historical_prediction_match":True,"prior_mvp_prediction_match":prior_match,
            "maximum_score_difference":float(np.max(np.abs(scores-expected))),"processing_seconds":result["processing_seconds"],
            "effective_processing_fps":result["effective_processing_fps"],"counters":result["counters"],
            "timings":result["timings"],"metadata":result["metadata"],"visual_errors":len(result["visual_errors"])})
        print(f'Visual {number}/12: {row["class_name"]} -> {prediction["predicted_class"]}; '
              f'{result["processing_seconds"]:.2f}s, {result["effective_processing_fps"]:.2f} FPS; '
              f'{result["counters"]["detection_observations"]} detecções-frame',flush=True)
    predictor.verify_frozen()
    montage = make_montage(output,entries,entries[0]["metadata"])
    # Independent checkpoint reload and historical, chronologically verified frame embeddings.
    reloaded = OperationalPredictor(protocol,device)
    replay = []
    for entry in entries:
        signature = cache_signature(root,entry["relative_path"],reloaded.identification["extraction_config_id"])
        cached,warning = load_embedding_cache(Path("artifacts/visual-cache"),signature,16)
        if cached is None:
            raise ValueError(f"Cache histórico indisponível para recarga independente: {warning}")
        validate_chronology(cached["details"],reloaded.extraction)
        prediction = reloaded.predict_embeddings(cached["embeddings"])
        saved = json.loads((output/entry["folder"]/"analysis.json").read_text(encoding="utf-8"))["windows"][0]
        if prediction["predicted_id"] != saved["predicted_id"]:
            raise ValueError("Recarga da GRU diverge")
        np.testing.assert_allclose([prediction["scores"][name] for name in CLASS_IDS],
            [saved["scores"][name] for name in CLASS_IDS],atol=1e-6,rtol=1e-5)
        replay.append({"relative_path":entry["relative_path"],"match":True,
            "maximum_score_difference":max(abs(prediction["scores"][name]-saved["scores"][name]) for name in CLASS_IDS)})
    reloaded.verify_frozen()
    write_json(output/"verification.json",{"checkpoint_reload_replays":replay,"frozen_states_unchanged":True,
        "historical_predictions_matched":len(entries),"test_evaluated":False,"training":False})
    elapsed = time.perf_counter()-started
    report = {"examples":entries,"selection_seed":seed,"food_model":predictor.identification,
        "detector":detector.identification if detector else None,"hardware":hardware(),"montage":montage,
        "processing_seconds_including_model_load_montage_verification":elapsed,
        "sum_real_video_seconds":sum(e["metadata"]["duration_seconds"] for e in entries),
        "real_source_frames":sum(e["metadata"]["frame_count"] for e in entries),
        "effective_processing_fps_end_to_end":sum(e["metadata"]["frame_count"] for e in entries)/elapsed,
        "test_evaluated":False,"training":False,"realtime_claim":False}
    write_json(output/"demo_report.json",report)
    write_page(output,entries,detector is not None)
    write_dynamic_report(output,report)
    return report
