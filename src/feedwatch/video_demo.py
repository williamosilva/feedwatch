"""Galeria local de clipes independentes: seleção cega às previsões, somente validação."""

import html
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_IDS, COLUMNS, data_root, file_hash, read_manifest, utc_now, write_json
from .evaluation import check_metadata
from .video_export import export_video, prepare_output
from .video_inference import OperationalPredictor, PORTUGUESE


def select_validation(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    check_metadata(rows)
    if set(rows.date) != {"2022_6_21"}:
        raise ValueError("A demonstração deve usar somente a validação de 21/06; não abrir o teste")
    rng, chosen = random.Random(seed), []
    for name in CLASS_IDS:
        candidates = rows[rows.class_name.eq(name)].sort_values("relative_path").to_dict("records")
        if len(candidates) < 3:
            raise ValueError(f"Menos de três clipes de validação para {name}")
        chosen.extend(rng.sample(candidates, 3))
    return pd.DataFrame(chosen, columns=COLUMNS)


def write_gallery(output: Path, entries: list[dict], seed: int):
    sections = []
    for name in CLASS_IDS:
        cards = []
        for item in [entry for entry in entries if entry["class_name"] == name]:
            folder = html.escape(item["folder"], quote=True)
            prediction = item.get("predicted_class")
            prediction_text = f"{prediction} · {PORTUGUESE[prediction]}" if prediction else "Análise indisponível"
            media = (f'<video controls playsinline preload="metadata" poster="{folder}/poster.jpg" aria-label="Clipe {html.escape(item["relative_path"],quote=True)}"><source src="{folder}/annotated.mp4" type="video/mp4">Seu navegador não reproduz este vídeo. Use o link abaixo.</video>'
                     if item["exported"] else f'<img src="{folder}/analysis_unavailable.png" alt="Análise indisponível">')
            video_link = f'<a href="{folder}/annotated.mp4">Abrir vídeo anotado</a>' if item["exported"] else "Sem vídeo exportado"
            cards.append(f'''<article class="clip">{media}<div class="clip-copy"><div class="labels"><div><span>Rótulo do dataset</span><strong>{name} · {PORTUGUESE[name]}</strong></div><div><span>Previsão do modelo</span><strong>{prediction_text}</strong></div></div><p class="mode">Análise do clipe completo · offline</p><p class="path">{html.escape(item['relative_path'])}</p><div class="links">{video_link}<a href="{folder}/analysis.json">Registro da análise</a></div></div></article>''')
        sections.append(f'<section id="{name}" aria-labelledby="title-{name}"><div class="section-title"><h2 id="title-{name}">{PORTUGUESE[name]}</h2><span>Rótulo do dataset: {name} · 3 clipes independentes</span></div><div class="clips">'+"".join(cards)+"</div></section>")
    document = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FeedWatch · Caderno de vídeo</title><style>
:root{color-scheme:light;--lake:#edf4f5;--ink:#163948;--teal:#246971;--slate:#526b75;--white:#fff;--amber:#865d26}*{box-sizing:border-box}body{margin:0;background:var(--lake);color:var(--ink);font:16px/1.55 "Segoe UI",Arial,sans-serif}a{color:var(--teal);text-underline-offset:4px}a:focus-visible,video:focus-visible{outline:3px solid var(--amber);outline-offset:4px}header,main,footer{max-width:1440px;margin:auto;padding:32px 40px}.brand{font-weight:700;font-size:23px;letter-spacing:-.5px}.masthead{display:flex;justify-content:space-between;gap:24px;align-items:center;border-bottom:1px solid #b9cfd3;padding-bottom:22px}.kicker{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--slate)}h1{font:500 clamp(32px,4.5vw,58px)/1.1 Cambria,Georgia,serif;max-width:850px;margin:28px 0 18px}.intro{max-width:850px;margin:0;color:var(--slate)}.notice{background:#fff;border-left:4px solid var(--teal);padding:17px 22px;max-width:1000px;margin:26px 0 0}.notice p{margin:4px 0}.notice strong{font-weight:600}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:26px}nav a{background:#dcebed;border:1px solid #b9cfd3;padding:7px 17px;border-radius:24px;text-decoration:none}main{padding-top:0}.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin:30px 0 14px}.section-title h2{font:500 30px Cambria,Georgia,serif;margin:0}.section-title span{color:var(--slate);font-size:13px}.clips{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.clip{min-width:0;border:1px solid #c7dadd;background:var(--white);border-radius:12px;overflow:hidden}.clip video,.clip img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#142d39}.clip-copy{padding:20px 22px}.labels{display:grid;grid-template-columns:1fr 1fr;gap:16px}.labels span{display:block;font-size:12px;color:var(--slate);margin-bottom:5px}.labels strong{font-weight:600;font-size:18px}.mode{font-size:13px;color:var(--teal);margin:16px 0 8px}.path{font:12px/1.5 Consolas,monospace;overflow-wrap:anywhere;color:var(--slate);margin:8px 0 18px}.links{display:flex;flex-wrap:wrap;gap:14px;font-size:13px}footer{color:var(--slate);font-size:14px;padding-top:16px;padding-bottom:48px}footer p{max-width:1000px}.crop-mark{display:inline-block;width:15px;height:15px;border-top:2px solid var(--teal);border-left:2px solid var(--teal);margin-right:7px;vertical-align:middle}@media(max-width:700px){header,main,footer{padding-left:18px;padding-right:18px}.masthead,.section-title{align-items:flex-start;flex-direction:column;gap:7px}.clips{grid-template-columns:1fr}.labels strong{font-size:16px}.clip-copy{padding:17px}.notice{padding:13px 16px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><header><div class="masthead"><div class="brand">FeedWatch</div><div class="kicker">Caderno de vídeo · protótipo local</div></div><h1>Observar a intensidade.<br>Conferir o que o modelo vê.</h1><p class="intro">Doze clipes independentes de validação, selecionados por classe com seed SEED, sem filtrar por acerto. Cada vídeo mantém seu próprio início e fim. Os rótulos aparecem separados das previsões e não entram na inferência.</p><div class="notice"><p><strong>Análise do clipe completo.</strong> Esta reprodução offline mostra o resultado obtido após analisar o clipe. Não é previsão ao vivo.</p><p>Intensidade observada não comprova fome, saciedade nem segurança para controlar um alimentador. As probabilidades não foram calibradas.</p></div><nav aria-label="Classes do dataset">NAV</nav></header><main>SECTIONS</main><footer><p><span class="crop-mark" aria-hidden="true"></span>Os cantos discretos indicam o recorte da imagem recebido pelo modelo; não localizam peixes. A CNN e a GRU seed 42 permanecem congeladas.</p><p>As janelas consecutivas de dois segundos são uma extensão experimental sem avaliação de desempenho entre janelas. Não há suavização. A classe medium apresentou baixo recall na avaliação experimental. Outros dias, câmeras ou fazendas podem produzir resultados diferentes.</p><p><a href="selection.csv">Seleção reproduzível</a> · <a href="demo_report.json">Execução e compatibilidade</a>. Somente validação de 21/06; o conjunto de teste não foi reavaliado.</p></footer></body></html>'''
    document = document.replace("<title>", '<link rel="icon" href="data:,"><title>', 1)
    technical = output.parent / "technical-checks" / "sequential" / "annotated.mp4"
    if technical.exists():
        document = document.replace("</footer>", '<p>Verificação temporal separada: <a href="../technical-checks/sequential/annotated.mp4">padrão sintético contínuo de 5,2 s com áudio</a>. Sem peixes; verifica atraso de apresentação e trecho final incompleto, não desempenho alimentar.</p></footer>')
    document = document.replace("SEED", str(seed)).replace("NAV", "".join(f'<a href="#{name}">{PORTUGUESE[name]}</a>' for name in CLASS_IDS)).replace("SECTIONS", "".join(sections))
    (output / "index.html").write_text(document, encoding="utf-8")


def generate_demo(root: Path, splits: Path, output: Path, seed=42, device_name="auto", protocol=Path("artifacts/experiments/final-test")) -> dict:
    started = time.perf_counter()
    root = data_root(root)
    predictor = OperationalPredictor(protocol, device_name)
    prepare_output(output, predictor)
    rows = select_validation(read_manifest(splits / "val.csv"), seed)
    # A seleção é persistida antes de consultar as previsões históricas ou inferir.
    rows.to_csv(output / "selection.csv", index=False)
    write_json(output / "selection_protocol.json", {"at_utc": utc_now(), "seed": seed, "per_class": 3,
        "selection": "random.Random(seed).sample(sorted class paths, 3), class order none/weak/medium/strong",
        "prediction_filter": False, "split": "validation", "val_sha256": file_hash(splits / "val.csv")})
    historical_path = Path(predictor.identification["checkpoint"]).parent / "val" / "predictions.csv"
    historical = pd.read_csv(historical_path).set_index("relative_path")
    entries = []
    for number, row in enumerate(rows.to_dict("records"), 1):
        folder = f"{number:02d}_{row['class_name']}"
        # Inferência recebe somente um caminho, nunca class_name/class_id como argumento.
        result = export_video(root / row["relative_path"], output / folder, predictor, "clip")
        item = {**row, "folder": folder, "exported": bool(result.get("output_video")),
                "processing_seconds": result["processing_seconds"], "duration_seconds": result.get("metadata", {}).get("duration_seconds"),
                "predicted_class": None, "prediction_matches_existing_pipeline": False}
        if result.get("windows") and result["windows"][0]["status"] == "ok":
            prediction = result["windows"][0]
            previous = historical.loc[row["relative_path"]]
            if previous.class_name != row["class_name"] or int(previous.class_id) != row["class_id"]:
                raise ValueError("IDs/rótulos históricos não correspondem ao caminho selecionado")
            if prediction["predicted_id"] != int(previous.predicted_id):
                raise ValueError("Previsão diverge do pipeline existente; não alterar o modelo para corrigi-la")
            values = [prediction["scores"][name] for name in CLASS_IDS]
            expected = [previous[f"prob_{name}"] for name in CLASS_IDS]
            np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)
            item.update(predicted_class=prediction["predicted_class"], prediction_matches_existing_pipeline=True,
                        probability_max_abs_difference=float(np.max(np.abs(np.asarray(values)-expected))))
        entries.append(item)
        print(f"Demonstração {number}/12: {row['class_name']} -> {item['predicted_class']}; {result['processing_seconds']:.2f}s", flush=True)
    predictor.verify_frozen()
    report = {"created_at_utc": utc_now(), "selection_seed": seed, "examples": entries,
              "videos": len(entries), "exported": sum(entry["exported"] for entry in entries),
              "compatible_with_existing_pipeline": sum(entry["prediction_matches_existing_pipeline"] for entry in entries),
              "processing_seconds_including_model_load": time.perf_counter()-started,
              "sum_video_duration_seconds": sum(entry["duration_seconds"] or 0 for entry in entries),
              "sum_export_processing_seconds": sum(entry["processing_seconds"] for entry in entries),
              "checkpoint": predictor.identification, "selection_uses_predictions": False,
              "test_evaluated": False, "montage": False, "realtime_claim": False}
    write_json(output / "demo_report.json", report)
    write_gallery(output, entries, seed)
    return report
