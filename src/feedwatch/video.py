"""Auditoria amostral por OpenCV e exemplos com rótulos do dataset."""

from __future__ import annotations

import html
import json
import math
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import (CLASS_IDS, data_root, file_hash, protect_originals,
                   read_manifest, stratified_sample, utc_now, video_path, write_json)


def open_video(path: Path) -> cv2.VideoCapture:
    # FFmpeg incorporado ao wheel OpenCV; não exige ffmpeg.exe no PATH.
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
    ])


def inspect_video(root: Path, record: dict, frames_per_video: int) -> dict:
    result = {**record, "opened": False, "fps": None, "width": None,
              "height": None, "frame_count": None, "duration_seconds": None,
              "requested_indices": "[]", "decoded_indices": "[]",
              "frames_decoded": 0, "backend": "", "ok": False, "issues": ""}
    issues, cap = [], None
    try:
        cap = open_video(video_path(root, record["relative_path"]))
        result["opened"] = cap.isOpened()
        if not cap.isOpened():
            raise ValueError("Não foi possível abrir o vídeo")
        result["backend"] = cap.getBackendName()
        properties = {"fps": cv2.CAP_PROP_FPS, "width": cv2.CAP_PROP_FRAME_WIDTH,
                      "height": cv2.CAP_PROP_FRAME_HEIGHT, "frame_count": cv2.CAP_PROP_FRAME_COUNT}
        for name, prop in properties.items():
            value = cap.get(prop)
            if not math.isfinite(value) or value <= 0:
                issues.append(f"Metadado inválido: {name}={value}")
            else:
                result[name] = value if name == "fps" else int(round(value))
        if result["fps"] and result["frame_count"]:
            result["duration_seconds"] = result["frame_count"] / result["fps"]
        if result["frame_count"]:
            indices = np.unique(np.linspace(0, result["frame_count"] - 1,
                                           min(frames_per_video, result["frame_count"]), dtype=int)).tolist()
            result["requested_indices"] = json.dumps(indices)
            decoded = []
            for index in indices:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                    issues.append(f"Falha de posicionamento no frame {index}")
                    continue
                success, frame = cap.read()
                if not success or frame is None or frame.size == 0:
                    issues.append(f"Falha de decodificação no frame {index}")
                    continue
                actual = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if not math.isfinite(actual) or abs(actual - index) > 1:
                    issues.append(f"Posição retornada difere do frame solicitado {index}: {actual}")
                if (frame.shape[1], frame.shape[0]) != (result["width"], result["height"]):
                    issues.append(f"Resolução decodificada diverge no frame {index}")
                decoded.append(index)
            result["decoded_indices"] = json.dumps(decoded)
            result["frames_decoded"] = len(decoded)
        else:
            # Mesmo com metadados inválidos, tenta abrir um frame e registra a falha.
            success, frame = cap.read()
            result["frames_decoded"] = int(success and frame is not None and frame.size > 0)
            issues.append("Sem contagem válida para amostrar posições distribuídas")
    except (ValueError, OSError, cv2.error) as error:
        issues.append(str(error))
    finally:
        if cap is not None:
            cap.release()
    result["issues"] = " | ".join(issues)
    result["ok"] = not issues
    return result


def coverage(frame: pd.DataFrame) -> dict:
    return {"dates": sorted(frame.date.unique().tolist()),
            "groups": sorted(frame.group.unique().tolist()),
            "classes": sorted(frame.class_name.unique().tolist()),
            "strata_date_group_class": int(frame.groupby(["date", "group", "class_name"]).ngroups)}


def audit_dataset(root: Path, manifest_path: Path, output: Path,
                  sample_size: int, seed: int, full: bool, frames_per_video: int) -> dict:
    root = data_root(root)
    protect_originals(root, output)
    if frames_per_video < 2:
        raise ValueError("Use pelo menos 2 frames para verificar posições distribuídas")
    manifest = read_manifest(manifest_path)
    selected = manifest if full else stratified_sample(manifest, sample_size, seed)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "sample_manifest.csv", index=False)
    results = []
    for index, record in enumerate(selected.to_dict("records"), start=1):
        results.append(inspect_video(root, record, frames_per_video))
        if index % 20 == 0 or index == len(selected):
            print(f"Auditados {index}/{len(selected)} vídeos", flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "videos.csv", index=False)
    frame[~frame.ok].to_csv(output / "failures.csv", index=False)
    ranges = {}
    for column in ["fps", "width", "height", "frame_count", "duration_seconds"]:
        values = frame[column].dropna()
        ranges[column] = {"min": float(values.min()), "max": float(values.max())} if len(values) else None
    report = {"created_at_utc": utc_now(), "data_dir": str(root),
              "manifest_sha256": file_hash(manifest_path), "mode": "full" if full else "sample",
              "seed": seed, "sampling": "round-robin por data/grupo/classe; caminhos ordenados antes do shuffle",
              "frames_per_video": frames_per_video, "total_videos": len(manifest),
              "audited_videos": len(frame), "passed": int(frame.ok.sum()),
              "failed": int((~frame.ok).sum()), "decoded_frames": int(frame.frames_decoded.sum()),
              "manifest_coverage": coverage(manifest), "sample_coverage": coverage(selected),
              "successful_coverage": coverage(frame[frame.ok]), "metadata_ranges": ranges,
              "duration_method": "frame_count / fps informados pelo OpenCV; estimativa, especialmente para VFR",
              "limit": "Decodificação de posições amostradas, não verificação de todos os frames ou da exatidão dos rótulos."}
    write_json(output / "report.json", report)
    return report


def generate_examples(root: Path, manifest_path: Path, audit_dir: Path,
                      output: Path, seed: int) -> dict:
    root = data_root(root)
    protect_originals(root, output)
    manifest = read_manifest(manifest_path)
    audit_report = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))
    if audit_report["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Auditoria pertence a outro manifest; execute audit novamente")
    if Path(audit_report["data_dir"]).resolve() != root:
        raise ValueError("Auditoria pertence a outra raiz de dados")
    audited = pd.read_csv(audit_dir / "videos.csv")
    good = set(audited.loc[audited.ok.eq(True), "relative_path"])
    candidates = manifest[manifest.relative_path.isin(good)].sort_values("relative_path")
    if candidates.empty:
        raise ValueError("Nenhum vídeo aprovado na auditoria para gerar exemplos")
    rng = random.Random(seed)
    selected = []
    for _, group in candidates.groupby(["date", "class_name"], sort=True):
        selected.append(group.iloc[rng.randrange(len(group))].to_dict())
    output.mkdir(parents=True, exist_ok=True)
    cards, entries, issues = [], [], []
    translations = {"none": "ausente", "weak": "fraca", "medium": "media", "strong": "forte"}
    for number, record in enumerate(selected, start=1):
        cap = None
        try:
            cap = open_video(video_path(root, record["relative_path"]))
            if not cap.isOpened():
                raise ValueError("Falha de abertura")
            index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError("Falha ao posicionar frame central")
            success, image = cap.read()
            if not success or image is None:
                raise ValueError("Falha ao decodificar frame central")
            card = np.full((350, 480, 3), 245, dtype=np.uint8)
            ratio = min(480 / image.shape[1], 248 / image.shape[0])
            thumb = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))))
            top, left = (248 - thumb.shape[0]) // 2, (480 - thumb.shape[1]) // 2
            card[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb
            lines = [f"ROTULO DO DATASET: {record['class_name']} ({translations[record['class_name']]})",
                     "Exemplo rotulado; nao e previsao"]
            lines += textwrap.wrap(record["relative_path"], width=60, break_long_words=True)
            lines += [f"Frame {index}"]
            for line_number, line in enumerate(lines):
                cv2.putText(card, line, (8, 267 + 17 * line_number), cv2.FONT_HERSHEY_SIMPLEX,
                            0.37, (25, 25, 25), 1, cv2.LINE_AA)
            filename = f"example_{number:02d}.jpg"
            if not cv2.imwrite(str(output / filename), card):
                raise OSError("Não foi possível salvar o exemplo")
            cards.append(card)
            entries.append({**record, "frame_index": index, "image": filename, "label_source": "dataset"})
        except (ValueError, OSError, cv2.error) as error:
            issues.append({"relative_path": record["relative_path"], "reason": str(error)})
        finally:
            if cap is not None:
                cap.release()
    if cards:
        columns = min(4, len(cards))
        sheet = np.full((math.ceil(len(cards) / columns) * 350, columns * 480, 3), 230, dtype=np.uint8)
        for index, card in enumerate(cards):
            row, col = divmod(index, columns)
            sheet[row * 350:(row + 1) * 350, col * 480:(col + 1) * 480] = card
        if not cv2.imwrite(str(output / "contact_sheet.jpg"), sheet):
            raise OSError("Não foi possível salvar a seleção visual")
    write_json(output / "examples.json", {"seed": seed, "manifest_sha256": file_hash(manifest_path),
                                          "label_source": "dataset", "examples": entries, "issues": issues})
    figures = "\n".join(f'<figure><img src="{e["image"]}" alt="{html.escape(e["relative_path"])}">'
                         f'<figcaption>Rótulo do dataset: <b>{e["class_name"]}</b><br>'
                         f'{html.escape(e["relative_path"])}</figcaption></figure>' for e in entries)
    (output / "index.html").write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
        '<title>FeedWatch — rótulos do dataset</title><style>body{font:16px system-ui;margin:32px;background:#fafafa}'
        'main{display:flex;flex-wrap:wrap;gap:20px}figure{margin:0;max-width:480px}img{width:100%}'
        'figcaption{overflow-wrap:anywhere;padding:8px}</style><h1>Exemplos rotulados do AV-FFIA</h1>'
        '<p>Rótulos fornecidos pelas pastas do dataset. Nenhum modelo foi treinado; estas imagens não são previsões.</p>'
        f'<main>{figures}</main></html>', encoding="utf-8")
    return {"examples": len(entries), "issues": issues, "output": str(output)}
