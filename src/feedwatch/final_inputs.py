"""Inferência dos extratores históricos no dia reservado; sem ajuste de estatísticas."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import COLUMNS, file_hash, read_manifest, utc_now, write_json
from .evaluation import check_metadata
from .motion import FEATURE_NAMES, extract_video, extraction_identity, identity_hash, sample_indices
from .temporal_data import check_identity, validate_chronology
from .visual import (cache_signature, decode_visual, encode_frames, load_embedding_cache,
                     load_extractor, save_embedding_cache, state_hash, visual_identity)


def load_reserved_rows(config: dict, output: Path) -> pd.DataFrame:
    """Só pode ser chamado após verificar o selo prospectivo."""
    if not (output / "first_test_access.json").exists():
        write_json(output / "first_test_access.json", {"at_utc": utc_now(), "operation": "read manifest and test split",
                                                     "protocol_already_saved": True})
    manifest = read_manifest(Path(config["manifest"]))
    splits = Path(config["splits_dir"])
    assignment = json.loads((splits / "assignment.json").read_text(encoding="utf-8"))
    validation = json.loads((splits / "validation.json").read_text(encoding="utf-8"))
    if not validation["valid"] or validation["manifest_sha256"] != file_hash(Path(config["manifest"])):
        raise ValueError("Manifest difere da divisão original")
    if assignment != validation["assignment"] or assignment["test"] != [config["test_date"]]:
        raise ValueError("A data de teste foi alterada")
    rows = read_manifest(splits / "test.csv").sort_values("relative_path").reset_index(drop=True)
    expected = manifest[manifest.date.eq(config["test_date"])].sort_values("relative_path").reset_index(drop=True)
    check_metadata(rows)
    if not rows.equals(expected) or len(rows) != config["expected_videos"]:
        raise ValueError("Teste não corresponde aos 1.501 vídeos do dia reservado")
    for name in ("train", "val"):
        other = read_manifest(splits / f"{name}.csv")
        if set(other.relative_path) & set(rows.relative_path) or set(other.date) & set(rows.date):
            raise ValueError("Vazamento de vídeos/datas para o teste")
    rows.to_csv(output / "requested_test.csv", index=False)
    write_json(output / "test_provenance.json", {"date": config["test_date"], "videos": len(rows),
        "by_class": rows.class_name.value_counts().to_dict(),
        "input_sha256": {str(p): file_hash(p) for p in [Path(config["manifest"]),
              *[splits / name for name in ("train.csv", "val.csv", "test.csv", "assignment.json", "validation.json")]]}})
    return rows


def validate_motion_details(details: dict, parameters: dict) -> None:
    expected = sample_indices(details["fps"], details["frame_count"], parameters)
    times = np.asarray(details["sampled_times_seconds"], dtype=float)
    if details["sampled_indices"] != expected or details["requested_indices"] != expected:
        raise ValueError("Cache de movimento usa frames diferentes da grade original")
    if (times.shape != (len(expected),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
        or np.max(np.abs(times-np.asarray(expected)/details["fps"])) > max(1.5/details["fps"], .02)):
        raise ValueError("Cache de movimento usa timestamps incompatíveis")


def motion_item(row: dict, root: Path, identity: dict, caches: list[Path]) -> dict:
    started, warnings, signature = time.perf_counter(), [], None
    config_id, parameters = identity_hash(identity), identity["parameters"]
    try:
        signature = cache_signature(root, row["relative_path"], config_id)
        for index, cache in enumerate(caches):
            path = cache / config_id / f"{identity_hash(signature)}.json"
            if not path.exists():
                continue
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                result = entry["result"]
                if (entry["signature"] != signature or not result["ok"]
                    or set(result["features"]) != set(FEATURE_NAMES)
                    or not np.isfinite([result["features"][key] for key in FEATURE_NAMES]).all()):
                    raise ValueError("Assinatura/características inválidas")
                validate_motion_details(result["details"], parameters)
                return {**result, "cache_source": "historical" if index == 0 else "new_cache",
                        "cache_file": str(path), "signature": signature, "warning": "; ".join(warnings),
                        "seconds": time.perf_counter()-started}
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.append(f"{path}: {error}")
        result = extract_video(root, row["relative_path"], parameters)
        if signature != cache_signature(root, row["relative_path"], config_id):
            raise ValueError("Vídeo mudou durante a extração")
        path = caches[-1] / config_id / f"{identity_hash(signature)}.json"
        if result["ok"]:
            validate_motion_details(result["details"], parameters)
            write_json(path, {"signature": signature, "result": result})
        return {**result, "cache_source": "extracted", "cache_file": str(path) if result["ok"] else "",
                "signature": signature, "warning": "; ".join(warnings), "seconds": time.perf_counter()-started}
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        return {"ok": False, "features": None, "details": {}, "error": str(error), "signature": signature,
                "cache_source": "failure", "cache_file": "", "warning": "; ".join(warnings),
                "seconds": time.perf_counter()-started}


def _save_log(logs: list[dict], output: Path) -> None:
    log = pd.DataFrame(logs)
    output.mkdir(parents=True, exist_ok=True)
    log.to_csv(output / "extraction.csv", index=False)
    log[~log.ok].to_csv(output / "failures.csv", index=False)


def extract_motion_test(rows: pd.DataFrame, config: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    identity = config["extraction_identities"]["motion"]
    if extraction_identity(identity["parameters"]) != identity:
        raise ValueError("Extrator/versões de movimento diferem dos originais")
    started, logs, features, used = time.perf_counter(), [], [], []
    root = Path(config["data_dir"]).resolve()
    caches = [Path(config["read_motion_cache"]), Path(config["new_cache_dir"]) / "motion"]
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        records = rows.to_dict("records")
        items = executor.map(lambda row: motion_item(row, root, identity, caches), records)
        for i, (row, item) in enumerate(zip(records, items), 1):
            logs.append({**row, **{key: item[key] for key in ("ok", "error", "cache_source", "cache_file", "warning", "seconds")},
                         "details": json.dumps(item["details"]), "signature": json.dumps(item["signature"]),
                         "cache_sha256": file_hash(Path(item["cache_file"])) if item["ok"] else ""})
            if item["ok"]:
                used.append(row)
                features.append([item["features"][key] for key in FEATURE_NAMES])
            if i == 1 or i % 100 == 0 or i == len(rows):
                _save_log(logs, output)
                print(f"Teste movimento {i}/{len(rows)}; válidos {len(used)}; {time.perf_counter()-started:.1f}s", flush=True)
    frame, x = pd.DataFrame(used, columns=COLUMNS), pd.DataFrame(features, columns=FEATURE_NAMES, dtype=np.float64)
    if frame.empty:
        raise ValueError("Nenhuma característica de movimento válida; falhas registradas")
    frame.to_csv(output / "used_test.csv", index=False)
    np.save(output / "features.npy", x.to_numpy())
    report = {"requested": len(rows), "used": len(frame), "failed": len(rows)-len(frame),
              "wall_seconds": time.perf_counter()-started, "cache_sources": pd.DataFrame(logs).cache_source.value_counts().to_dict(),
              "config_id": identity_hash(identity), "features": FEATURE_NAMES}
    write_json(output / "extraction_report.json", report)
    return frame, x, report


def visual_item(row: dict, root: Path, identity: dict, caches: list[Path], transform) -> dict:
    started, warnings, signature = time.perf_counter(), [], None
    config_id, parameters = identity_hash(identity), identity["parameters"]
    try:
        signature = cache_signature(root, row["relative_path"], config_id)
        for index, cache in enumerate(caches):
            entry, warning = load_embedding_cache(cache, signature, parameters["frames_per_video"])
            if warning:
                warnings.append(warning)
            if entry is not None:
                try:
                    validate_chronology(entry["details"], parameters)
                    return {**entry, "ok": True, "error": "", "cache_source": "historical" if index == 0 else "new_cache",
                            "signature": signature, "warning": "; ".join(warnings), "seconds": time.perf_counter()-started,
                            "cache_file": str(cache / config_id / f"{identity_hash(signature)}.npz")}
                except (ValueError, KeyError, TypeError) as error:
                    warnings.append(str(error))
        decoded = decode_visual(root, row["relative_path"], parameters, transform)
        if signature != cache_signature(root, row["relative_path"], config_id):
            raise ValueError("Vídeo mudou durante a decodificação")
        if decoded["ok"]:
            validate_chronology(decoded["details"], parameters)
        return {**decoded, "cache_source": "extracted", "signature": signature, "cache_file": "",
                "warning": "; ".join(warnings)}
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        return {"ok": False, "error": str(error), "details": {}, "cache_source": "failure", "signature": signature,
                "warning": "; ".join(warnings), "seconds": time.perf_counter()-started, "cache_file": ""}


def extract_visual_test(rows: pd.DataFrame, config: dict, output: Path, device: torch.device):
    identity = config["extraction_identities"]["visual"]
    check_identity(identity)
    if identity["device_type"] != device.type:
        raise ValueError("Use o mesmo tipo de dispositivo do extrator histórico")
    started = time.perf_counter()
    model, transform, weights_info = load_extractor(Path(identity["weights"]["weights_file"]).parent, device)
    if visual_identity(identity["parameters"], weights_info, device) != identity:
        raise ValueError("Pré-processamento/pesos diferentes do extrator histórico")
    before = state_hash(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logs, sequences, used = [], [], []
    root, workers = Path(config["data_dir"]).resolve(), config["workers"]
    caches = [Path(config["read_visual_cache"]), Path(config["new_cache_dir"]) / "visual"]
    config_id, records = identity_hash(identity), rows.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(records), workers):
            group = records[start:start+workers]
            items = list(executor.map(lambda row: visual_item(row, root, identity, caches, transform), group))
            new = [item for item in items if item["ok"] and item["cache_source"] == "extracted"]
            # Mesmo batch máximo histórico de 4 vídeos × 16 frames; não ajustar BatchNorm.
            if new:
                encode_started = time.perf_counter()
                try:
                    encoded = encode_frames(model, torch.cat([item["frames"] for item in new]), device)
                    for item, embeddings in zip(new, np.split(encoded, len(new))):
                        item["embeddings"] = embeddings
                        item["seconds"] += (time.perf_counter()-encode_started)/len(new)
                        save_embedding_cache(caches[-1], item["signature"], embeddings, item["details"], item["seconds"])
                        item["cache_file"] = str(caches[-1] / config_id / f"{identity_hash(item['signature'])}.npz")
                except (RuntimeError, ValueError) as error:
                    for item in new:
                        item.update(ok=False, error=f"Falha no extrator: {error}")
                for item in new:
                    del item["frames"]
            for row, item in zip(group, items):
                logs.append({**row, **{key: item[key] for key in ("ok", "error", "cache_source", "cache_file", "warning", "seconds")},
                             "details": json.dumps(item["details"]), "signature": json.dumps(item["signature"]),
                             "cache_sha256": file_hash(Path(item["cache_file"])) if item["ok"] else ""})
                if item["ok"]:
                    used.append(row)
                    sequences.append(item["embeddings"])
            done = start+len(items)
            if start == 0 or done % 100 == 0 or done == len(records):
                _save_log(logs, output)
                print(f"Teste visual {done}/{len(rows)}; válidos {len(used)}; {time.perf_counter()-started:.1f}s", flush=True)
    if state_hash(model) != before or any(module.training for module in model.modules()):
        raise ValueError("Extrator mudou durante a inferência")
    frame = pd.DataFrame(used, columns=COLUMNS)
    if frame.empty:
        raise ValueError("Nenhuma sequência válida; falhas registradas")
    x = np.stack(sequences).astype(np.float32)
    frame.to_csv(output / "used_test.csv", index=False)
    np.save(output / "sequences.npy", x)
    memory = {"available": device.type == "cuda"}
    if device.type == "cuda":
        memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      scope="PyTorch allocator in this process, not entire board")
    report = {"requested": len(rows), "used": len(frame), "failed": len(rows)-len(frame),
              "shape": list(x.shape), "wall_seconds": time.perf_counter()-started,
              "cache_sources": pd.DataFrame(logs).cache_source.value_counts().to_dict(),
              "config_id": config_id, "extractor_unchanged": True, "gpu_memory": memory}
    write_json(output / "extraction_report.json", report)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame, x, report
