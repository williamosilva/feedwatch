"""Sequências reais do cache visual, sem reconstrução a partir de médias."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

from .data import COLUMNS, file_hash, read_manifest, write_json
from .evaluation import check_metadata
from .experiment import load_train_val, pilot_sample
from .motion import identity_hash
from .visual import (cache_signature, decode_visual, encode_frames, load_embedding_cache, load_extractor,
                     sample_visual_indices, save_embedding_cache, state_hash)


def validate_chronology(details: dict, config: dict) -> None:
    count = config["frames_per_video"]
    indices = details["sampled_indices"]
    times = np.asarray(details["sampled_times_seconds"], dtype=np.float64)
    expected = sample_visual_indices(details["fps"], details["frame_count"], count, config["window_seconds"])
    if indices != expected or details["requested_indices"] != expected:
        raise ValueError("Frames em ordem incorreta ou diferentes da grade visual original")
    if (times.shape != (count,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or times[0] < 0 or times[-1] >= config["window_seconds"]+1e-6):
        raise ValueError("Timestamps não são estritamente cronológicos na janela")
    if np.max(np.abs(times-np.asarray(indices)/details["fps"])) > max(1.5/details["fps"], .02):
        raise ValueError("Timestamps incompatíveis com os índices/FPS")


def check_identity(identity: dict) -> None:
    from . import visual
    if identity["source_sha256"] != file_hash(Path(visual.__file__)):
        raise ValueError("Código do extrator difere do experimento visual; não reutilizar identidade silenciosamente")
    versions = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "numpy": np.__version__, "opencv": cv2.__version__}
    if any(str(identity[key]) != value for key, value in versions.items()):
        raise ValueError("Versões diferem das usadas na extração visual original")
    if file_hash(Path(identity["weights"]["weights_file"])) != identity["weights"]["weights_sha256"]:
        raise ValueError("Pesos do extrator foram alterados")


def load_sequences(root: Path, manifest: Path, splits: Path, visual_dir: Path, cache: Path,
                   repair_cache: Path, output: Path, sample_per_class: int | None, seed: int,
                   device: torch.device) -> tuple[np.ndarray, pd.DataFrame, dict, dict]:
    started = time.perf_counter()
    original = json.loads((visual_dir / "config.json").read_text(encoding="utf-8"))
    identity, config_id = original["extraction_identity"], original["config_id"]
    if identity_hash(identity) != config_id:
        raise ValueError("Identidade de extração inválida")
    check_identity(identity)
    ecfg = identity["parameters"]
    if ecfg["frames_per_video"] != 16:
        raise ValueError("O experimento controlado requer os 16 frames originais")
    parts, provenance = load_train_val(manifest, splits)
    source_rows = pd.read_csv(visual_dir / "used_examples.csv")
    check_metadata(source_rows)
    if source_rows.relative_path.duplicated().any():
        raise ValueError("Caminhos duplicados na referência visual")
    reference_means = np.load(visual_dir / "mean_embeddings.npy", allow_pickle=False)
    if reference_means.shape != (len(source_rows), 576):
        raise ValueError("Médias históricas não correspondem à lista de linhas")
    positions = {path: i for i, path in enumerate(source_rows.relative_path)}
    selected = []
    for name in ("train", "val"):
        recorded = read_manifest(visual_dir / f"used_{name}.csv")
        expected = parts[name].sort_values("relative_path").reset_index(drop=True)
        if not recorded.sort_values("relative_path").reset_index(drop=True).equals(expected):
            raise ValueError("Exemplos visuais não correspondem aos dias inteiros existentes")
        indexed = source_rows[source_rows.split == name][COLUMNS].sort_values("relative_path").reset_index(drop=True)
        if not indexed.equals(expected):
            raise ValueError("Divisão/caminho/rótulo divergente na referência visual")
        if sample_per_class is not None:
            recorded = pilot_sample(recorded, sample_per_class, seed)
        selected.append(recorded.assign(split=name))
    rows = pd.concat(selected, ignore_index=True)
    sequences, logs, model, initial_state = [], [], None, None
    write_json(output / "extraction_identity.json", {"config_id": config_id, "identity": identity})
    rows.to_csv(output / "used_examples.csv", index=False)
    for name in ("train", "val"):
        rows[rows.split == name][COLUMNS].to_csv(output / f"used_{name}.csv", index=False)
    for i, row in enumerate(rows.to_dict("records"), start=1):
        signature = cache_signature(root, row["relative_path"], config_id)
        entry, warning = load_embedding_cache(cache, signature, 16)
        source = "original_cache"
        if entry is not None:
            try:
                validate_chronology(entry["details"], ecfg)
            except (ValueError, KeyError, TypeError) as error:
                entry, warning = None, str(error)
        if entry is None:
            entry, repair_warning = load_embedding_cache(repair_cache, signature, 16)
            source = "repair_cache"
            if entry is not None:
                validate_chronology(entry["details"], ecfg)
            if entry is None:
                if model is None:
                    if identity["device_type"] != device.type:
                        raise ValueError("Reextração exige o mesmo tipo de dispositivo da identidade original")
                    model, transform, _ = load_extractor(Path(identity["weights"]["weights_file"]).parent, device)
                    initial_state = state_hash(model)
                decoded = decode_visual(root, row["relative_path"], ecfg, transform)
                if not decoded["ok"]:
                    write_json(output / "input_failure.json", {"relative_path": row["relative_path"], "error": decoded["error"]})
                    raise ValueError(f"Falha na sequência {row['relative_path']}: {decoded['error']}; não será imputada ou excluída silenciosamente")
                validate_chronology(decoded["details"], ecfg)
                embeddings = encode_frames(model, decoded["frames"], device)
                if signature != cache_signature(root, row["relative_path"], config_id):
                    raise ValueError("Vídeo mudou durante a reextração")
                save_embedding_cache(repair_cache, signature, embeddings, decoded["details"], decoded["seconds"])
                entry = {"embeddings": embeddings, "details": decoded["details"]}
                source = "new_extraction"
        validate_chronology(entry["details"], ecfg)
        # A média histórica é somente verificação de associação, nunca uma fonte de sequência.
        delta = float(np.max(np.abs(entry["embeddings"].mean(0)-reference_means[positions[row["relative_path"]]])))
        if not np.allclose(entry["embeddings"].mean(0), reference_means[positions[row["relative_path"]]], rtol=1e-4, atol=2e-5):
            raise ValueError(f"Sequência e média histórica divergentes: {row['relative_path']}")
        sequences.append(entry["embeddings"])
        logs.append({**row, "cache_source": source, "cache_warning": warning, "mean_max_abs_difference": delta,
                     "indices": json.dumps(entry["details"]["sampled_indices"]),
                     "timestamps_seconds": json.dumps(entry["details"]["sampled_times_seconds"]),
                     "cache_key": identity_hash(signature), "config_id": config_id})
        if i % 500 == 0 or i == len(rows):
            print(f"Sequências verificadas {i}/{len(rows)}", flush=True)
    if model is not None and state_hash(model) != initial_state:
        raise ValueError("Extrator mudou durante a reextração")
    log = pd.DataFrame(logs)
    log.to_csv(output / "sequence_manifest.csv", index=False)
    array = np.stack(sequences).astype(np.float32)
    report = {"videos": len(rows), "shape": list(array.shape), "chronology_verified": True,
              "original_cache": int(log.cache_source.eq("original_cache").sum()),
              "repair_cache": int(log.cache_source.eq("repair_cache").sum()),
              "new_extractions": int(log.cache_source.eq("new_extraction").sum()),
              "mean_max_abs_difference": float(log.mean_max_abs_difference.max()),
              "wall_seconds": time.perf_counter()-started, "class_mapping_verified": True,
              "failures": 0, "test_evaluated": False, "provenance": provenance, "config_id": config_id}
    write_json(output / "input_audit.json", report)
    return array, rows, report, original
