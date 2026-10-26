"""Experimento controlado com três seeds e ablação fixa da ordem dos frames."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import data_root, file_hash, protect_originals, utc_now, write_json
from .temporal_data import load_sequences
from .temporal_examples import create_input_examples
from .temporal_heads import TemporalHead, train_head, validation_permutations
from .temporal_reporting import summarize_temporal


def run_temporal(root: Path, manifest: Path, splits: Path, visual_dir: Path, motion_dir: Path,
                 config_path: Path, cache: Path, repair_cache: Path, output: Path,
                 sample_per_class: int | None = None, device_name: str = "auto") -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = data_root(root)
    protect_originals(root, output)
    protect_originals(root, repair_cache)
    for previous in (visual_dir, motion_dir):
        if output.resolve() == previous.resolve() or previous.resolve() in output.resolve().parents:
            raise ValueError("A nova pasta deve ficar fora dos experimentos anteriores")
    if (output / "results.json").exists() or (output / "summary.json").exists():
        raise ValueError("Experimento já concluído; use uma nova pasta para preservá-lo")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["seeds"] != [42, 43, 44]:
        raise ValueError("Esta comparação fixa usa as seeds 42, 43 e 44")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(output / "run_status.json", {"status": "running", "created_at_utc": utc_now()})
    source_files = [manifest, splits / "train.csv", splits / "val.csv", splits / "assignment.json", splits / "validation.json",
                    visual_dir / "results.json", visual_dir / "config.json", visual_dir / "used_examples.csv",
                    visual_dir / "used_train.csv", visual_dir / "used_val.csv", visual_dir / "mean_embeddings.npy"]
    source_hashes = {str(path): file_hash(path) for path in source_files}
    sequences, rows, audit, visual_config = load_sequences(root, manifest, splits, visual_dir, cache, repair_cache,
                                                          output, sample_per_class, config["examples_seed"], device)
    indices = {name: np.flatnonzero(rows.split.eq(name)) for name in ("train", "val")}
    if sample_per_class is None and (len(indices["train"]), len(indices["val"])) != (3898, 1705):
        raise ValueError("O controle completo deve usar exatamente 3.898 exemplos de treino e 1.705 de validação")
    if set(rows.iloc[indices["train"]].relative_path) & set(rows.iloc[indices["val"]].relative_path):
        raise ValueError("Vazamento de caminhos")
    counts = {name: {"videos": len(index), "by_class": rows.iloc[index].class_name.value_counts().to_dict()}
              for name, index in indices.items()}
    example_report = create_input_examples(root, sequences, rows, visual_config, device,
                                            output / "input_examples", config["examples_seed"])
    # Uma única normalização para as duas arquiteturas: estimada nas médias de vídeo do treino.
    train_means = sequences[indices["train"]].mean(1)
    mean, std = train_means.mean(0), np.maximum(train_means.std(0), 1e-6)
    normalization_matches_linear = None
    if sample_per_class is None:
        historical = torch.load(visual_dir / "classifier" / "best_classifier.pt", map_location="cpu", weights_only=True)
        np.testing.assert_allclose(mean, historical["mean"].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(std, historical["std"].numpy(), rtol=1e-5, atol=1e-6)
        normalization_matches_linear = True
    np.savez(output / "normalization_train_only.npz", mean=mean, std=std)
    tensors = {name: torch.from_numpy((sequences[index]-mean)/std).to(device) for name, index in indices.items()}
    targets = {name: rows.iloc[index].class_id.to_numpy() for name, index in indices.items()}
    frames = {name: rows.iloc[index].reset_index(drop=True) for name, index in indices.items()}
    permutations = validation_permutations(frames["val"].relative_path.tolist(), config["ablation"]["permutation_seed"])
    permutation_frame = frames["val"].copy()
    permutation_frame["permutation"] = [json.dumps(row.tolist()) for row in permutations]
    permutation_frame.to_csv(output / "validation_permutations.csv", index=False)
    parameter_counts = {kind: sum(p.numel() for p in TemporalHead(kind, config["heads"]).parameters()) for kind in ("mlp", "gru")}
    parameter_difference = parameter_counts["gru"]-parameter_counts["mlp"]
    if abs(parameter_difference)/parameter_counts["gru"] > .01:
        raise ValueError("O controle de capacidade deve ficar a menos de 1% dos parâmetros da GRU")
    provenance = {"source_sha256": source_hashes, "extraction_config_id": audit["config_id"],
                  "used_train_sha256": file_hash(output / "used_train.csv"),
                  "permutation_sha256": file_hash(output / "validation_permutations.csv"),
                  "permutation_seed": config["ablation"]["permutation_seed"], "test_evaluated": False}
    write_json(output / "config.json", {"experiment": config, "device": str(device), "mode": "pilot" if sample_per_class else "full",
                                         "sample_per_class": sample_per_class, "parameter_counts": parameter_counts,
                                         "parameter_difference_gru_minus_mlp": parameter_difference,
                                         "parameter_difference_percent_of_gru": 100*parameter_difference/parameter_counts["gru"],
                                         "normalization_matches_historical_linear": normalization_matches_linear,
                                         "provenance": provenance})
    seeds = [42] if sample_per_class is not None else config["seeds"]
    runs = []
    for seed in seeds:
        for kind in ("mlp", "gru"):
            result = train_head(kind, seed, config["heads"], config["training"], tensors["train"], targets["train"],
                                tensors["val"], targets["val"], frames["train"], frames["val"], mean, std,
                                output / kind / f"seed_{seed}", provenance, permutations)
            runs.append(result)
            print(f"Concluído {kind}, seed {seed}: val_f1={result['validation']['macro_f1']:.6f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if not all(file_hash(Path(path)) == digest for path, digest in source_hashes.items()):
        raise ValueError("Artefatos de origem mudaram durante o experimento")
    summary = summarize_temporal(runs, output, visual_dir, motion_dir, audit)
    verification = {"source_artifacts_unchanged": True, "all_checkpoints_replay_verified": all(run["checkpoint_reproduces_predictions"] for run in runs),
                    "example_embeddings_match_cache": True, "example_extractor_unchanged": example_report["extractor_state_unchanged"],
                    "shared_normalization_train_only": True, "normalization_matches_historical_linear": normalization_matches_linear,
                    "same_permutations_across_seeds": True, "permutation_sha256": provenance["permutation_sha256"],
                    "same_frames_for_both_heads": True, "test_evaluated": False}
    write_json(output / "verification.json", verification)
    result = {"mode": "pilot" if sample_per_class else "full", "created_at_utc": utc_now(), "config": config,
              "counts": counts, "runs": runs, "summary": summary, "verification": verification,
              "wall_seconds": time.perf_counter()-started, "test_evaluated": False}
    write_json(output / "results.json", result)
    write_json(output / "run_status.json", {"status": "complete"})
    print(f"Relatório: {output / 'report.md'}; teste não utilizado.", flush=True)
    return result
