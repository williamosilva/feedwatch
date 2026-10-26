"""Avaliação final de checkpoints fixos: não contém ajuste ou seleção de modelos."""

import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch

from .data import CLASS_IDS, COLUMNS, data_root, file_hash, protect_originals, utc_now, write_json
from .evaluation import NAMES, check_metadata, metrics, save_evaluation
from .final_inputs import extract_motion_test, extract_visual_test, load_reserved_rows
from .final_protocol import verify_protocol
from .linear_probe import load_probe
from .motion import FEATURE_NAMES, identity_hash
from .temporal_heads import load_temporal_checkpoint


def inference_setup(device_name: str) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA histórica não está disponível; não trocar o extrator silenciosamente")
    return device


@torch.inference_mode()
def infer_checkpoint(spec: dict, config: dict, motion_x: pd.DataFrame, sequences: np.ndarray,
                     device: torch.device) -> tuple[np.ndarray, dict]:
    """Carrega checkpoint e estatísticas salvas. Nunca recebe rótulos do teste."""
    started = time.perf_counter()
    path, family = Path(spec["checkpoint"]), spec["family"]
    if file_hash(path) != config["locked_input_sha256"][str(path)]:
        raise ValueError("Checkpoint difere do protocolo")
    info = {**spec, "checkpoint_sha256": file_hash(path), "trained_in_this_stage": False}
    if family in ("dummy", "random_forest"):
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if (bundle["class_mapping"] != CLASS_IDS or bundle["feature_names"] != FEATURE_NAMES
            or bundle["extraction_identity"] != config["extraction_identities"]["motion"]
            or bundle["config_id"] != identity_hash(config["extraction_identities"]["motion"])
            or bundle["sklearn_version"] != sklearn.__version__
            or bundle["estimator"].classes_.tolist() != list(range(4))):
            raise ValueError("Modelo de movimento incompatível")
        estimator = bundle["estimator"]
        if family == "dummy" and estimator.strategy != "most_frequent":
            raise ValueError("Dummy deve manter a classe majoritária do treino")
        if list(motion_x.columns) != FEATURE_NAMES or not np.isfinite(motion_x.to_numpy()).all():
            raise ValueError("Somente as características visuais originais podem entrar na floresta")
        probabilities = estimator.predict_proba(motion_x)
        np.testing.assert_array_equal(estimator.predict(motion_x), probabilities.argmax(1))
        info.update(extraction_config_id=bundle["config_id"], class_mapping=bundle["class_mapping"],
                    used_train_sha256=bundle["used_train_sha256"])
    else:
        model, bundle = (load_probe(path, device) if family == "linear" else load_temporal_checkpoint(path, device))
        if bundle["seed"] != spec["seed"] or bundle["extractor_trained"]:
            raise ValueError("Seed/extrator diferentes do checkpoint fixado")
        config_id = identity_hash(config["extraction_identities"]["visual"])
        provenance = bundle["provenance"]
        if provenance.get("config_id", provenance.get("extraction_config_id")) != config_id:
            raise ValueError("Cabeça foi treinada com outro extrator")
        if family in ("mlp", "gru") and bundle["kind"] != family:
            raise ValueError("Família do checkpoint divergente")
        mean, std = bundle["mean"].numpy(), bundle["std"].numpy()
        if (mean.shape != (576,) or std.shape != (576,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Estatísticas de treino salvas são inválidas")
        if sequences.ndim != 3 or sequences.shape[1:] != (16, 576) or not np.isfinite(sequences).all():
            raise ValueError("Sequências inválidas; não imputar com zeros")
        # Preserva a ordem das operações de cada experimento histórico.
        source = sequences.mean(1) if family == "linear" else sequences
        x = torch.from_numpy((source-mean)/std).to(device)
        model.requires_grad_(False).eval()
        logits = torch.cat([model(batch).cpu() for batch in x.split(config["evaluation_batch_size"])])
        probabilities = logits.softmax(1).numpy()
        info.update(selected_epoch=bundle["selected_epoch"], selection=bundle["selection"],
                    validation_macro_f1_at_selection=bundle["validation_macro_f1"],
                    head_config=bundle.get("head_config", {"input_dim": 576, "output_dim": 4}),
                    training_config=bundle["training_config"], extraction_config_id=config_id,
                    parameters=sum(p.numel() for p in model.parameters()), class_mapping=bundle["class_mapping"],
                    used_train_sha256=provenance["used_train_sha256"],
                    normalization_source="saved train-only mean/std from original checkpoint",
                    normalization_sha256=__import__("hashlib").sha256(mean.tobytes()+std.tobytes()).hexdigest(),
                    extractor_weights_sha256=config["extraction_identities"]["visual"]["weights"]["weights_sha256"])
    if info["used_train_sha256"] != file_hash(Path(config["splits_dir"]) / "train.csv"):
        raise ValueError("Modelo não corresponde à lista original de treino")
    info["inference_seconds_including_load"] = time.perf_counter()-started
    return probabilities, info


def verify_predictions(stored: pd.DataFrame, rows: pd.DataFrame, probabilities: np.ndarray) -> dict:
    check_metadata(stored)
    if not stored[COLUMNS].equals(rows[COLUMNS].reset_index(drop=True)):
        raise ValueError("Ordem/associação dos vídeos nas previsões diverge das entradas")
    if any(CLASS_IDS.get(name) != ident for name, ident in zip(stored.predicted_class, stored.predicted_id)):
        raise ValueError("Classe e ID previstos divergem")
    expected = stored[[f"prob_{name}" for name in NAMES]].to_numpy()
    np.testing.assert_array_equal(probabilities.argmax(1), stored.predicted_id.to_numpy())
    np.testing.assert_allclose(probabilities, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(stored.correct.to_numpy(), stored.class_id.to_numpy() == stored.predicted_id.to_numpy())
    return {"ids_exact": True, "probabilities_match": True,
            "max_abs_probability_difference": float(np.max(np.abs(probabilities-expected)))}


def snapshot_previous(config: dict, output: Path) -> dict:
    excluded = [output.resolve(), Path(config["new_cache_dir"]).resolve()]
    paths = [path for path in Path("artifacts").rglob("*") if path.is_file()
             and not any(path.resolve() == target or target in path.resolve().parents for target in excluded)]
    paths.extend(path for path in Path("config").glob("*.json"))
    snapshot = {str(path): file_hash(path) for path in sorted(set(paths))}
    write_json(output / "prior_artifacts_sha256.json", snapshot)
    return snapshot


def check_previous(snapshot: dict) -> None:
    changed = [path for path, digest in snapshot.items() if not Path(path).exists() or file_hash(Path(path)) != digest]
    if changed:
        raise ValueError(f"Artefatos anteriores alterados: {changed[:5]}")


def replay_final(output: Path) -> dict:
    """Recarga independente dos nove checkpoints e das entradas persistidas, sem decodificar."""
    config = verify_protocol(output)
    if (output / "deliverables_sha256.json").exists():
        saved = json.loads((output / "deliverables_sha256.json").read_text(encoding="utf-8"))
        if any(file_hash(Path(path)) != digest for path, digest in saved.items()):
            raise ValueError("Artefatos finais não correspondem aos hashes publicados")
    device = inference_setup(config["device"])
    source_hashes = json.loads((output / "inputs_sha256.json").read_text(encoding="utf-8"))
    if any(file_hash(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError("Entradas/features/proveniência foram alteradas")
    motion_rows = read_rows(output / "inputs" / "motion" / "used_test.csv")
    visual_rows = read_rows(output / "inputs" / "visual" / "used_test.csv")
    motion_x = pd.DataFrame(np.load(output / "inputs" / "motion" / "features.npy", allow_pickle=False), columns=FEATURE_NAMES)
    sequences = np.load(output / "inputs" / "visual" / "sequences.npy", allow_pickle=False)
    replay = []
    for spec in config["models"]:
        rows = motion_rows if spec["family"] in ("dummy", "random_forest") else visual_rows
        probabilities, _ = infer_checkpoint(spec, config, motion_x, sequences, device)
        directory = output / "models" / spec["id"]
        stored = pd.read_csv(directory / "predictions.csv")
        result = verify_predictions(stored, rows, probabilities)
        recomputed = metrics(stored.class_id.to_numpy(), stored.predicted_id.to_numpy())
        original = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for key in ("examples", "accuracy", "macro_f1", "classification_report", "confusion_matrix"):
            if recomputed[key] != original[key]:
                raise ValueError(f"Métricas não reproduzidas: {spec['id']}/{key}")
        replay.append({"model": spec["id"], **result, "metrics_recomputed": True})
    check_previous(json.loads((output / "prior_artifacts_sha256.json").read_text(encoding="utf-8")))
    seal = json.loads((output / "protocol_seal.json").read_text(encoding="utf-8"))
    access = json.loads((output / "first_test_access.json").read_text(encoding="utf-8"))
    if not seal["sealed_at_utc"] < access["at_utc"]:
        raise ValueError("O acesso ao teste precede o protocolo")
    result = {"verified_at_utc": utc_now(), "models": replay, "protocol_precedes_test_access": True,
              "prior_artifacts_unchanged": True, "no_training": True, "operational_seed_unchanged": 42,
              "test_evaluated": True, "inputs_sha256_verified": True}
    if not (output / "results.json").exists():
        write_json(output / "verification.json", result)
    return result


def read_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    check_metadata(rows)
    return rows


def run_final(output: Path) -> dict:
    config = verify_protocol(output)
    if (output / "results.json").exists():
        raise ValueError("Avaliação concluída; use final-replay para verificar sem substituir os resultados")
    root = data_root(Path(config["data_dir"]))
    protect_originals(root, output)
    protect_originals(root, Path(config["new_cache_dir"]))
    if output.resolve().is_relative_to(Path(config["new_cache_dir"]).resolve()):
        raise ValueError("Saída deve ficar fora do cache")
    started = time.perf_counter()
    device = inference_setup(config["device"])
    # O protocolo está verificado antes de ler qualquer dado do dia reservado.
    write_json(output / "run_status.json", {"status": "running", "at_utc": utc_now(), "training": False})
    write_json(output / "execution_source_sha256.json", {str(path): file_hash(path)
               for path in Path(__file__).parent.glob("*.py")})
    snapshot_previous(config, output)
    rows = load_reserved_rows(config, output)
    # Registra o conteúdo dos vídeos com metadados; não duplica nem modifica originais.
    inventory = [{"relative_path": row.relative_path, "size_bytes": (root / row.relative_path).stat().st_size,
                  "mtime_ns": (root / row.relative_path).stat().st_mtime_ns} for row in rows.itertuples()]
    write_json(output / "test_video_inventory.json", inventory)
    motion_rows, motion_x, motion_report = extract_motion_test(rows, config, output / "inputs" / "motion")
    visual_rows, sequences, visual_report = extract_visual_test(rows, config, output / "inputs" / "visual", device)
    input_paths = [output / "requested_test.csv", output / "test_provenance.json", output / "test_video_inventory.json",
                   *[path for path in (output / "inputs").rglob("*") if path.is_file()]]
    write_json(output / "inputs_sha256.json", {str(path): file_hash(path) for path in input_paths})
    results = []
    for spec in config["models"]:
        frame = motion_rows if spec["family"] in ("dummy", "random_forest") else visual_rows
        probabilities, info = infer_checkpoint(spec, config, motion_x, sequences, device)
        directory = output / "models" / spec["id"]
        values = save_evaluation(frame, probabilities, directory, "test")
        info.update(requested=len(rows), used=len(frame), failed=len(rows)-len(frame),
                    coverage_fraction=len(frame)/len(rows))
        write_json(directory / "identification.json", info)
        results.append({"spec": spec, "metrics": values, "identification": info})
        print(f"Teste {spec['id']}: N={len(frame)}; macro-F1={values['macro_f1']:.6f}", flush=True)
    for item in inventory:
        stat = (root / item["relative_path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
            raise ValueError("Vídeo original mudou durante a avaliação")
    verification = replay_final(output)
    from .final_reporting import report_final
    summary = report_final(output, config, rows, results)
    result = {"completed_at_utc": utc_now(), "extraction": {"motion": motion_report, "visual": visual_report},
              "models": results, "summary": summary, "verification": verification,
              "wall_seconds": time.perf_counter()-started, "training_seconds": 0, "test_evaluated": True}
    write_json(output / "results.json", result)
    write_json(output / "run_status.json", {"status": "complete", "at_utc": utc_now()})
    write_json(output / "deliverables_sha256.json", {str(path): file_hash(path) for path in output.rglob("*")
               if path.is_file() and path.name != "deliverables_sha256.json"})
    print(f"Avaliação final concluída: {output / 'report.md'}; sem treinamento.", flush=True)
    return result
