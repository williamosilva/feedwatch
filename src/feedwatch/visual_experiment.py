"""Orquestra extração congelada, classificador linear e comparação por caminho."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import CLASS_IDS, COLUMNS, data_root, file_hash, protect_originals, read_manifest, write_json
from .evaluation import NAMES, check_metadata, metrics, save_evaluation
from .experiment import load_train_val, pilot_sample
from .linear_probe import load_probe, plot_history, probabilities, train_probe
from .motion import identity_hash
from .visual import (EMBEDDING_DIM, cache_signature, decode_visual, encode_frames, load_embedding_cache,
                     load_extractor, save_embedding_cache, state_hash, visual_identity)


def gpu_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"available": False}
    return {"available": True, "device": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "scope": "PyTorch allocator in this process; not total board usage"}


def prepare_video(record, root, config, transform, cache, config_id):
    started = time.perf_counter()
    try:
        signature = cache_signature(root, record["relative_path"], config_id)
        entry, warning = load_embedding_cache(cache, signature, config["frames_per_video"])
        if entry is not None:
            return {"record": record, "signature": signature, "ok": True, "cache_hit": True,
                    "cache_warning": warning, "error": "", **entry, "seconds": time.perf_counter()-started}
        decoded = decode_visual(root, record["relative_path"], config, transform)
        if signature != cache_signature(root, record["relative_path"], config_id):
            raise ValueError("Arquivo mudou durante a decodificação")
        return {"record": record, "signature": signature, "cache_hit": False, "cache_warning": warning, **decoded}
    except (OSError, ValueError) as error:
        return {"record": record, "signature": None, "ok": False, "cache_hit": False, "cache_warning": "",
                "error": str(error), "details": {}, "seconds": time.perf_counter()-started}


def compare_predictions(current: pd.DataFrame, baseline: Path, output: Path) -> dict:
    tables = {"mobilenet_linear": current}
    for name in ("dummy", "random_forest"):
        tables[name] = pd.read_csv(baseline / name / "predictions.csv")
    for table in tables.values():
        check_metadata(table)
        if any(CLASS_IDS.get(label) != ident for label, ident in zip(table.predicted_class, table.predicted_id)):
            raise ValueError("IDs/classes previstos inconsistentes na comparação")
    common = sorted(set.intersection(*(set(frame.relative_path) for frame in tables.values())))
    if not common:
        raise ValueError("Nenhum exemplo comum para comparação")
    common_metrics, own_metrics = {}, {}
    actual = current.set_index("relative_path").loc[common, "class_id"].to_numpy()
    for name, table in tables.items():
        aligned = table.set_index("relative_path").loc[common]
        np.testing.assert_array_equal(aligned.class_id, actual)
        own_metrics[name] = metrics(table.class_id.to_numpy(), table.predicted_id.to_numpy())
        common_metrics[name] = metrics(actual, aligned.predicted_id.to_numpy())
    deltas = {name: {metric: 100*(common_metrics["mobilenet_linear"][metric]-common_metrics[name][metric])
                     for metric in ("accuracy", "macro_f1")} for name in ("dummy", "random_forest")}
    per_class = [{"class_name": label, **{f"{metric}_delta_pp_vs_rf": 100*(common_metrics["mobilenet_linear"]["classification_report"][label][metric]
                       - common_metrics["random_forest"]["classification_report"][label][metric])
                 for metric in ("precision", "recall", "f1-score")}} for label in NAMES]
    result = {"common_examples": len(common), "own_validation": own_metrics, "common_validation": common_metrics,
              "delta_pp_vs": deltas, "per_class_delta_vs_rf": per_class}
    write_json(output / "comparison.json", result)
    pd.DataFrame([{"model": name, "own_examples": own_metrics[name]["examples"], "common_examples": len(common),
                   "common_accuracy": values["accuracy"], "common_macro_f1": values["macro_f1"]} for name, values in common_metrics.items()]).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class_delta_vs_rf.csv", index=False)
    pd.DataFrame({"relative_path": common}).to_csv(output / "common_val_examples.csv", index=False)
    return result


def write_visual_report(output: Path, result: dict):
    lines = ["# MobileNetV3 Small congelada + classificador linear", "",
             f"Modo: {result['mode']}. Teste não aberto nem avaliado. Somente a cabeça linear foi treinada.", "",
             f"Extração: {result['extraction']['wall_seconds']:.2f} s; treinamento: {result['training']['training_seconds']:.2f} s.",
             f"Melhor checkpoint: época {result['training']['selected_epoch']} de {result['training']['epochs']}, selecionada pelo macro-F1 da validação.", "",
             "| Divisão | Solicitados | Usados | Falhas |", "|---|---:|---:|---:|"]
    for name, count in result["counts"].items():
        lines.append(f"| {name} | {count['requested']} | {count['used']} | {count['failed']} |")
    lines += ["", "## Comparação na validação comum", "", "| Modelo | N | Acurácia | Macro-F1 |", "|---|---:|---:|---:|"]
    for name, item in result["comparison"]["common_validation"].items():
        lines.append(f"| {name} | {item['examples']} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |")
    for name, delta in result["comparison"]["delta_pp_vs"].items():
        lines += ["", f"Linear − {name}: acurácia {delta['accuracy']:+.3f} pp; macro-F1 {delta['macro_f1']:+.3f} pp."]
    lines += ["", "## Desempenho por classe do visual", "", "| Classe | Precisão | Recall | F1 | N |", "|---|---:|---:|---:|---:|"]
    for label in NAMES:
        c = result["validation"]["classification_report"][label]
        lines.append(f"| {label} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1-score']:.6f} | {int(c['support'])} |")
    lines += ["", "Matriz: linhas reais, colunas previstas na ordem none, weak, medium, strong.", "",
              "| Real / prevista | none | weak | medium | strong |", "|---|---:|---:|---:|---:|"]
    for label, values in zip(NAMES, result["validation"]["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## F1 por classe: diferença para a floresta", "",
              "| Classe | F1 da floresta | F1 do visual | Diferença (pp) |", "|---|---:|---:|---:|"]
    common = result["comparison"]["common_validation"]
    for label in NAMES:
        rf = common["random_forest"]["classification_report"][label]["f1-score"]
        visual = common["mobilenet_linear"]["classification_report"][label]["f1-score"]
        lines.append(f"| {label} | {rf:.6f} | {visual:.6f} | {100*(visual-rf):+.3f} |")
    train = result["train_resubstitution"]
    lines += ["", f"No checkpoint selecionado, treino: acurácia {train['accuracy']:.6f}; macro-F1 {train['macro_f1']:.6f}. "
              "São métricas de reutilização dos dados de ajuste, não resultados de generalização."]
    history = pd.read_csv(output / "classifier" / "history.csv")
    lines += ["", f"Da primeira à última época, loss de treino: {history.train_loss.iloc[0]:.6f} → {history.train_loss.iloc[-1]:.6f}; "
              f"loss de validação: {history.val_loss.iloc[0]:.6f} → {history.val_loss.iloc[-1]:.6f}. "
              "O checkpoint é escolhido pelo macro-F1, que não é equivalente à entropia cruzada."]
    for stage in ("extraction", "training"):
        memory = result[stage]["gpu_memory"]
        if memory["available"]:
            lines += ["", f"GPU ({stage}): pico alocado {memory['peak_allocated_bytes']/2**20:.2f} MiB; "
                      f"reservado {memory['peak_reserved_bytes']/2**20:.2f} MiB, somente allocator PyTorch deste processo."]
    lines += ["", "## Verificação e limites", "",
              "O checkpoint foi recarregado e reproduz as previsões salvas. Parâmetros e buffers do extrator, inclusive BatchNorm, permaneceram idênticos.",
              "Treino é reutilização dos dados de ajuste, não medida de generalização. Validação foi usada para selecionar a época: esta métrica também participa da seleção do modelo.",
              "Média temporal de embeddings não representa ordem ou velocidade do movimento. ImageNet é um domínio diferente de atividade alimentar de peixes.",
              "Corte proporcional de bordas seguido pelo preset oficial (lado menor 256, center crop 224, RGB e normalização ImageNet), sem padding. O recorte central descarta laterais; difere do esticamento 320×180 em cinza da referência de movimento.",
              "Não houve fine-tuning nem CNN treinada do zero. Comparação envolve características, pré-processamento e seleção de checkpoint diferentes; não isola causalmente qual mudança explica um ganho ou perda.",
              "Um único dia de validação e clipes correlacionados não comprovam generalização para outras fazendas. Intensidade observada não comprova fome ou saciedade.",
              "A identidade do cache inclui código, configuração, pesos, versões e dispositivo. A origem usa caminho, tamanho e mtime, não hash binário completo dos vídeos.", "",
              "Curvas: `classifier/curves.png`, `classifier/curves.svg`, `classifier/history.csv`. Métricas e previsões: `val/`. Pesos treinados: `classifier/best_classifier.pt`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_visual(root: Path, manifest: Path, splits: Path, baseline: Path, config_path: Path,
               cache: Path, weights_dir: Path, output: Path, workers: int = 4,
               sample_per_class: int | None = None, device_name: str = "auto") -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = data_root(root)
    for path in (cache, weights_dir, output):
        protect_originals(root, path)
    if baseline.resolve() == output.resolve() or baseline.resolve() in output.resolve().parents:
        raise ValueError("Use uma nova pasta de experimento, fora da referência")
    if (output / "results.json").exists():
        raise ValueError("Experimento já concluído; informe outro --output-dir para preservar os resultados")
    if not 1 <= workers <= 16:
        raise ValueError("workers deve estar entre 1 e 16")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ecfg = config["extraction"]
    x1, y1, x2, y2 = ecfg["crop_xyxy"]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or ecfg["weights"] != "IMAGENET1K_V1":
        raise ValueError("Recorte/pesos inválidos")
    parts, provenance = load_train_val(manifest, splits)
    for name in ("train", "val"):
        recorded = read_manifest(baseline / f"used_{name}.csv")
        expected = parts[name][parts[name].relative_path.isin(recorded.relative_path)]
        if not recorded.sort_values("relative_path").reset_index(drop=True).equals(expected.sort_values("relative_path").reset_index(drop=True)):
            raise ValueError("Referência não corresponde às divisões existentes")
        parts[name] = recorded if sample_per_class is None else pilot_sample(recorded, sample_per_class, config["seed"])
    requested = pd.concat([frame.assign(split=name) for name, frame in parts.items()], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_status.json", {"status": "running"})
    requested.to_csv(output / "requested_examples.csv", index=False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    init_started = time.perf_counter()
    model, transform, weights_info = load_extractor(weights_dir, device)
    init_seconds = time.perf_counter()-init_started
    initial_state = state_hash(model)
    identity = visual_identity(ecfg, weights_info, device)
    config_id = identity_hash(identity)
    write_json(output / "config.json", {"experiment": config, "extraction_identity": identity, "config_id": config_id,
                                         "workers": workers, "device": str(device), "sample_per_class": sample_per_class})
    write_json(output / "classes.json", CLASS_IDS)
    write_json(output / "provenance.json", provenance)
    write_json(cache / config_id / "config.json", identity)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    extraction_started, logs, means, used_records = time.perf_counter(), [], [], []
    records = requested.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(records), workers):
            prepared = list(executor.map(lambda record: prepare_video(record, root, ecfg, transform, cache, config_id), records[start:start+workers]))
            new = [item for item in prepared if item["ok"] and not item["cache_hit"]]
            if new:
                inference_started = time.perf_counter()
                encoded = encode_frames(model, torch.cat([item["frames"] for item in new]), device)
                inference_seconds = time.perf_counter()-inference_started
                for item, embeddings in zip(new, np.split(encoded, len(new))):
                    item["embeddings"] = embeddings
                    item["seconds"] += inference_seconds/len(new)
                    save_embedding_cache(cache, item["signature"], embeddings, item["details"], item["seconds"])
                    del item["frames"]
            for item in prepared:
                record = item["record"]
                logs.append({**record, "ok": item["ok"], "cache_hit": item["cache_hit"], "cache_warning": item["cache_warning"],
                             "error": item["error"], "processing_seconds": item["seconds"], "details": json.dumps(item["details"])})
                if item["ok"]:
                    means.append(item["embeddings"].mean(axis=0))
                    used_records.append(record)
            if start == 0 or (start+len(prepared)) % 100 == 0 or start+len(prepared) == len(records):
                print(f"Visual {start+len(prepared)}/{len(records)}; usados {len(means)}; falhas {len(logs)-len(means)}; {time.perf_counter()-extraction_started:.1f}s", flush=True)
                pd.DataFrame(logs).to_csv(output / "extraction.csv", index=False)
    extraction_seconds = time.perf_counter()-extraction_started
    final_state = state_hash(model)
    if initial_state != final_state or any(module.training for module in model.modules()):
        raise ValueError("Extrator mudou durante a extração")
    extraction_memory = gpu_memory(device)
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    log = pd.DataFrame(logs)
    log.to_csv(output / "extraction.csv", index=False)
    log[~log.ok].to_csv(output / "extraction_failures.csv", index=False)
    used = pd.DataFrame(used_records)
    embeddings = np.asarray(means, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError("Nenhum embedding válido")
    used.to_csv(output / "used_examples.csv", index=False)
    np.save(output / "mean_embeddings.npy", embeddings)
    counts = {}
    indices = {}
    for name in ("train", "val"):
        indices[name] = np.flatnonzero(used.split.eq(name))
        subset = used.iloc[indices[name]]
        if set(subset.class_id) != set(range(4)):
            raise ValueError(f"Classes ausentes após extração em {name}")
        subset[COLUMNS].to_csv(output / f"used_{name}.csv", index=False)
        counts[name] = {"requested": len(parts[name]), "used": len(subset), "failed": len(parts[name])-len(subset),
                        "by_class": subset.class_name.value_counts().to_dict(),
                        "same_examples_as_reference": set(subset.relative_path) == set(read_manifest(baseline / f"used_{name}.csv").relative_path)}
    training = train_probe(embeddings[indices["train"]], used.iloc[indices["train"]].class_id.to_numpy(),
                           embeddings[indices["val"]], used.iloc[indices["val"]].class_id.to_numpy(),
                           config["training"], config["seed"], device, output / "classifier",
                           {"config_id": config_id, "weights": weights_info, "used_train_sha256": file_hash(output / "used_train.csv")})
    training_memory = gpu_memory(device)
    train_result = save_evaluation(used.iloc[indices["train"]], training.pop("train_probabilities"), output / "train", "train")
    val_result = save_evaluation(used.iloc[indices["val"]], training.pop("val_probabilities"), output / "val", "val")
    head, bundle = load_probe(output / "classifier" / "best_classifier.pt", device)
    verify_x = torch.from_numpy((embeddings[indices["val"]]-bundle["mean"].numpy())/bundle["std"].numpy()).to(device)
    verified = probabilities(head, verify_x)
    saved = pd.read_csv(output / "val" / "predictions.csv")
    np.testing.assert_array_equal(verified.argmax(1), saved.predicted_id)
    np.testing.assert_allclose(verified, saved[[f"prob_{name}" for name in NAMES]], atol=1e-7)
    plot_history(output / "classifier")
    comparison = compare_predictions(saved, baseline, output)
    result = {"mode": "pilot" if sample_per_class else "full_train_val", "counts": counts,
              "extraction": {"wall_seconds": extraction_seconds, "initialization_seconds": init_seconds,
                             "cache_hits": int(log.cache_hit.sum()), "new_successes": int((log.ok & ~log.cache_hit).sum()),
                             "failed": int((~log.ok).sum()), "cache_warnings": int(log.cache_warning.ne("").sum()),
                             "sampled_frames_used": len(used)*ecfg["frames_per_video"], "gpu_memory": extraction_memory},
              "training": {**training, "gpu_memory": training_memory}, "train_resubstitution": train_result,
              "validation": val_result, "comparison": comparison, "extractor_state_unchanged": initial_state == final_state,
              "extractor_state_sha256": initial_state, "checkpoint_reproduces_predictions": True,
              "test_evaluated": False, "config_id": config_id, "provenance": provenance}
    write_json(output / "results.json", result)
    write_visual_report(output, result)
    write_json(output / "run_status.json", {"status": "complete"})
    print(f"Visual: val_accuracy={val_result['accuracy']:.6f}; val_macro_f1={val_result['macro_f1']:.6f}; época {training['selected_epoch']}", flush=True)
    return result
