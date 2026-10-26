"""Experimento supervisionado de movimento: somente treino e validação existentes."""

from __future__ import annotations

import csv
import json
import pickle
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import (CLASS_IDS, COLUMNS, data_root, file_hash, protect_originals,
                   read_manifest, stratified_sample, utc_now, write_json)
from .motion import FEATURE_NAMES, cached_extract, extraction_identity, identity_hash, validate_config

LABELS = list(CLASS_IDS.values())
CLASS_NAMES = list(CLASS_IDS)


def load_train_val(manifest_path: Path, splits_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Não lê test.csv: verifica treino/validação contra datas e manifest originais."""
    manifest = read_manifest(manifest_path)
    validation = json.loads((splits_dir / "validation.json").read_text(encoding="utf-8"))
    assignment = json.loads((splits_dir / "assignment.json").read_text(encoding="utf-8"))
    if not validation["valid"] or validation["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Manifest/divisões diferem da preparação validada")
    if assignment != validation["assignment"] or set(assignment) != {"train", "val", "test"}:
        raise ValueError("Composição das datas diverge da separação registrada")
    days = [d for group in assignment.values() for d in group]
    if len(days) != len(set(days)) or set(days) != set(manifest.date):
        raise ValueError("Datas não formam uma partição da preparação")
    parts = {}
    for name in ("train", "val"):
        frame = read_manifest(splits_dir / f"{name}.csv")
        expected = manifest[manifest.date.isin(assignment[name])]
        if not frame.sort_values("relative_path").reset_index(drop=True).equals(expected.sort_values("relative_path").reset_index(drop=True)):
            raise ValueError(f"{name}.csv não corresponde aos dias inteiros da preparação")
        if set(frame.class_name) != set(CLASS_IDS):
            raise ValueError(f"{name}: faltam classes")
        parts[name] = frame
    if set(parts["train"].relative_path) & set(parts["val"].relative_path) or set(parts["train"].date) & set(parts["val"].date):
        raise ValueError("Vazamento entre treino e validação")
    paths = [manifest_path, splits_dir / "train.csv", splits_dir / "val.csv",
             splits_dir / "assignment.json", splits_dir / "validation.json"]
    return parts, {"assignment": assignment, "input_sha256": {str(p): file_hash(p) for p in paths},
                   "test_evaluated": False, "test_csv_opened": False}


def pilot_sample(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    if per_class < 1:
        raise ValueError("sample_per_class deve ser positivo")
    return pd.concat([stratified_sample(frame[frame.class_name == label], per_class, seed)
                      for label in CLASS_NAMES], ignore_index=True).sort_values("relative_path").reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Lista positiva: nenhuma coluna de identificação/rótulo entra em X."""
    result = frame.loc[:, FEATURE_NAMES].astype(np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Características não finitas; não será feita imputação por zeros")
    return result


def evaluate(model, validation: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    x, y = feature_matrix(validation), validation.class_id.to_numpy()
    started = time.perf_counter()
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    prediction_seconds = time.perf_counter() - started
    matrix = confusion_matrix(y, predictions, labels=LABELS)
    report = classification_report(y, predictions, labels=LABELS, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {"evaluation_split": "val", "examples": len(y),
               "accuracy": float(accuracy_score(y, predictions)),
               "macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
               "classification_report": report, "confusion_matrix": matrix.tolist(),
               "confusion_labels": CLASS_NAMES, "prediction_seconds": prediction_seconds}
    write_json(output / "metrics.json", metrics)
    pd.DataFrame([{**report[label], "class_name": label, "class_id": CLASS_IDS[label]}
                  for label in CLASS_NAMES]).to_csv(output / "per_class.csv", index=False)
    pd.DataFrame(matrix, index=pd.Index(CLASS_NAMES, name="actual"), columns=CLASS_NAMES).to_csv(output / "confusion_matrix.csv")
    rows = validation[COLUMNS].copy()
    rows["predicted_id"] = predictions
    rows["predicted_class"] = [CLASS_NAMES[int(value)] for value in predictions]
    rows["correct"] = predictions == y
    for index, label in enumerate(model.classes_):
        rows[f"prob_{CLASS_NAMES[int(label)]}"] = probabilities[:, index]
    rows.to_csv(output / "predictions.csv", index=False)
    errors = rows[~rows.correct].groupby(["class_name", "predicted_class"]).size().reset_index(name="count")
    errors = errors.sort_values(["count", "class_name", "predicted_class"], ascending=[False, True, True])
    errors.to_csv(output / "error_pairs.csv", index=False)
    rows.assign(error=~rows.correct).groupby("group").agg(examples=("error", "size"), errors=("error", "sum"),
                                                        error_rate=("error", "mean")).to_csv(output / "errors_by_group.csv")
    return {**metrics, "top_errors": errors.head(5).to_dict("records")}


def write_report(output: Path, report: dict) -> None:
    extraction = report["extraction"]
    lines = ["# Referência de movimento — validação", "",
             f"Modo: {report['mode']}. Seed: {report['config']['seed']}. Teste não avaliado.", "",
             f"Extração: {extraction['wall_seconds']:.2f} s nesta execução; cache: {extraction['cache_hits']} acertos; "
             f"novas extrações aprovadas: {extraction['new_successes']}; falhas: {extraction['failed']}.", "",
             "| Divisão | Solicitados | Usados | Falhas |", "|---|---:|---:|---:|"]
    for name, values in report["counts"].items():
        lines.append(f"| {name} | {values['requested']} | {values['used']} | {values['failed']} |")
    lines += ["", "| Modelo | Acurácia na validação | Macro-F1 na validação | Treino (s) |", "|---|---:|---:|---:|"]
    for name, values in report["models"].items():
        lines.append(f"| {name} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} | {values['training_seconds']:.3f} |")
    for name, values in report["models"].items():
        lines += ["", f"## {name}", "", "Matriz: linhas = classe real; colunas = classe prevista.", "",
                  "| Real / prevista | " + " | ".join(CLASS_NAMES) + " |", "|---|---:|---:|---:|---:|"]
        for label, row in zip(CLASS_NAMES, values["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")
        lines += ["", "| Classe | Precisão | Recall | F1 | Suporte |", "|---|---:|---:|---:|---:|"]
        for label in CLASS_NAMES:
            row = values["classification_report"][label]
            lines.append(f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} | {int(row['support'])} |")
        lines += ["", "Erros mais frequentes:", ""]
        lines += [f"- {e['class_name']} → {e['predicted_class']}: {e['count']} vídeos." for e in values["top_errors"]] or ["Nenhum erro."]
        lines += ["", f"Artefatos: `{name}/predictions.csv`, `per_class.csv`, `confusion_matrix.csv`, `error_pairs.csv`, `model.pkl`."]
    lines += ["", "## Limites", "",
              "Características globais resumem movimento aparente: peixes, água, reflexos, iluminação e câmera. Não detectam peixes nem comprovam fome/saciedade.",
              "O corte retira bordas/textos, mas não isola o tanque; o redimensionamento pode alterar proporções. Analisamos somente os primeiros 2 s na configuração inicial.",
              "Percentis e médias perdem ordem temporal e comportamento individual. A amostragem requer FPS/timestamps compatíveis com a grade CFR; arquivos incompatíveis são excluídos e registrados.",
              "Validação contém somente um dia; clipes próximos podem ser correlacionados. As métricas são por clipe, sem comprovar independência estatística ou generalização para outras fazendas.",
              "O piloto usa amostra balanceada apenas para verificar o pipeline; seus números não são a referência final. Nenhuma busca de hiperparâmetros foi feita.",
              "Falhas nunca viram movimento zero. O cache usa configuração, código, versões, raiz, caminho, tamanho e mtime; não detecta mudança binária que preserve tamanho e mtime.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(root: Path, manifest_path: Path, splits_dir: Path, config_path: Path,
                   cache_dir: Path, output: Path, workers: int, sample_per_class: int | None) -> dict:
    started = time.perf_counter()
    root = data_root(root)
    protect_originals(root, output)
    protect_originals(root, cache_dir)
    if workers < 1 or workers > 16:
        raise ValueError("Use workers entre 1 e 16")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config["extraction"])
    parts, provenance = load_train_val(manifest_path, splits_dir)
    index_report_path = manifest_path.parent / "index_report.json"
    if index_report_path.exists():
        index_report = json.loads(index_report_path.read_text(encoding="utf-8"))
        if Path(index_report["data_dir"]).resolve() != root or index_report["manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Raiz/manifest diferem da indexação original")
    if sample_per_class is not None:
        parts = {name: pilot_sample(frame, sample_per_class, config["seed"]) for name, frame in parts.items()}
    requested = pd.concat([frame.assign(split=name) for name, frame in parts.items()], ignore_index=True)
    identity = extraction_identity(config["extraction"])
    config_id = identity_hash(identity)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_status.json", {"status": "running", "started_at_utc": utc_now()})
    requested.to_csv(output / "requested_examples.csv", index=False)
    write_json(output / "config.json", {"experiment": config, "extraction_identity": identity, "config_id": config_id,
                                         "workers": workers, "sample_per_class": sample_per_class})
    write_json(cache_dir / config_id / "config.json", identity)
    write_json(output / "classes.json", CLASS_IDS)
    write_json(output / "provenance.json", provenance)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    extraction_started, last_progress = time.perf_counter(), time.perf_counter()
    logs, successes = [], []
    fields = [*COLUMNS, "split", "ok", "cache_hit", "cache_warning", "error", "elapsed_seconds",
              "extraction_seconds", "frames_decoded", "details", "source_signature"]
    with (output / "extraction.csv").open("w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(cached_extract, root, record["relative_path"], config["extraction"], cache_dir, config_id): record
                       for record in requested.to_dict("records")}
            for index, future in enumerate(as_completed(futures), start=1):
                record, result = futures[future], future.result()
                row = {**record, **{name: result[name] for name in ["ok", "cache_hit", "cache_warning", "error", "elapsed_seconds", "extraction_seconds"]},
                       "frames_decoded": result["details"].get("frames_decoded", 0),
                       "details": json.dumps(result["details"]), "source_signature": json.dumps(result["source_signature"])}
                writer.writerow(row)
                log_file.flush()
                logs.append(row)
                if result["ok"]:
                    successes.append({**record, **result["features"]})
                now = time.perf_counter()
                if index % 100 == 0 or index == len(requested) or now - last_progress > 15:
                    print(f"Movimento {index}/{len(requested)}; usados {len(successes)}; falhas {index - len(successes)}; "
                          f"{now - extraction_started:.1f}s", flush=True)
                    last_progress = now
    extraction_wall = time.perf_counter() - extraction_started
    log = pd.DataFrame(logs).sort_values(["split", "relative_path"])
    log.to_csv(output / "extraction.csv", index=False)
    log[~log.ok].to_csv(output / "extraction_failures.csv", index=False)
    features = pd.DataFrame(successes, columns=[*COLUMNS, "split", *FEATURE_NAMES]).sort_values(["split", "relative_path"])
    features.to_csv(output / "features.csv", index=False)
    counts, used = {}, {}
    for name, frame in parts.items():
        used[name] = features[features.split == name].copy()
        counts[name] = {"requested": len(frame), "used": len(used[name]), "failed": len(frame) - len(used[name]),
                        "requested_by_class": {label: int((frame.class_name == label).sum()) for label in CLASS_NAMES},
                        "used_by_class": {label: int((used[name].class_name == label).sum()) for label in CLASS_NAMES}}
        used[name][COLUMNS].to_csv(output / f"used_{name}.csv", index=False)
    extraction_report = {"wall_seconds": extraction_wall, "requested": len(requested), "used": len(features),
                         "failed": int((~log.ok).sum()), "cache_hits": int(log.cache_hit.sum()),
                         "new_successes": int((log.ok & ~log.cache_hit).sum()),
                         "cache_warnings": int(log.cache_warning.ne("").sum()),
                         "new_video_processing_seconds_sum": float(log.loc[~log.cache_hit, "extraction_seconds"].sum()),
                         "cached_original_processing_seconds_sum": float(log.loc[log.cache_hit, "extraction_seconds"].sum()),
                         "sampled_frames_used": int(log.loc[log.ok, "frames_decoded"].sum()), "workers": workers}
    write_json(output / "extraction_report.json", {"extraction": extraction_report, "counts": counts})
    if any(set(frame.class_id) != set(LABELS) for frame in used.values()):
        write_json(output / "run_status.json", {"status": "failed", "reason": "Classes ausentes após extração"})
        raise ValueError("Faltam classes após extração; consulte extraction_failures.csv. Nenhum modelo treinado.")
    x, y = feature_matrix(used["train"]), used["train"].class_id.to_numpy()
    models = {"dummy": DummyClassifier(strategy="most_frequent", random_state=config["seed"]),
              "random_forest": RandomForestClassifier(**config["random_forest"], random_state=config["seed"], n_jobs=workers)}
    metrics = {}
    for name, model in models.items():
        fit_started = time.perf_counter()
        model.fit(x, y)
        training_seconds = time.perf_counter() - fit_started
        metrics[name] = {**evaluate(model, used["val"], output / name), "training_seconds": training_seconds,
                         "parameters": model.get_params()}
        bundle = {"estimator": model, "feature_names": FEATURE_NAMES, "class_mapping": CLASS_IDS,
                  "extraction_identity": identity, "config_id": config_id, "sklearn_version": sklearn.__version__,
                  "used_train_sha256": file_hash(output / "used_train.csv"), "test_evaluated": False}
        with (output / name / "model.pkl").open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame({"feature": FEATURE_NAMES, "importance": models["random_forest"].feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(output / "random_forest" / "feature_importances.csv", index=False)
    report = {"created_at_utc": utc_now(), "mode": "pilot" if sample_per_class is not None else "full_train_val",
              "config": config, "config_id": config_id, "feature_names": FEATURE_NAMES,
              "provenance": provenance, "counts": counts, "extraction": extraction_report, "models": metrics,
              "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                           "pandas": pd.__version__, "sklearn": sklearn.__version__}, "test_evaluated": False,
              "dummy_majority_class": CLASS_NAMES[int(models["dummy"].classes_[np.argmax(models["dummy"].class_prior_)])],
              "total_wall_seconds": time.perf_counter() - started}
    write_json(output / "results.json", report)
    write_report(output, report)
    write_json(output / "run_status.json", {"status": "complete", "finished_at_utc": utc_now()})
    for name, values in metrics.items():
        print(f"{name}: val_accuracy={values['accuracy']:.6f}; val_macro_f1={values['macro_f1']:.6f}; fit={values['training_seconds']:.3f}s")
    print(f"Relatório: {output / 'report.md'}; teste não avaliado.", flush=True)
    return report
