"""Métricas com IDs explícitos e saídas independentes do tipo de classificador."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import CLASS_IDS, COLUMNS, parse_relative_path, write_json

NAMES = [name for name, _ in sorted(CLASS_IDS.items(), key=lambda item: item[1])]
LABELS = list(range(len(NAMES)))


def check_metadata(frame: pd.DataFrame) -> None:
    if frame.relative_path.duplicated().any():
        raise ValueError("Identificadores de vídeo duplicados")
    for row in frame[COLUMNS].to_dict("records"):
        if row != parse_relative_path(row["relative_path"]):
            raise ValueError(f"Rótulo/ID divergente do caminho: {row['relative_path']}")


def metrics(y: np.ndarray, predictions: np.ndarray) -> dict:
    return {"examples": len(y), "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
            "classification_report": classification_report(y, predictions, labels=LABELS, target_names=NAMES,
                                                           output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist(), "confusion_labels": NAMES}


def save_evaluation(frame: pd.DataFrame, probabilities: np.ndarray, output: Path, split: str) -> dict:
    check_metadata(frame)
    if probabilities.shape != (len(frame), 4) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilidades inválidas")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    predicted = probabilities.argmax(axis=1)
    result = {**metrics(frame.class_id.to_numpy(), predicted), "evaluation_split": split,
              "generalization_estimate": split != "train"}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    rows = frame[COLUMNS].copy()
    rows["predicted_id"] = predicted
    rows["predicted_class"] = [NAMES[p] for p in predicted]
    rows["correct"] = predicted == rows.class_id.to_numpy()
    for index, name in enumerate(NAMES):
        rows[f"prob_{name}"] = probabilities[:, index]
    rows.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(result["confusion_matrix"], index=pd.Index(NAMES, name="actual"), columns=NAMES).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame([{**result["classification_report"][name], "class_name": name, "class_id": CLASS_IDS[name]}
                  for name in NAMES]).to_csv(output / "per_class.csv", index=False)
    errors = rows[~rows.correct].groupby(["class_name", "predicted_class"]).size().reset_index(name="count")
    errors.sort_values("count", ascending=False).to_csv(output / "error_pairs.csv", index=False)
    return result
