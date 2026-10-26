"""Training-only session balancing and day-level model comparison."""

import numpy as np
import pandas as pd

from .data import CLASS_IDS
from .evaluation import check_metadata, metrics


def validate_development_days(frame: pd.DataFrame, assignment: dict) -> None:
    check_metadata(frame)
    if frame.empty or frame.relative_path.duplicated().any():
        raise ValueError("Empty or duplicated development examples")
    seen = set()
    for split, days in assignment.items():
        if not days or len(days) != len(set(days)) or seen.intersection(days):
            raise ValueError("A recording day must belong to exactly one split")
        observed = frame.loc[frame.split.eq(split)]
        if set(observed.date) != set(days) or set(observed.class_name) != set(CLASS_IDS):
            raise ValueError("Recording days or classes differ from the sealed assignment")
        seen.update(days)
    if set(frame.split) != set(assignment) or set(frame.date) != seen:
        raise ValueError("Unexpected development split or day")


def session_weights(training_rows: pd.DataFrame) -> np.ndarray:
    """Equal probability mass for each observed training day/session/class stratum."""
    check_metadata(training_rows)
    if training_rows.empty or ("split" in training_rows and not training_rows.split.eq("train").all()):
        raise ValueError("Sampler may only inspect training rows")
    counts = training_rows.groupby(["date", "group", "class_name"])["relative_path"].transform("size")
    result = 1.0/counts.to_numpy(dtype=np.float64)
    return result/result.sum()


def day_metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict:
    if probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities do not match examples")
    return {day: metrics(rows.iloc[indices].class_id.to_numpy(), probabilities[indices].argmax(1))
            for day, indices in rows.groupby("date", sort=True).indices.items()}


def aggregate_days(values: list[dict]) -> dict:
    flat = [item for run in values for item in run.values()]
    if not flat: raise ValueError("No day-level measurements")
    return {"macro_f1": float(np.mean([m["macro_f1"] for m in flat])),
            "accuracy": float(np.mean([m["accuracy"] for m in flat])),
            "recalls": {label: float(np.mean([m["classification_report"][label]["recall"] for m in flat]))
                        for label in CLASS_IDS}}


def validation_gate(candidate: list[dict], baseline: list[dict], rules: dict,
                    original_day: str = "2022_6_21") -> dict:
    if len(candidate) != len(baseline) or not candidate or any(set(c) != set(b) for c, b in zip(candidate, baseline)):
        raise ValueError("Paired seed/day measurements are required")
    current, reference = aggregate_days(candidate), aggregate_days(baseline)
    recalls = {name: current["recalls"][name]-reference["recalls"][name] for name in CLASS_IDS}
    existing_delta = float(np.mean([c[original_day]["macro_f1"]-b[original_day]["macro_f1"]
                                    for c, b in zip(candidate, baseline)]))
    deltas = {"mean_day_macro_f1": current["macro_f1"]-reference["macro_f1"],
              "mean_day_recalls": recalls, "existing_day_macro_f1": existing_delta}
    checks = {"mean_day_macro_f1_gain": deltas["mean_day_macro_f1"] >= rules["mean_day_macro_f1_gain"],
              "medium_recall_gain": recalls["medium"] >= rules["mean_day_medium_recall_gain"],
              "other_recalls_preserved": all(value >= -rules["max_other_mean_day_recall_drop"]
                                             for name, value in recalls.items() if name != "medium"),
              "existing_day_preserved": existing_delta >= -rules["max_existing_day_macro_f1_drop"]}
    return {"candidate": current, "baseline": reference, "deltas": deltas, "checks": checks,
            "eligible_for_heldout": all(checks.values())}


def session_bootstrap_delta(rows: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray,
                            seed: int, replicates: int) -> dict:
    """Paired cluster bootstrap preserves clips from the same recording session."""
    if baseline.shape != (len(rows),) or candidate.shape != baseline.shape or replicates < 100:
        raise ValueError("Invalid paired bootstrap inputs")
    actual = rows.class_id.to_numpy()
    clusters = list(rows.groupby(["date", "group"], sort=True).indices.values())
    if len(clusters) < 2: raise ValueError("At least two recording sessions are required")
    rng = np.random.default_rng(seed)
    differences = (candidate == actual).astype(float)-(baseline == actual).astype(float)
    draws = []
    for _ in range(replicates):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        draws.append(float(differences[indices].mean()))
    low, high = np.quantile(draws, [.025, .975])
    return {"accuracy_delta": float(differences.mean()), "lower_95": float(low), "upper_95": float(high),
            "clusters": len(clusters), "replicates": replicates, "seed": seed,
            "unit": "recording date/session; one research site, not farm-level uncertainty"}
