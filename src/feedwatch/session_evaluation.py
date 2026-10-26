"""Import independent session reviews and score saved predictions by elapsed time.

No predictions are needed to create a review form. Human ``unobservable`` time
is not a negative feeding label; missing model output on observable time is an
error, including when computing temporal distributions.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from .video_overview import CLASSES, summarize_video

HUMAN_LABELS = (*CLASSES, "unobservable")
HOLISTIC_LABELS = (*HUMAN_LABELS, "ambiguous")
REVIEW_FIELDS = ("session_id", "reviewer_id", "domain_expertise", "blinded", "kind",
                 "start_seconds", "end_seconds", "label", "notes")
CONSENSUS_FIELDS = ("session_id", "kind", "start_seconds", "end_seconds", "label", "notes")
EPS = 1e-6


def review_grid(session_id: str, duration: float) -> list[dict]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    rows = []
    for i in range(math.ceil(duration / 2)):
        rows.append(dict(session_id=session_id, kind="interval", start_seconds=2. * i,
                         end_seconds=min(duration, 2. * i + 2), label="", notes=""))
    rows.append(dict(session_id=session_id, kind="whole", start_seconds=0.,
                     end_seconds=duration, label="", notes=""))
    for i, kind in enumerate(("start", "middle", "end")):
        rows.append(dict(session_id=session_id, kind=kind, start_seconds=duration * i / 3,
                         end_seconds=duration * (i + 1) / 3, label="", notes=""))
    return rows


def _key(row):
    return row["kind"], round(float(row["start_seconds"]), 6), round(float(row["end_seconds"]), 6)


def _bool(value):
    if value is True or str(value).strip().lower() in ("true", "1"):
        return True
    if value is False or str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError("Boolean must be true/false or 1/0")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_review(path: Path) -> dict:
    rows = read_csv(path)
    if not rows:
        raise ValueError("Empty review")
    profiles = {(r.get("reviewer_id", "").strip(), _bool(r.get("domain_expertise")),
                 _bool(r.get("blinded"))) for r in rows}
    if len(profiles) != 1:
        raise ValueError("Reviewer metadata must be consistent in every CSV row")
    reviewer_id, domain, blinded = profiles.pop()
    if not reviewer_id or not blinded:
        raise ValueError("An identified, blinded reviewer is required")
    return dict(reviewer_id=reviewer_id, domain_expertise=domain, blinded=blinded, rows=rows)


def _validated_rows(rows, session_id, duration, allow_blank=False):
    expected = {_key(r): r for r in review_grid(session_id, duration)}
    result = {}
    for row in rows:
        if row.get("session_id") != session_id:
            raise ValueError("Review contains another session")
        key = _key(row)
        if key not in expected or key in result:
            raise ValueError("Review has duplicate, overlapping or unexpected intervals")
        label = row.get("label", "").strip()
        labels = HUMAN_LABELS if row["kind"] == "interval" else HOLISTIC_LABELS
        if label not in labels and not (allow_blank and not label):
            raise ValueError(f"Invalid or missing human label: {label!r}")
        result[key] = {**expected[key], "label": label, "notes": row.get("notes", "")}
    if not allow_blank and result.keys() != expected.keys():
        raise ValueError("Every interval and holistic assessment must be reviewed")
    return result


def import_reviews(session, review_a, review_b, consensus_rows=()):
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ValueError("Two different reviewers are required")
    if not review_a.get("blinded") or not review_b.get("blinded"):
        raise ValueError("Reviews must be completed without model output")
    if not any(r.get("domain_expertise") for r in (review_a, review_b)):
        raise ValueError("At least one reviewer must declare domain expertise")
    args = (session["session_id"], session["duration_seconds"])
    a = _validated_rows(review_a["rows"], *args)
    b = _validated_rows(review_b["rows"], *args)
    c = _validated_rows(consensus_rows, *args, allow_blank=True)
    rows, disagreements = [], []
    for key, row in a.items():
        la, lb = row["label"], b[key]["label"]
        resolved = c.get(key, {}).get("label")
        if la != lb:
            if not resolved:
                raise ValueError(f"Explicit consensus required for {key}")
            if not c[key].get("notes", "").strip():
                raise ValueError("Document the reason for disagreement consensus")
            disagreements.append({**row, "reviewer_a": la, "reviewer_b": lb,
                                  "consensus": resolved, "notes": c[key]["notes"]})
        elif resolved and resolved != la:
            raise ValueError("Consensus cannot silently override agreeing reviewers")
        rows.append({**row, "label": resolved or la})
    matrix = [[0. for _ in HUMAN_LABELS] for _ in HUMAN_LABELS]
    for key, row in a.items():
        if row["kind"] != "interval":
            continue
        matrix[HUMAN_LABELS.index(row["label"])][HUMAN_LABELS.index(b[key]["label"])] += row["end_seconds"] - row["start_seconds"]
    total = sum(map(sum, matrix))
    observed = sum(matrix[i][i] for i in range(5)) / total
    expected = sum(sum(matrix[i]) * sum(row[i] for row in matrix) for i in range(5)) / total**2
    agreement = {"labels": list(HUMAN_LABELS), "duration_confusion_seconds": matrix,
                 "agreement_fraction": observed,
                 "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 - EPS else None,
                 "disagreements": disagreements,
                 "holistic_agreement": {kind: a[next(k for k in a if k[0] == kind)]["label"] ==
                                        b[next(k for k in b if k[0] == kind)]["label"]
                                        for kind in ("whole", "start", "middle", "end")}}
    return rows, agreement


def validate_session_set(sessions, allow_practice=False):
    if not sessions:
        raise ValueError("At least one session is required")
    purposes = {s.get("purpose", "evaluation") for s in sessions}
    if not purposes <= {"evaluation", "practice"}:
        raise ValueError("Unknown session purpose")
    if len(purposes) > 1:
        raise ValueError("Practice and evaluation recordings must be reported separately")
    seen = set()
    groups = defaultdict(lambda: defaultdict(set))
    for session in sessions:
        sid = session.get("session_id")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("Session IDs must be distinct and nonempty")
        seen.add(sid)
        duration = float(session["duration_seconds"])
        practice = session.get("purpose") == "practice"
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        if practice and not allow_practice:
            raise ValueError("Practice sessions cannot be included in final evaluation")
        if not practice and not 50 - EPS <= duration <= 60 + EPS:
            raise ValueError("Final evaluation requires continuous 50–60 second recordings")
        for field in ("camera_id", "source_video_sha256", "authorization", "continuity_note",
                      "source_url", "selection_locked_at_utc", "selection_basis"):
            if not isinstance(session.get(field), str) or not session[field].strip():
                raise ValueError(f"Missing session provenance: {field}")
        date.fromisoformat(session["recorded_date"])
        if not re.fullmatch(r"[a-fA-F0-9]{64}", session["source_video_sha256"]):
            raise ValueError("Invalid source-video SHA256")
        locked_at = datetime.fromisoformat(session["selection_locked_at_utc"].replace("Z", "+00:00"))
        if locked_at.tzinfo is None:
            raise ValueError("Selection lock timestamp must include a timezone")
        if session.get("continuous") is not True:
            raise ValueError("Continuity must be explicitly attested; do not concatenate clips")
        if not practice and session.get("predictions_unseen_at_selection") is not True:
            raise ValueError("Final sessions must be selected before consulting predictions")
        split = session.get("split")
        if split not in ("development", "final"):
            raise ValueError("Split must be development or final")
        for field in ("camera_id", "recorded_date", "source_video_sha256"):
            groups[field][session[field]].add(split)
    for field, values in groups.items():
        if any(len(splits) > 1 for splits in values.values()):
            raise ValueError(f"Development/final leakage through {field}")


def _prediction_rows(windows, duration):
    """Validate failed rows too: they must not hide malformed/overlapping input."""
    rows = []
    for item in windows:
        start, end = float(item["start_seconds"]), float(item["end_seconds"])
        if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start or end > duration + EPS:
            raise ValueError("Invalid prediction interval")
        status = item.get("status")
        if status not in ("ok", "failed", "incomplete", "error"):
            raise ValueError("Unknown prediction status")
        label = item.get("predicted_class") if status == "ok" else None
        if status == "ok" and label not in CLASSES:
            raise ValueError("Invalid prediction class")
        rows.append(dict(start_seconds=start, end_seconds=min(end, duration),
                         status="ok" if label else "failed", predicted_class=label))
    rows.sort(key=lambda row: row["start_seconds"])
    if any(b["start_seconds"] < a["end_seconds"] - EPS for a, b in zip(rows, rows[1:])):
        raise ValueError("Overlapping prediction intervals")
    return rows


def validate_prediction_record(record, session):
    """Bind a saved inference JSON to this source, duration, window plan and model."""
    if record.get("input_signature", {}).get("sha256") != session["source_video_sha256"]:
        raise ValueError("Inference source hash differs from selected session")
    if record.get("mode") != "sequential":
        raise ValueError("Session evaluation requires sequential inference")
    duration = float(session["duration_seconds"])
    metadata = record.get("metadata", {})
    actual_duration = float(metadata.get("duration_seconds", math.nan))
    fps, count = float(metadata.get("fps", math.nan)), float(metadata.get("frame_count", math.nan))
    if not all(math.isfinite(v) and v > 0 for v in (actual_duration, fps, count)):
        raise ValueError("Inference metadata is missing or invalid")
    if not math.isclose(actual_duration, duration, abs_tol=EPS, rel_tol=0) or not math.isclose(count / fps, duration, abs_tol=EPS, rel_tol=0):
        raise ValueError("Inference duration differs from selected session")
    prepared = session.get("video_verification")
    if prepared and (prepared["frame_count"] != count or not math.isclose(prepared["fps"], fps, abs_tol=EPS, rel_tol=0)):
        raise ValueError("Inference frame count/FPS differs from prepared source")
    identity = record.get("food_model", {})
    if not isinstance(identity.get("kind"), str) or not identity["kind"].strip() or not re.fullmatch(r"[a-fA-F0-9]{64}", identity.get("checkpoint_sha256", "")):
        raise ValueError("Inference model kind and checkpoint SHA256 are required")
    if identity.get("class_mapping") != dict(zip(CLASSES, range(4))):
        raise ValueError("Inference model class mapping differs")
    windows = record.get("windows")
    expected = [r for r in review_grid(session["session_id"], duration) if r["kind"] == "interval"]
    if not isinstance(windows, list) or len(windows) != len(expected):
        raise ValueError("Inference does not contain the complete two-second window plan")
    for index, (window, slot) in enumerate(zip(windows, expected)):
        if window.get("window_id") != index or any(not math.isclose(float(window[k]), slot[k], abs_tol=EPS, rel_tol=0) for k in ("start_seconds", "end_seconds")):
            raise ValueError("Inference window plan differs from the selected session")
    validated = _prediction_rows(windows, duration)
    return validated, dict(kind=identity["kind"], checkpoint_sha256=identity["checkpoint_sha256"],
                           class_mapping=identity["class_mapping"], seed=identity.get("seed"),
                           extraction_config_id=identity.get("extraction_config_id"))


def _classification_metrics(matrix):
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(4))
    by_class = {}
    for i, label in enumerate(CLASSES):
        tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(r[i] for r in matrix)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        by_class[label] = dict(support_seconds=support, precision=precision, recall=recall,
                               f1=2 * precision * recall / (precision + recall) if precision + recall else 0.)
    return dict(reference_labels=list(CLASSES), prediction_labels=[*CLASSES, "missing"],
                confusion_seconds=matrix, observable_seconds=total,
                missing_prediction_seconds=sum(r[4] for r in matrix),
                accuracy=correct / total if total else None,
                macro_f1=sum(v["f1"] for v in by_class.values()) / 4 if total else None,
                per_class=by_class)


def _single_winner(overview):
    winners = overview["predominant_classes"]
    return winners[0] if len(winners) == 1 else "ambiguous" if winners else None


def _assessment(expected, actual):
    eligible = expected != "unobservable"
    return dict(reference=expected, prediction=actual, eligible=eligible,
                available=actual is not None, correct=(expected == actual) if eligible else None)


def evaluate_session(session, windows, review_a, review_b, consensus_rows=()):
    duration = float(session["duration_seconds"])
    human, agreement = import_reviews(session, review_a, review_b, consensus_rows)
    predictions = _prediction_rows(windows, duration)
    overview = summarize_video(predictions, duration)
    matrix = [[0. for _ in range(5)] for _ in range(4)]
    reference = {label: 0. for label in CLASSES}
    predicted = {label: 0. for label in CLASSES}
    for row in human:
        if row["kind"] != "interval" or row["label"] == "unobservable":
            continue
        label = row["label"]
        seconds = row["end_seconds"] - row["start_seconds"]
        reference[label] += seconds
        covered = 0.
        for p in predictions:
            overlap = max(0., min(row["end_seconds"], p["end_seconds"]) - max(row["start_seconds"], p["start_seconds"]))
            if p["status"] == "ok" and overlap:
                matrix[CLASSES.index(label)][CLASSES.index(p["predicted_class"])] += overlap
                predicted[p["predicted_class"]] += overlap
                covered += overlap
        matrix[CLASSES.index(label)][4] += max(0., seconds - covered)
    metrics = _classification_metrics(matrix)
    observable, missing = metrics["observable_seconds"], metrics["missing_prediction_seconds"]
    fractions = {label: reference[label] / observable if observable else None for label in CLASSES}
    predicted_fractions = {label: predicted[label] / observable if observable else None for label in CLASSES}
    predicted_fractions["missing"] = missing / observable if observable else None
    distribution_error = (sum(abs(reference[l] - predicted[l]) for l in CLASSES) + missing) / (2 * observable) if observable else None
    holistic = {r["kind"]: r["label"] for r in human if r["kind"] != "interval"}
    phases = {p["key"]: p for p in overview["evolution"]["phases"]}
    phase_assessments = {kind: _assessment(holistic[kind], _single_winner(phases[kind]) if phases[kind]["comparable"] else None)
                         for kind in ("start", "middle", "end")}
    human_comparable = all(holistic[k] in CLASSES for k in ("start", "middle", "end"))
    direction = None
    if human_comparable:
        first, last = CLASSES.index(holistic["start"]), CLASSES.index(holistic["end"])
        direction = "lower" if last < first else "higher" if last > first else "same"
    comparison = overview["evolution"]["comparison"]
    predicted_direction = comparison["direction"] if comparison["available"] else None
    reference_windows = [dict(status="ok", start_seconds=r["start_seconds"], end_seconds=r["end_seconds"],
                              predicted_class=r["label"]) for r in human if r["kind"] == "interval" and r["label"] in CLASSES]
    reference_overview = summarize_video(reference_windows, duration)
    whole_prediction = _single_winner(overview)
    whole_available = overview["summary_available"]
    whole = _assessment(holistic["whole"], whole_prediction if whole_available else None)
    whole.update(raw_prediction=whole_prediction,
                 minimum_coverage_fraction=overview["summary_policy"]["min_coverage_fraction"],
                 unavailable_reason=overview["summary_reason"])
    return dict(session_id=session["session_id"], camera_id=session["camera_id"],
                recorded_date=session["recorded_date"], split=session["split"],
                purpose=session.get("purpose", "evaluation"),
                eligible_final=(session.get("purpose") != "practice" and session["split"] == "final"
                                and 50 - EPS <= duration <= 60 + EPS),
                duration_seconds=duration, human_observable_fraction=observable / duration,
                human_unobservable_seconds=duration - observable,
                model_coverage_fraction=overview["coverage_fraction"],
                model_coverage_of_observable_time=(observable - missing) / observable if observable else None,
                reviewers=[{k: r[k] for k in ("reviewer_id", "domain_expertise", "blinded")} for r in (review_a, review_b)],
                reviewer_agreement=agreement, interval_metrics=metrics,
                distribution=dict(reference_fractions=fractions, prediction_fractions_including_missing=predicted_fractions,
                                  total_variation_error=distribution_error,
                                  denominator="human-observable time; missing model output is retained"),
                whole=whole, phases=phase_assessments,
                interval_derived_predominance=_assessment(_single_winner(reference_overview) or "unobservable", whole["prediction"]),
                comparison=dict(reference=direction, prediction=predicted_direction, eligible=human_comparable,
                                available=comparison["available"], unavailable_reason=comparison["reason"],
                                correct=(predicted_direction == direction) if human_comparable else None,
                                reference_middle_differs=(holistic["start"] == holistic["end"] and holistic["middle"] != holistic["start"]) if human_comparable else None,
                                predicted_middle_differs=comparison["middle_differs"]),
                model_overview=overview)


def _group_metrics(results):
    matrix = [[sum(r["interval_metrics"]["confusion_seconds"][i][j] for r in results) for j in range(5)] for i in range(4)]
    group = dict(session_count=len(results), duration_seconds=sum(r["duration_seconds"] for r in results),
                 pooled_interval_metrics=_classification_metrics(matrix))
    for key in ("whole", "comparison"):
        scored = [r[key] for r in results if r[key]["eligible"]]
        group[key] = dict(eligible_sessions=len(scored), correct_sessions=sum(x["correct"] for x in scored),
                          unavailable_sessions=sum(not x["available"] for x in scored),
                          accuracy=sum(x["correct"] for x in scored) / len(scored) if scored else None)
    for name, getter in (("mean_session_interval_accuracy", lambda r: r["interval_metrics"]["accuracy"]),
                         ("mean_session_distribution_error", lambda r: r["distribution"]["total_variation_error"]),
                         ("mean_session_observable_coverage", lambda r: r["model_coverage_of_observable_time"])):
        values = [getter(r) for r in results if getter(r) is not None]
        group[name] = sum(values) / len(values) if values else None
    return group


def summarize_sessions(results):
    return dict(overall=_group_metrics(results),
                by_camera={camera: _group_metrics([r for r in results if r["camera_id"] == camera])
                           for camera in sorted({r["camera_id"] for r in results})},
                by_split={split: _group_metrics([r for r in results if r["split"] == split])
                          for split in sorted({r["split"] for r in results})})
