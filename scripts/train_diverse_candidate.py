"""Frozen day-disjoint experiment: new sessions, paired baseline, gated holdout.

Stages are explicit so acquisition can run before the GPU is available. Published
historical tests and the previous additional sample are never read by this script.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from feedwatch.data import CLASS_IDS, COLUMNS, file_hash, parse_relative_path, utc_now, write_json
from feedwatch.diverse_training import (aggregate_days, day_metrics, session_bootstrap_delta,
                                       session_weights, validate_development_days, validation_gate)
from feedwatch.evaluation import save_evaluation
from feedwatch.experiment_integrity import seal_inputs, verify_checkpoint_run
from feedwatch.research_acquisition import RemoteZip, verified_member
from feedwatch.temporal_data import load_sequences, validate_chronology
from feedwatch.temporal_heads import EarlyStopping, TemporalHead, evaluate_head, load_temporal_checkpoint
from feedwatch.visual import (cache_signature, decode_visual, encode_frames, load_embedding_cache,
                              load_extractor, save_embedding_cache, state_hash)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/experiments/diverse-v3"
DATA = ROOT/"artifacts/research-development-v3"


def configuration():
    cfg = json.loads((ROOT/"config/diverse_v3.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = OUT/"protocol.json"
    if sealed.exists() and json.loads(sealed.read_text(encoding="utf-8"))["configuration"] != cfg:
        raise ValueError("Do not modify the preregistered experiment after it starts")
    if not sealed.exists():
        write_json(sealed, {"sealed_at_utc": utc_now(), "configuration": cfg,
                            "training_started": False, "heldout_opened": False})
    return cfg


def select_members(cfg):
    target = DATA/"selection.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        if result["protocol_sha256"] != file_hash(OUT/"protocol.json"):
            raise ValueError("Selection belongs to another protocol")
        return result
    remote = RemoteZip(cfg["source"]["url"], cfg["source"]["size"])
    allowed = {day: split for split, days in cfg["new_days"].items() for day in days}
    strata = defaultdict(list)
    with zipfile.ZipFile(remote) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            position = next((i for i, part in enumerate(parts) if part in allowed), None)
            if position is None: continue
            relative = PurePosixPath(*parts[position:]).as_posix()
            try: row = parse_relative_path(relative)
            except ValueError: continue
            strata[row["date"], row["group"], row["class_name"]].append((row, member))
    selected = []
    count = cfg["members_per_day_session_class"]
    for key, entries in sorted(strata.items()):
        if len(entries) < count: raise ValueError(f"Incomplete stratum: {key}")
        entries.sort(key=lambda item: hashlib.sha256(
            f'{cfg["selection_seed"]}:{item[0]["relative_path"]}'.encode()).hexdigest())
        for row, member in entries[:count]:
            selected.append({**row, "split": allowed[row["date"]], "zip_member": member.filename,
                "header_offset": member.header_offset, "flag_bits": member.flag_bits,
                "compression": member.compress_type, "crc32": member.CRC,
                "size": member.file_size, "compressed_size": member.compress_size})
    frame = pd.DataFrame(selected)
    validate_development_days(frame, cfg["new_days"])
    total = int(frame["size"].sum())
    if total > cfg["maximum_selected_bytes"]: raise ValueError("Sealed selection exceeds transfer budget")
    result = {"protocol_sha256": file_hash(OUT/"protocol.json"), "selected_at_utc": utc_now(),
              "source": cfg["source"], "selection_seed": cfg["selection_seed"],
              "selected_count": len(selected), "selected_bytes": total, "strata": len(strata),
              "by_split": dict(Counter(frame.split)), "by_day": dict(Counter(frame.date)),
              "full_archive_md5_verified": False, "members": selected}
    write_json(target, result)
    print(json.dumps({key: result[key] for key in ("selected_count", "selected_bytes", "strata", "by_split")}), flush=True)
    return result


def acquire(cfg, selection, splits):
    if "heldout" in splits:
        gate = json.loads((OUT/"validation-results.json").read_text(encoding="utf-8"))
        if not gate["comparison"]["eligible_for_heldout"]:
            raise ValueError("Heldout acquisition is forbidden after a failed validation gate")
    entries = [row for row in selection["members"] if row["split"] in splits]
    started, measurements, failures = time.perf_counter(), {}, []
    def download(row):
        member = zipfile.ZipInfo(row["zip_member"])
        member.header_offset, member.flag_bits = row["header_offset"], row["flag_bits"]
        member.compress_type, member.CRC = row["compression"], row["crc32"]
        member.file_size, member.compress_size = row["size"], row["compressed_size"]
        return row["relative_path"], verified_member(cfg["source"]["url"], cfg["source"]["size"],
                                                   member, DATA/"videos"/row["relative_path"])
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as executor:
        futures = [executor.submit(download, row) for row in entries]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                relative, result = future.result()
                measurements[relative] = result
            except Exception as error:
                failure = {"error_type": type(error).__name__, "error": str(error)}
                failures.append(failure)
                write_json(DATA/"acquisition-failures.json", failures)
                print(json.dumps({"stage": "acquisition_member_failed", **failure}), flush=True)
            if completed % 20 == 0 or completed == len(entries):
                elapsed = time.perf_counter()-started
                progress = {"stage": "acquisition", "splits": splits, "completed": completed,
                    "requested": len(entries), "failed": len(failures), "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed/completed*(len(entries)-completed)}
                write_json(DATA/"progress.json", progress)
                print(json.dumps(progress), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} members failed; rerun the same acquisition to verify/resume, never substitute examples")
    result = {"status": "complete", "splits": splits, "selection_sha256": file_hash(DATA/"selection.json"),
              "members": measurements, "all_crc_verified": True, "full_archive_md5_verified": False,
              "transferred_bytes_this_run": sum(row["transferred_bytes"] for row in measurements.values()),
              "seconds": time.perf_counter()-started}
    write_json(DATA/("acquisition-heldout.json" if "heldout" in splits else "acquisition-development.json"), result)


def setup(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def execution_seal():
    sources = ["scripts/train_diverse_candidate.py"] + [f"src/feedwatch/{name}.py" for name in (
        "diverse_training", "research_acquisition", "experiment_integrity", "temporal_heads",
        "temporal_data", "visual", "evaluation", "data")]
    inputs = {"protocol_sha256": file_hash(OUT/"protocol.json"),
              "selection_sha256": file_hash(DATA/"selection.json"),
              "temporal_config_sha256": file_hash(ROOT/"config/temporal.json"),
              "head_configuration": json.loads((ROOT/"config/temporal.json").read_text(encoding="utf-8"))["heads"],
              "source_sha256": {name: file_hash(ROOT/name) for name in sources},
              "baseline_sha256": {str(seed): file_hash(ROOT/f"artifacts/experiments/temporal-controlled/gru/seed_{seed}/best_head.pt")
                                  for seed in (42, 43, 44)}}
    return seal_inputs(OUT/"execution-seal.json", inputs)


def training_inputs(config_id):
    return {"extraction_config_id": config_id, "protocol_sha256": file_hash(OUT/"protocol.json"),
            "execution_seal_sha256": file_hash(OUT/"execution-seal.json"),
            "selection_sha256": file_hash(DATA/"selection.json"),
            "train_sha256": file_hash(OUT/"expanded_train.csv"),
            "val_sha256": file_hash(OUT/"expanded_val.csv")}


def verify_development_results(cfg, result, device, seal):
    original = json.loads((ROOT/"artifacts/experiments/visual-full/config.json").read_text(encoding="utf-8"))
    expected = training_inputs(original["config_id"])
    if result["protocol_sha256"] != file_hash(OUT/"protocol.json") or result["training_inputs"] != expected:
        raise ValueError("Development results belong to different experiment inputs")
    if [run["seed"] for run in result["runs"]] != cfg["seeds"] or [run["seed"] for run in result["baseline"]] != cfg["seeds"]:
        raise ValueError("Development results have missing, duplicate or reordered seeds")
    for run in result["runs"]:
        folder = OUT/f"seed_{run['seed']}"
        if json.loads((folder/"results.json").read_text(encoding="utf-8")) != run:
            raise ValueError("Saved seed result changed after the validation decision")
        _, bundle = load_temporal_checkpoint(folder/"best_head.pt", device)
        verify_checkpoint_run(folder/"best_head.pt", run, bundle, expected, run["seed"],
                              seal["inputs"]["head_configuration"], cfg)
    for reference in result["baseline"]:
        seed = reference["seed"]
        if file_hash(ROOT/f"artifacts/experiments/temporal-controlled/gru/seed_{seed}/best_head.pt") != reference["checkpoint_sha256"]:
            raise ValueError("Baseline changed after validation")
    fresh = validation_gate([run["by_day"] for run in result["runs"]],
                            [run["by_day"] for run in result["baseline"]], cfg["validation_gate"])
    if fresh != result["comparison"]:
        raise ValueError("Saved validation gate does not match its paired measurements")


def new_sequences(selection, splits, original, device):
    identity, config_id = original["extraction_identity"], original["config_id"]
    entries = [row for row in selection["members"] if row["split"] in splits]
    arrays, model, initial, started = [], None, None, time.perf_counter()
    acquisition_path = DATA/("acquisition-heldout.json" if "heldout" in splits else "acquisition-development.json")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8")) if acquisition_path.exists() else {"members": {}}
    for index, row in enumerate(entries, 1):
        path = DATA/"videos"/row["relative_path"]
        deadline, reported = time.monotonic()+1800, time.monotonic()
        while not path.exists():
            if time.monotonic() > deadline: raise TimeoutError("Acquisition did not supply the sealed member")
            if time.monotonic()-reported >= 30:
                print(f"Waiting for acquisition; new sequences completed {index-1}/{len(entries)}", flush=True)
                reported = time.monotonic()
            time.sleep(2)
        # The downloader writes atomically. Size/CRC also permits extraction while
        # acquisition is in progress, before its final SHA256 ledger is written.
        import zlib
        payload = path.read_bytes()
        if len(payload) != row["size"] or zlib.crc32(payload) != row["crc32"]:
            raise ValueError("Acquired video differs from the sealed member")
        if row["relative_path"] in acquisition["members"]:
            if hashlib.sha256(payload).hexdigest() != acquisition["members"][row["relative_path"]]["sha256"]:
                raise ValueError("Acquired video changed before extraction")
        signature = cache_signature(DATA/"videos", row["relative_path"], config_id)
        entry, _ = load_embedding_cache(DATA/"visual-cache", signature, 16)
        if entry is None:
            if model is None:
                if device.type != identity["device_type"]:
                    raise ValueError("New embeddings must use the device type recorded in the extraction identity")
                model, transform, _ = load_extractor(Path(identity["weights"]["weights_file"]).parent, device)
                initial = state_hash(model)
            decoded = decode_visual(DATA/"videos", row["relative_path"], identity["parameters"], transform)
            if not decoded["ok"]: raise ValueError(f'{row["relative_path"]}: {decoded["error"]}')
            validate_chronology(decoded["details"], identity["parameters"])
            embeddings = encode_frames(model, decoded["frames"], device)
            save_embedding_cache(DATA/"visual-cache", signature, embeddings, decoded["details"], decoded["seconds"])
            entry = {"embeddings": embeddings, "details": decoded["details"]}
        validate_chronology(entry["details"], identity["parameters"])
        arrays.append(entry["embeddings"])
        if index % 50 == 0 or index == len(entries):
            print(f"New-day sequences verified {index}/{len(entries)} ({time.perf_counter()-started:.1f}s)", flush=True)
    if model is not None and state_hash(model) != initial: raise ValueError("Extractor changed")
    return np.stack(arrays).astype(np.float32), pd.DataFrame(entries)[COLUMNS+["split"]]


def baseline_predictions(rows, array, device, seeds, split):
    results = []
    sealed = execution_seal()["inputs"]["baseline_sha256"]
    for seed in seeds:
        path = ROOT/f"artifacts/experiments/temporal-controlled/gru/seed_{seed}/best_head.pt"
        if file_hash(path) != sealed[str(seed)]: raise ValueError("Sealed baseline checkpoint changed")
        model, bundle = load_temporal_checkpoint(path, device)
        x = torch.from_numpy((array-bundle["mean"].numpy())/bundle["std"].numpy()).to(device)
        probabilities, _, _ = evaluate_head(model, x, rows.class_id.to_numpy(), 256)
        metrics = save_evaluation(rows, probabilities, OUT/f"baseline_seed_{seed}"/split, split)
        result = {"seed": seed, "checkpoint_sha256": file_hash(path), "metrics": metrics,
                  "by_day": day_metrics(rows, probabilities)}
        write_json(OUT/f"baseline_seed_{seed}"/f"{split}-results.json", result)
        results.append(result)
    return results


def train(cfg, selection, device):
    seal = execution_seal()
    if (OUT/"validation-results.json").exists():
        result = json.loads((OUT/"validation-results.json").read_text(encoding="utf-8"))
        verify_development_results(cfg, result, device, seal)
        print("Completed experiment retained; no adaptive rerun.", flush=True)
        return
    array, rows, audit, original = load_sequences(ROOT/"video_6.19-6.22", ROOT/"artifacts/manifest.csv",
        ROOT/"artifacts/splits", ROOT/"artifacts/experiments/visual-full", ROOT/"artifacts/visual-cache",
        OUT/"repair-cache", OUT, None, 42, device)
    added, added_rows = new_sequences(selection, ["train", "val"], original, device)
    array, rows = np.concatenate([array, added]), pd.concat([rows, added_rows], ignore_index=True)
    assignment = {name: cfg["existing_days"][name]+cfg["new_days"][name] for name in ("train", "val")}
    validate_development_days(rows, assignment)
    for name in assignment:
        rows.loc[rows.split.eq(name), COLUMNS].to_csv(OUT/f"expanded_{name}.csv", index=False)
    indices = {name: np.flatnonzero(rows.split.eq(name)) for name in assignment}
    training_means = array[indices["train"]].mean(1)
    mean, std = training_means.mean(0), np.maximum(training_means.std(0), 1e-6)
    x = {name: torch.from_numpy((array[idx]-mean)/std).to(device) for name, idx in indices.items()}
    y = {name: rows.iloc[idx].class_id.to_numpy() for name, idx in indices.items()}
    metadata = {name: rows.iloc[idx].reset_index(drop=True) for name, idx in indices.items()}
    baseline = baseline_predictions(metadata["val"], array[indices["val"]], device, cfg["seeds"], "validation")
    weights = torch.from_numpy(session_weights(metadata["train"])).to(device)
    targets = torch.from_numpy(y["train"]).long().to(device)
    heads = seal["inputs"]["head_configuration"]
    provenance = training_inputs(audit["config_id"])
    write_json(OUT/"input-audit.json", {"original_cache": audit, "new_selection_sha256": file_hash(DATA/"selection.json"),
        "train_sha256": file_hash(OUT/"expanded_train.csv"), "val_sha256": file_hash(OUT/"expanded_val.csv"),
        "assignment": assignment, "by_split": {name: len(idx) for name, idx in indices.items()},
        "heldout_opened": False, "excluded_days_opened": False, "normalization_uses_training_only": True})
    runs = []
    for seed in cfg["seeds"]:
        folder = OUT/f"seed_{seed}"
        if (folder/"results.json").exists():
            result = json.loads((folder/"results.json").read_text(encoding="utf-8"))
            restored, bundle = load_temporal_checkpoint(folder/"best_head.pt", device)
            verify_checkpoint_run(folder/"best_head.pt", result, bundle, provenance, seed, heads, cfg)
            replay, _, _ = evaluate_head(restored, x["val"], y["val"], cfg["evaluation_batch_size"])
            if day_metrics(metadata["val"], replay) != result["by_day"]:
                raise ValueError("Saved seed metrics do not replay from the checkpoint")
            runs.append(result)
            continue
        folder.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        model = TemporalHead("gru", heads).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        stopping, history, started = EarlyStopping(cfg["patience"]), [], time.perf_counter()
        for epoch in range(1, cfg["max_epochs"]+1):
            model.train()
            order = torch.multinomial(weights, len(targets), replacement=True, generator=generator)
            for batch in order.split(cfg["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x["train"][batch]), targets[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            vp, vm, vl = evaluate_head(model, x["val"], y["val"], cfg["evaluation_batch_size"])
            _, tm, tl = evaluate_head(model, x["train"], y["train"], cfg["evaluation_batch_size"])
            days = day_metrics(metadata["val"], vp)
            day_mean = aggregate_days([days])
            improved, stop = stopping.step(day_mean["macro_f1"])
            history.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"],
                "train_macro_f1": tm["macro_f1"], "train_accuracy": tm["accuracy"], "train_loss": tl,
                "val_mean_day_macro_f1": day_mean["macro_f1"], "val_mean_day_medium_recall": day_mean["recalls"]["medium"],
                "val_loss": vl, **{f"{day}_macro_f1": value["macro_f1"] for day, value in days.items()}})
            if improved:
                torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "kind": "gru", "seed": seed, "head_config": heads, "training_config": cfg,
                    "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "class_mapping": CLASS_IDS,
                    "selected_epoch": epoch, "validation_macro_f1": vm["macro_f1"],
                    "validation_mean_day_macro_f1": day_mean["macro_f1"],
                    "provenance": provenance,
                    "extractor_trained": False, "test_evaluated": False}, folder/"best_head.pt")
            pd.DataFrame(history).to_csv(folder/"history.csv", index=False)
            if epoch == 1 or epoch % 5 == 0 or stop:
                print(f"diverse seed={seed} epoch={epoch} day_f1={day_mean['macro_f1']:.4f} medium={day_mean['recalls']['medium']:.4f}", flush=True)
            if stop: break
        model, bundle = load_temporal_checkpoint(folder/"best_head.pt", device)
        vp, _, _ = evaluate_head(model, x["val"], y["val"], cfg["evaluation_batch_size"])
        validation = save_evaluation(metadata["val"], vp, folder/"validation", "development_validation_days_17_21")
        by_day = day_metrics(metadata["val"], vp)
        if abs(aggregate_days([by_day])["macro_f1"]-bundle["validation_mean_day_macro_f1"]) > 1e-12:
            raise ValueError("Selected checkpoint does not replay")
        restored, _ = load_temporal_checkpoint(folder/"best_head.pt", device)
        replay, _, _ = evaluate_head(restored, x["val"], y["val"], cfg["evaluation_batch_size"])
        np.testing.assert_allclose(vp, replay, atol=1e-7)
        _, training, _ = evaluate_head(model, x["train"], y["train"], cfg["evaluation_batch_size"])
        result = {"seed": seed, "selected_epoch": bundle["selected_epoch"], "epochs_run": len(history),
                  "seconds": time.perf_counter()-started, "checkpoint_sha256": file_hash(folder/"best_head.pt"),
                  "validation": validation, "by_day": by_day, "training": training, "checkpoint_replay_verified": True,
                  "training_inputs": provenance}
        write_json(folder/"results.json", result)
        runs.append(result)
    comparison = validation_gate([run["by_day"] for run in runs], [run["by_day"] for run in baseline], cfg["validation_gate"])
    result = {"protocol_sha256": file_hash(OUT/"protocol.json"), "training_inputs": provenance, "completed_at_utc": utc_now(),
              "runs": runs, "baseline": baseline, "comparison": comparison, "operational_model_replaced": False,
              "heldout_evaluated": False, "excluded_days_opened": False,
              "limitations": ["Same research location and species; new days are not new farms.",
                              "Only four clips per new day/session/class; nearby clips remain correlated."]}
    write_json(OUT/"validation-results.json", result)
    print(json.dumps({"validation_gate": comparison["checks"], "eligible_for_heldout": comparison["eligible_for_heldout"]}), flush=True)


def heldout(cfg, selection, device):
    seal = execution_seal()
    if (OUT/"heldout-results.json").exists():
        result = json.loads((OUT/"heldout-results.json").read_text(encoding="utf-8"))
        if result["heldout_frozen_sha256"] != file_hash(OUT/"heldout-frozen.json"):
            raise ValueError("Final examination seal changed")
        frozen = json.loads((OUT/"heldout-frozen.json").read_text(encoding="utf-8"))["inputs"]
        if (frozen["candidate_sha256"] != file_hash(OUT/"seed_42/best_head.pt")
                or frozen["validation_sha256"] != file_hash(OUT/"validation-results.json")
                or frozen["selection_sha256"] != file_hash(DATA/"selection.json")
                or frozen["execution_seal_sha256"] != file_hash(OUT/"execution-seal.json")
                or frozen["baseline_sha256"] != seal["inputs"]["baseline_sha256"]):
            raise ValueError("Final examination inputs changed")
        print("One final examination already retained; no repeated tuning.")
        return
    development = json.loads((OUT/"validation-results.json").read_text(encoding="utf-8"))
    verify_development_results(cfg, development, device, seal)
    if not development["comparison"]["eligible_for_heldout"]:
        raise ValueError("Validation failed; do not open the heldout day")
    paths = {seed: OUT/f"seed_{seed}/best_head.pt" for seed in cfg["seeds"]}
    for run in development["runs"]:
        if file_hash(paths[run["seed"]]) != run["checkpoint_sha256"]: raise ValueError("Candidate changed after validation")
    frozen = {"candidate_sha256": file_hash(paths[42]), "validation_sha256": file_hash(OUT/"validation-results.json"),
              "selection_sha256": file_hash(DATA/"selection.json"), "execution_seal_sha256": file_hash(OUT/"execution-seal.json"),
              "baseline_sha256": seal["inputs"]["baseline_sha256"]}
    seal_inputs(OUT/"heldout-frozen.json", frozen)
    acquire(cfg, selection, ["heldout"])
    original = json.loads((ROOT/"artifacts/experiments/visual-full/config.json").read_text(encoding="utf-8"))
    array, rows = new_sequences(selection, ["heldout"], original, device)
    baseline = baseline_predictions(rows, array, device, [42], "heldout_day_18")[0]
    model, bundle = load_temporal_checkpoint(paths[42], device)
    x = torch.from_numpy((array-bundle["mean"].numpy())/bundle["std"].numpy()).to(device)
    probability, _, _ = evaluate_head(model, x, rows.class_id.to_numpy(), 256)
    candidate = save_evaluation(rows, probability, OUT/"seed_42/heldout", "heldout_day_18")
    old = baseline["metrics"]
    old_predictions = pd.read_csv(OUT/"baseline_seed_42/heldout_day_18/predictions.csv").predicted_id.to_numpy()
    rules = cfg["promotion"]
    bootstrap = session_bootstrap_delta(rows, old_predictions, probability.argmax(1), rules["bootstrap_seed"], rules["bootstrap_replicates"])
    recalls = {name: candidate["classification_report"][name]["recall"]-old["classification_report"][name]["recall"] for name in CLASS_IDS}
    checks = {"macro_f1_gain": candidate["macro_f1"]-old["macro_f1"] >= rules["macro_f1_gain"],
              "medium_recall_gain": recalls["medium"] >= rules["medium_recall_gain"],
              "other_recalls_preserved": all(value >= -rules["max_other_recall_drop"] for name, value in recalls.items() if name != "medium"),
              "paired_session_bootstrap": bootstrap["lower_95"] >= rules["paired_session_bootstrap_lower_accuracy_delta"]}
    result = {"completed_at_utc": utc_now(), "protocol_sha256": file_hash(OUT/"protocol.json"),
              "heldout_frozen_sha256": file_hash(OUT/"heldout-frozen.json"),
              "candidate": candidate, "baseline": old, "bootstrap": bootstrap, "promotion_checks": checks,
              "eligible_for_promotion": all(checks.values()), "candidate_sha256": file_hash(paths[42]),
              "operational_model_replaced": False, "excluded_days_opened": False,
              "scope": "Held-out recording day in the same AV-FFIA research setting; no external-farm validation."}
    write_json(OUT/"heldout-results.json", result)
    print(json.dumps({"heldout_checks": checks, "eligible_for_promotion": all(checks.values())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "acquire", "train", "heldout"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = configuration()
    selection = select_members(cfg)
    if args.stage == "acquire": acquire(cfg, selection, ["train", "val"])
    elif args.stage == "train": train(cfg, selection, setup(args.device))
    elif args.stage == "heldout": heldout(cfg, selection, setup(args.device))


if __name__ == "__main__": main()
