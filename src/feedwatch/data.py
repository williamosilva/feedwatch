"""Manifest e separação por dia, sem abrir ou modificar os vídeos."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath

import pandas as pd

CLASS_IDS = json.loads(files("feedwatch").joinpath("classes.json").read_text(encoding="utf-8"))
COLUMNS = ["relative_path", "date", "group", "class_name", "class_id"]
SPLITS = ("train", "val", "test")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_root(path: Path) -> Path:
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset não encontrado: {root}. Informe --data-dir apontando para video_6.19-6.22.")
    return root


def protect_originals(root: Path, output: Path) -> None:
    target = output.resolve()
    if target == root or root in target.parents:
        raise ValueError(f"A saída deve ficar fora do dataset original: {target}")


def video_path(root: Path, relative_path: str) -> Path:
    # Também impede que links simbólicos escapem da raiz após a indexação.
    parse_relative_path(relative_path)
    path = root.joinpath(*PurePosixPath(relative_path).parts).resolve()
    if root not in path.parents:
        raise ValueError(f"Caminho fora do dataset: {relative_path}")
    return path


def parse_relative_path(relative_path: str) -> dict:
    if "\\" in relative_path or ":" in relative_path:
        raise ValueError("Use caminho relativo POSIX, sem unidade ou barras invertidas")
    parts = relative_path.split("/")
    if len(parts) != 4 or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Estrutura esperada: data/grupo/classe/arquivo.mp4")
    day, group, label, filename = parts
    match = re.fullmatch(r"(\d{4})_(\d{1,2})_(\d{1,2})", day)
    if not match:
        raise ValueError(f"Data inesperada: {day}")
    date(*map(int, match.groups()))
    if not re.fullmatch(r"(?:AM|PM)_\d+", group):
        raise ValueError(f"Grupo inesperado: {group}; números são identificadores opacos")
    if label not in CLASS_IDS:
        raise ValueError(f"Classe desconhecida: {label}")
    if Path(filename).suffix.lower() != ".mp4":
        raise ValueError(f"Extensão inesperada: {filename}")
    return dict(zip(COLUMNS, [relative_path, day, group, label, CLASS_IDS[label]]))


def index_dataset(root: Path, output: Path) -> tuple[pd.DataFrame, list[dict]]:
    root = data_root(root)
    protect_originals(root, output)
    output.mkdir(parents=True, exist_ok=True)
    records, issues, seen = [], [], set()

    def issue(path: str, reason: str) -> None:
        issues.append({"relative_path": path, "reason": reason})

    def walk_error(error: OSError) -> None:
        issue(str(error.filename), f"Erro de leitura: {error}")

    for parent, directories, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        directories.sort()
        for name in directories[:]:
            directory = Path(parent) / name
            if directory.is_symlink() or directory.is_junction():
                issue(directory.relative_to(root).as_posix(), "Link de diretório não percorrido")
                directories.remove(name)
        if not directories and not filenames:
            issue(Path(parent).relative_to(root).as_posix(), "Diretório vazio")
        for name in sorted(filenames):
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            try:
                if path.is_symlink():
                    raise ValueError("Link de arquivo não indexado")
                record = parse_relative_path(relative)
                if relative.casefold() in seen:
                    raise ValueError("Caminho duplicado, considerando maiúsculas/minúsculas")
                seen.add(relative.casefold())
                records.append(record)
            except ValueError as error:
                issue(relative, str(error))
    frame = pd.DataFrame(records, columns=COLUMNS).sort_values("relative_path").reset_index(drop=True)
    frame.to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(issues, columns=["relative_path", "reason"]).to_csv(output / "index_issues.csv", index=False)
    write_json(output / "classes.json", CLASS_IDS)
    write_json(output / "index_report.json", {
        "created_at_utc": utc_now(), "data_dir": str(root), "videos_indexed": len(frame),
        "issues_count": len(issues), "issues": issues,
        "manifest_sha256": file_hash(output / "manifest.csv"),
        "label_source": "class directory", "class_mapping": CLASS_IDS,
    })
    return frame, issues


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={column: str for column in COLUMNS[:-1]}, keep_default_na=False)
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError(f"Manifest vazio ou colunas inválidas: {path}")
    if frame.relative_path.str.casefold().duplicated().any():
        raise ValueError(f"Caminhos duplicados no manifest: {path}")
    for row in frame.to_dict("records"):
        expected = parse_relative_path(row["relative_path"])
        if row != expected:
            raise ValueError(f"Metadados divergentes do caminho: {row['relative_path']}")
    return frame


def validate_splits(manifest: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> dict:
    errors = []
    if set(parts) != set(SPLITS):
        errors.append("São necessárias exatamente as divisões train, val e test")
    all_paths, all_dates, composition = set(), set(), {}
    original = manifest.set_index("relative_path")
    for name, frame in parts.items():
        paths, days = set(frame.relative_path), set(frame.date)
        if len(paths) != len(frame) or frame.relative_path.str.casefold().duplicated().any():
            errors.append(f"{name}: caminhos duplicados")
        if all_paths & paths:
            errors.append(f"{name}: sobreposição de caminhos")
        if all_dates & days:
            errors.append(f"{name}: sobreposição de datas")
        if frame.empty:
            errors.append(f"{name}: divisão vazia")
        missing = sorted(set(CLASS_IDS) - set(frame.class_name))
        if missing:
            errors.append(f"{name}: classes ausentes: {missing}")
        if not paths <= set(original.index):
            errors.append(f"{name}: caminhos não pertencem ao manifest")
        elif paths and not frame.set_index("relative_path").sort_index().equals(original.loc[sorted(paths)].sort_index()):
            errors.append(f"{name}: metadados diferem do manifest")
        composition[name] = {"dates": sorted(days), "videos": len(frame),
                             "classes": {label: int((frame.class_name == label).sum()) for label in CLASS_IDS}}
        all_paths.update(paths)
        all_dates.update(days)
    if all_paths != set(manifest.relative_path):
        errors.append("Divisões não cobrem exatamente os caminhos do manifest")
    return {"valid": not errors, "errors": errors, "composition": composition,
            "scope": "Generalização entre dias disponíveis; não comprova generalização para outras fazendas."}


def make_splits(manifest_path: Path, config_path: Path, output: Path) -> dict:
    manifest = read_manifest(manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if set(config) != set(SPLITS) or any(not isinstance(v, list) or not v or any(not isinstance(d, str) for d in v) for v in config.values()):
        raise ValueError("Configuração precisa de train/val/test com listas não vazias de datas")
    days = [day for name in SPLITS for day in config[name]]
    if len(days) != len(set(days)) or set(days) != set(manifest.date):
        raise ValueError("Cada data observada deve aparecer exatamente uma vez na configuração")
    parts = {name: manifest[manifest.date.isin(config[name])].copy() for name in SPLITS}
    report = validate_splits(manifest, parts)
    report.update({"created_at_utc": utc_now(), "manifest_sha256": file_hash(manifest_path), "assignment": config})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation.json", report)
    if not report["valid"]:
        raise ValueError("Divisão inválida: " + "; ".join(report["errors"]))
    for name, frame in parts.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "assignment.json", config)
    return report


def stratified_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Uma escolha por estrato data/grupo/classe antes de repetir estratos."""
    import random

    if count < 1:
        raise ValueError("Tamanho da amostra deve ser positivo")
    rng = random.Random(seed)
    buckets = []
    for _, group in frame.sort_values("relative_path").groupby(["date", "group", "class_name"], sort=True):
        paths = list(group.relative_path)
        rng.shuffle(paths)
        buckets.append(paths)
    rng.shuffle(buckets)
    selected = []
    while buckets and len(selected) < min(count, len(frame)):
        remaining = []
        for bucket in buckets:
            if len(selected) == min(count, len(frame)):
                break
            selected.append(bucket.pop())
            if bucket:
                remaining.append(bucket)
        buckets = remaining
    return frame.set_index("relative_path").loc[selected].reset_index()
