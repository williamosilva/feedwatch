"""Resumos derivados de manifests e verificações já executadas."""

import json
from pathlib import Path

import pandas as pd

from .data import CLASS_IDS, SPLITS, file_hash, read_manifest, validate_splits, write_json


def summarize(manifest_path: Path, audit_dir: Path, splits_dir: Path, output: Path) -> dict:
    manifest = read_manifest(manifest_path)
    audit = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))
    saved_validation = json.loads((splits_dir / "validation.json").read_text(encoding="utf-8"))
    digest = file_hash(manifest_path)
    if audit["manifest_sha256"] != digest or saved_validation["manifest_sha256"] != digest:
        raise ValueError("Manifest mudou: execute novamente audit e split antes de gerar o resumo")
    parts = {name: read_manifest(splits_dir / f"{name}.csv") for name in SPLITS}
    validation = validate_splits(manifest, parts)
    if not validation["valid"]:
        raise ValueError("Divisões inválidas: " + "; ".join(validation["errors"]))
    output.mkdir(parents=True, exist_ok=True)
    by_class = manifest.groupby(["class_id", "class_name"]).size().reset_index(name="videos")
    by_date = manifest.groupby("date").size().reset_index(name="videos")
    by_group = manifest.groupby("group").size().reset_index(name="videos")
    by_stratum = manifest.groupby(["date", "group", "class_name"]).size().reset_index(name="videos")
    combined = pd.concat([frame.assign(split=name) for name, frame in parts.items()], ignore_index=True)
    by_split = combined.groupby("split").size().reindex(SPLITS).reset_index(name="videos")
    by_split_class = combined.groupby(["split", "class_id", "class_name"]).size().reset_index(name="videos")
    for name, table in [("by_class", by_class), ("by_date", by_date), ("by_group", by_group),
                        ("by_date_group_class", by_stratum), ("by_split", by_split), ("by_split_class", by_split_class)]:
        table.to_csv(output / f"{name}.csv", index=False)
    summary = {"manifest_sha256": digest, "videos": len(manifest), "classes": CLASS_IDS,
               "by_class": by_class.to_dict("records"), "by_date": by_date.to_dict("records"),
               "by_group": by_group.to_dict("records"), "splits": validation, "audit": audit}
    write_json(output / "summary.json", summary)
    lines = ["# FeedWatch — preparação observada", "", f"Vídeos indexados: {len(manifest)}.", "",
             "Rótulos extraídos das pastas do dataset; nenhum modelo foi treinado.", "",
             "## Classes", "", "| Classe | ID | Vídeos |", "|---|---:|---:|"]
    lines += [f"| {row.class_name} | {row.class_id} | {row.videos} |" for row in by_class.itertuples()]
    lines += ["", "## Divisões por data", "", "| Divisão | Datas | Vídeos |", "|---|---|---:|"]
    lines += [f"| {name} | {', '.join(info['dates'])} | {info['videos']} |" for name, info in validation["composition"].items()]
    lines += ["", "Sobreposição de caminhos/datas: nenhuma. Quatro classes presentes em cada divisão.",
              "Este teste avalia generalização entre os dias disponíveis, sem comprovar generalização para outras fazendas.",
              "Use validação para escolher hiperparâmetros; preserve o teste para a avaliação final.", "",
              "## Auditoria", "", f"Modo: {audit['mode']}; seed: {audit['seed']}.",
              f"Auditados: {audit['audited_videos']}/{audit['total_videos']}; aprovados: {audit['passed']}; falhas: {audit['failed']}.",
              f"Frames decodificados: {audit['decoded_frames']}.",
              f"Estratos cobertos: {audit['sample_coverage']['strata_date_group_class']}/{audit['manifest_coverage']['strata_date_group_class']}.",
              "Duração estimada por quantidade de frames / FPS. A amostra não certifica os demais arquivos nem todos os frames.", "",
              "Consulte os CSVs para contagens por data, grupo, combinação e divisão; failures.csv contém os problemas individuais.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
