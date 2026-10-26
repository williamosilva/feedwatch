"""Análise posterior sem ajustar novamente a floresta ou alterar seus artefatos."""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_IDS, COLUMNS, data_root, file_hash, protect_originals, video_path, write_json
from .evaluation import NAMES, check_metadata, save_evaluation
from .experiment import feature_matrix, load_train_val


def review_motion(root: Path, manifest: Path, splits: Path, baseline: Path, output: Path, seed: int = 42) -> dict:
    root = data_root(root)
    protect_originals(root, output)
    if output.resolve() == baseline.resolve() or baseline.resolve() in output.resolve().parents:
        raise ValueError("A revisão deve ficar fora da pasta da referência original")
    parts, provenance = load_train_val(manifest, splits)
    features = pd.read_csv(baseline / "features.csv")
    with (baseline / "random_forest" / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle["class_mapping"] != CLASS_IDS or bundle["feature_names"] != list(feature_matrix(features).columns):
        raise ValueError("Mapeamento/esquema do modelo diverge do projeto")
    if bundle["used_train_sha256"] != file_hash(baseline / "used_train.csv"):
        raise ValueError("Lista de treino difere da registrada no modelo")
    if bundle["estimator"].classes_.tolist() != list(range(4)):
        raise ValueError("Ordem dos IDs do estimador inválida")
    for name in ("train", "val"):
        current = features[features.split == name][COLUMNS].sort_values("relative_path").reset_index(drop=True)
        recorded = pd.read_csv(baseline / f"used_{name}.csv").sort_values("relative_path").reset_index(drop=True)
        check_metadata(current)
        expected = parts[name][parts[name].relative_path.isin(recorded.relative_path)].sort_values("relative_path").reset_index(drop=True)
        if not current.equals(recorded) or not current.equals(expected):
            raise ValueError(f"Metadados divergentes em {name}")
    val = features[features.split == "val"].sort_values("relative_path").reset_index(drop=True)
    stored = pd.read_csv(baseline / "random_forest" / "predictions.csv").sort_values("relative_path").reset_index(drop=True)
    if not val[COLUMNS].equals(stored[COLUMNS]):
        raise ValueError("Previsões e características não identificam os mesmos vídeos")
    if any(CLASS_IDS.get(label) != ident for label, ident in zip(stored.predicted_class, stored.predicted_id)):
        raise ValueError("Classe prevista diverge do ID previsto")
    if not np.array_equal(stored.correct.to_numpy(), stored.class_id.to_numpy() == stored.predicted_id.to_numpy()):
        raise ValueError("Coluna correct inconsistente")
    probabilities = bundle["estimator"].predict_proba(feature_matrix(val))
    np.testing.assert_array_equal(probabilities.argmax(1), stored.predicted_id)
    np.testing.assert_allclose(probabilities, stored[[f"prob_{name}" for name in NAMES]], atol=1e-12)
    train = features[features.split == "train"].sort_values("relative_path").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    train_metrics = save_evaluation(train, bundle["estimator"].predict_proba(feature_matrix(train)), output / "train", "train")
    val_metrics = save_evaluation(val, probabilities, output / "val_verified", "val")
    rng, examples = random.Random(seed), []
    for real, predicted in [("medium", "strong"), ("weak", "medium"), ("strong", "none")]:
        candidates = stored[(stored.class_name == real) & (stored.predicted_class == predicted)].to_dict("records")
        for row in rng.sample(candidates, min(3, len(candidates))):
            path = video_path(root, row["relative_path"])
            if not path.is_file():
                raise ValueError(f"Clipe de exemplo não encontrado: {path}")
            examples.append({"example_id": len(examples) + 1, **row, "clip_path": str(path), "clip_uri": path.as_uri()})
    pd.DataFrame(examples).to_csv(output / "error_examples.csv", index=False)
    (output / "error_examples.m3u8").write_text("#EXTM3U\n" + "\n".join(
        f"#EXTINF:-1,{row['class_name']} -> {row['predicted_class']} | {row['relative_path']}\n{row['clip_path']}" for row in examples) + "\n", encoding="utf-8")
    result = {"consistency_verified": True, "model_refitted": False, "seed": seed,
              "train_resubstitution": train_metrics, "validation": val_metrics,
              "provenance": provenance, "test_evaluated": False, "examples": examples}
    write_json(output / "results.json", result)
    lines = ["# Revisão da referência de movimento", "",
             "Consistência de caminho, rótulo, ID, previsão e probabilidades verificada contra o modelo persistido. Nenhum novo ajuste da floresta.", "",
             f"Treino: acurácia {train_metrics['accuracy']:.6f}; macro-F1 {train_metrics['macro_f1']:.6f}. **Reutilização dos próprios dados de ajuste; não mede generalização.**", "",
             f"Validação: acurácia {val_metrics['accuracy']:.6f}; macro-F1 {val_metrics['macro_f1']:.6f}.", "",
             "A diferença treino/validação é observada; não determina isoladamente se a causa é capacidade do modelo, mudança de dia ou limitação das características.", "",
             "## Exemplos reproduzíveis", "", "Seleção: três exemplos por par, caminhos ordenados e random.Random(42).", "",
             "| ID | Real (ID) | Prevista (ID) | Caminho / acesso ao original |", "|---|---|---|---|"]
    lines += [f"| {row['example_id']} | {row['class_name']} ({row['class_id']}) | {row['predicted_class']} ({row['predicted_id']}) | [{row['relative_path']}]({Path(row['clip_path']).as_posix()}) |" for row in examples]
    lines += ["", "`error_examples.csv` contém caminhos absolutos; `error_examples.m3u8` permite abrir a seleção num reprodutor compatível. Os clipes não foram copiados.", "",
              "PowerShell na raiz do projeto, para abrir o exemplo 1:", "", "```powershell",
              "$exemplos = Import-Csv '.\\artifacts\\analyses\\motion-review\\error_examples.csv'",
              "Invoke-Item ($exemplos | Where-Object example_id -eq '1').clip_path", "```", "",
              "Os rótulos são os do dataset, sem validação semântica independente. Não atribuímos causas visuais específicas a cada erro sem examinar os vídeos.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"RF treino (reutilização): accuracy={train_metrics['accuracy']:.6f}; macro_f1={train_metrics['macro_f1']:.6f}")
    print(f"Consistência verificada; {len(examples)} clipes de erro: {output / 'report.md'}")
    return result
