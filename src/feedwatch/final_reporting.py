"""Comparação descritiva do teste reservado, sem selecionar modelos pelos resultados."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, write_json
from .evaluation import NAMES, check_metadata, metrics, save_evaluation
from .temporal_reporting import display, mean_sd


def focus(values: dict) -> dict:
    matrix = np.asarray(values["confusion_matrix"])
    medium_n, strong_n = int(matrix[2].sum()), int(matrix[3].sum())
    return {"accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
            "medium_recall": values["classification_report"]["medium"]["recall"],
            "medium_to_strong": int(matrix[2, 3]), "medium_support": medium_n,
            "medium_to_strong_rate": float(matrix[2, 3]/medium_n) if medium_n else None,
            "strong_to_none": int(matrix[3, 0]), "strong_support": strong_n,
            "strong_to_none_rate": float(matrix[3, 0]/strong_n) if strong_n else None}


def summarize_values(values: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summarized = {key: mean_sd([focus(value)[key] for value in values]) for key in
                  ("accuracy", "macro_f1", "medium_recall", "medium_to_strong", "strong_to_none")}
    for key in ("medium_to_strong_rate", "strong_to_none_rate"):
        rates = [focus(value)[key] for value in values if focus(value)[key] is not None]
        summarized[key] = mean_sd(rates) if rates else {"mean": None, "std": None, "n": 0, "values": []}
    summarized["per_class"] = {name: {key: mean_sd([value["classification_report"][name][key] for value in values])
        for key in ("precision", "recall", "f1-score", "support")} for name in NAMES}
    matrices = np.asarray([value["confusion_matrix"] for value in values], dtype=float)
    pd.DataFrame(matrices.mean(0), index=pd.Index(NAMES, name="actual"), columns=NAMES).to_csv(output / "confusion_mean.csv")
    if len(values) > 1:
        pd.DataFrame(matrices.std(0, ddof=1), index=pd.Index(NAMES, name="actual"), columns=NAMES).to_csv(output / "confusion_std.csv")
    write_json(output / "statistics.json", summarized)
    return summarized


def report_final(output: Path, config: dict, requested: pd.DataFrame, runs: list[dict]) -> dict:
    tables = {run["spec"]["id"]: pd.read_csv(output / "models" / run["spec"]["id"] / "predictions.csv") for run in runs}
    common = sorted(set.intersection(*(set(table.relative_path) for table in tables.values())))
    if not common:
        raise ValueError("Nenhum vídeo comum entre os modelos; comparação não estimável")
    common_rows = requested.set_index("relative_path").loc[common].reset_index()[COLUMNS]
    common_rows.to_csv(output / "common_test_examples.csv", index=False)
    coverage, comparisons, class_rows, model_metrics = [], [], [], {}
    val_expected = pd.read_csv(Path(config["splits_dir"]) / "val.csv").sort_values("relative_path").reset_index(drop=True)
    for run in runs:
        spec, info = run["spec"], run["identification"]
        table = tables[spec["id"]]
        check_metadata(table)
        aligned = table.set_index("relative_path").loc[common].reset_index()
        if not aligned[COLUMNS].equals(common_rows):
            raise ValueError("Modelos não concordam nos metadados dos vídeos comuns")
        test_metrics = save_evaluation(common_rows, aligned[[f"prob_{name}" for name in NAMES]].to_numpy(),
                                       output / "common" / spec["id"], "test")
        val = pd.read_csv(spec["validation_predictions"]).sort_values("relative_path").reset_index(drop=True)
        check_metadata(val)
        if not val[COLUMNS].equals(val_expected):
            raise ValueError("Referência de validação não corresponde aos 1.705 vídeos originais")
        if any(NAMES[int(ident)] != name for ident, name in zip(val.predicted_id, val.predicted_class)):
            raise ValueError("IDs previstos na validação estão inconsistentes")
        np.testing.assert_array_equal(val[[f"prob_{name}" for name in NAMES]].to_numpy().argmax(1), val.predicted_id)
        validation_metrics = metrics(val.class_id.to_numpy(), val.predicted_id.to_numpy())
        model_metrics[spec["id"]] = {"test_common": test_metrics, "validation": validation_metrics}
        coverage.append({"model": spec["id"], "requested": len(requested), "used": info["used"],
                         "failed": info["failed"], "coverage_fraction": info["coverage_fraction"],
                         "common_test": len(common)})
        for split, value in (("validation", validation_metrics), ("test_common", test_metrics)):
            comparisons.append({"model": spec["id"], "family": spec["family"], "seed": spec["seed"],
                                "split": split, "examples": value["examples"], **focus(value)})
            for name in NAMES:
                class_rows.append({"model": spec["id"], "family": spec["family"], "seed": spec["seed"],
                                   "split": split, "class_name": name, **value["classification_report"][name]})
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "validation_test_by_model.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class_by_model.csv", index=False)
    aggregate, aggregate_classes, differences = {}, [], []
    for family in ("dummy", "random_forest", "linear", "mlp", "gru"):
        selected = [run for run in runs if run["spec"]["family"] == family]
        aggregate[family] = {}
        for split in ("validation", "test_common"):
            values = [model_metrics[run["spec"]["id"]][split] for run in selected]
            stats = summarize_values(values, output / "aggregate" / family / split)
            aggregate[family][split] = stats
            for name in NAMES:
                aggregate_classes.append({"family": family, "split": split, "class_name": name,
                    **{f"{key}_{stat}": value[stat] for key, value in stats["per_class"][name].items() for stat in ("mean", "std")}})
        for run in selected:
            values = model_metrics[run["spec"]["id"]]
            differences.append({"model": run["spec"]["id"], "family": family, "seed": run["spec"]["seed"],
                **{f"test_minus_validation_{key}_pp": 100*(focus(values["test_common"])[key]-focus(values["validation"])[key])
                   for key in ("accuracy", "macro_f1", "medium_recall")}})
    pd.DataFrame(aggregate_classes).to_csv(output / "per_class_mean_std.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "test_minus_validation_by_model.csv", index=False)
    deltas = {split: {reference: {key: 100*(aggregate["gru"][split][key]["mean"]-aggregate[reference][split][key]["mean"])
               for key in ("accuracy", "macro_f1")} for reference in ("mlp", "linear", "random_forest")}
              for split in ("validation", "test_common")}
    paired = [{"seed": seed, **{f"{split}_gru_minus_mlp_macro_f1_pp": 100*(model_metrics[f"gru_seed_{seed}"][split]["macro_f1"]
                           - model_metrics[f"mlp_seed_{seed}"][split]["macro_f1"]) for split in ("validation", "test_common")}}
              for seed in (42, 43, 44)]
    pd.DataFrame(paired).to_csv(output / "paired_gru_minus_mlp.csv", index=False)
    summary = {"requested_test": len(requested), "common_test": len(common), "coverage": coverage,
               "models": model_metrics, "aggregate": aggregate, "gru_minus_reference_pp": deltas,
               "paired_gru_minus_mlp": paired, "standard_deviation_ddof": 1,
               "primary_candidate": "gru", "operational_reference": "gru_seed_42", "ensemble": False}
    write_json(output / "summary.json", summary)
    lines = ["# FeedWatch — avaliação experimental final", "",
        "Protocolo prospectivo: [protocol.md](protocol.md); configuração e SHA-256 dos checkpoints: [config.json](config.json). "
        "Nenhum treinamento, ajuste de hiperparâmetros, mudança de pré-processamento, ensemble ou seleção pelo teste.", "",
        f"Teste: **2022_6_22, {len(requested)} vídeos**. Interseção comum: **{len(common)}**. "
        "Métrica principal: macro-F1. Médias ± desvio-padrão amostral entre as três seeds (ddof=1); uma seed nas referências.", "",
        "## Cobertura", "", "| Modelo | Usados / solicitados | Falhas |", "|---|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['model']} | {row['used']} / {row['requested']} | {row['failed']} |")
    lines += ["", "Falhas e avisos por vídeo: `inputs/motion/failures.csv`, `inputs/visual/failures.csv` e respectivos `extraction.csv`. "
              "Nenhuma falha foi substituída por entrada zerada. As tabelas abaixo usam a interseção comum no teste; validação mantém seus 1.705 vídeos.",
              "", "## Validação versus teste", "",
              "| Modelo | Acurácia teste | Macro-F1 validação | Macro-F1 teste | Teste − val (pp) |",
              "|---|---:|---:|---:|---:|"]
    for family, item in aggregate.items():
        test, val = item["test_common"], item["validation"]
        def show(value):
            return f"{value['mean']:.6f}" if value["std"] is None else display(value)
        lines.append(f"| {family} | {show(test['accuracy'])} | {show(val['macro_f1'])} | {show(test['macro_f1'])} | "
                     f"{100*(test['macro_f1']['mean']-val['macro_f1']['mean']):+.3f} |")
    lines += ["", "| Modelo / seed | Acurácia teste | Macro-F1 validação | Macro-F1 teste |", "|---|---:|---:|---:|"]
    for run in runs:
        name = run["spec"]["id"]
        item = model_metrics[name]
        lines.append(f"| {name} | {item['test_common']['accuracy']:.6f} | {item['validation']['macro_f1']:.6f} | {item['test_common']['macro_f1']:.6f} |")
    lines += ["", "## Confusões de interesse", "",
              "Taxas usam o suporte da classe real; comparar somente contagens entre dias pode confundir tamanho/composição com desempenho.", "",
              "| Modelo | Recall medium val → teste | medium→strong val → teste | strong→none val → teste |",
              "|---|---:|---:|---:|"]
    for family, item in aggregate.items():
        val, test = item["validation"], item["test_common"]
        def percent_pair(key):
            a, b = val[key]["mean"], test[key]["mean"]
            return "n/a" if a is None or b is None else f"{100*a:.2f}% → {100*b:.2f}%"
        lines.append(f"| {family} | {percent_pair('medium_recall')} | {percent_pair('medium_to_strong_rate')} | {percent_pair('strong_to_none_rate')} |")
    lines += ["", "Contagens por seed, suportes e taxas: [validation_test_by_model.csv](validation_test_by_model.csv). "
              "Precisão/recall/F1 por classe: [per_class_by_model.csv](per_class_by_model.csv) e [per_class_mean_std.csv](per_class_mean_std.csv). "
              "Matrizes individuais: `models/<modelo>/confusion_matrix.csv`; médias e SD: `aggregate/<família>/<divisão>/confusion_*.csv`. "
              "Média de métricas/matrizes não é previsão de ensemble.", "", "## Conclusão", ""]
    for reference in ("mlp", "linear", "random_forest"):
        v = deltas["validation"][reference]["macro_f1"]
        t = deltas["test_common"][reference]["macro_f1"]
        status = "manteve vantagem média" if t > 0 else "não manteve vantagem média"
        lines.append(f"GRU − {reference}: macro-F1 {v:+.3f} pp na validação e {t:+.3f} pp no teste; {status} nesse dia reservado.")
    operational = model_metrics["gru_seed_42"]["test_common"]
    lines += ["", f"A referência operacional permanece **GRU seed 42** (macro-F1 teste {operational['macro_f1']:.6f}; "
              f"recall medium {100*operational['classification_report']['medium']['recall']:.2f}%), definida antes do acesso ao teste. "
              "Não houve troca de seed/modelo em resposta ao teste.", "",
              "Os números descrevem os clipes desse único dia. A variação entre seeds representa instabilidade de treinamento, "
              "não incerteza entre dias ou fazendas. Referências históricas usam uma seed e configurações de treinamento diferentes. "
              "Repetir seleção no mesmo dia de validação pode produzir otimismo; clipes próximos são correlacionados. "
              "Esta avaliação não demonstra causalmente benefício temporal: a ablação anterior não mostrou prejuízo médio ao embaralhar a ordem.", "",
              "Causas como reflexos, recorte central, mudança de iluminação/contexto ou ambiguidade de rótulos são hipóteses, "
              "não causas verificadas dos erros. Nenhum rótulo foi alterado nem correção implementada após observar o teste.", "",
              "## Limitações que a futura demonstração deverá comunicar", "",
              "- Classifica **intensidade alimentar observada**. Não comprova fome, saciedade nem segurança para controlar um alimentador.",
              "- Usa janelas iniciais de 2 s, 16 frames e recorte central que descarta laterais; não acompanha peixes individuais nem todo o vídeo de forma contínua.",
              "- Expor a dificuldade em medium e as confusões medium→strong/strong→none medidas neste relatório; não apresentar argmax como certeza ou probabilidades como confiança calibrada.",
              "- Falhas de leitura e vídeos fora dos requisitos temporais devem aparecer como indisponíveis; não representar erro como intensidade none.",
              "- Um único dia reservado do mesmo pacote não certifica desempenho em outras fazendas, câmeras, espécies ou condições; uso em vídeo contínuo ainda não foi avaliado.",
              "- Não há validação de latência, estabilidade entre janelas, detecção de domínio desconhecido ou operação autônoma de alimentador. Nenhuma interface foi implementada.", "",
              "## Integridade e reprodução", "",
              "Os nove checkpoints foram recarregados, IDs/probabilidades reproduzidos e métricas recalculadas dos CSVs. "
              "Protocolo/configuração e artefatos anteriores são verificados por SHA-256; entradas, cache e inventário de vídeos são registrados. "
              "Inventário de vídeos usa tamanho/mtime; não equivale a hash binário integral dos originais. "
              "Detalhes: [verification.json](verification.json), [results.json](results.json), `inputs_sha256.json`, `deliverables_sha256.json`. "
              "Tempos e memória do extrator estão em `results.json`; tempos individuais de inferência em `models/<modelo>/identification.json`. Tempo de treinamento nesta etapa: **zero**.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
