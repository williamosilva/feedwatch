"""Agregação de seeds, sem tratar clipes correlacionados como réplicas independentes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import write_json
from .evaluation import NAMES


def mean_sd(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else None,
            "n": len(array), "values": array.tolist()}


def display(stat: dict, multiplier=1.0) -> str:
    sd = "n/a" if stat["std"] is None else f"{multiplier*stat['std']:.6f}"
    return f"{multiplier*stat['mean']:.6f} ± {sd}"


def summarize_temporal(runs: list[dict], output: Path, visual_dir: Path, motion_dir: Path, audit: dict) -> dict:
    metrics_rows, class_rows, models = [], [], {}
    for scenario in ("mlp", "gru", "gru_shuffled"):
        selected = [run for run in runs if run["kind"] == ("gru" if scenario == "gru_shuffled" else scenario)]
        values = [run["shuffled_validation_diagnostic"] if scenario == "gru_shuffled" else run["validation"] for run in selected]
        for run, value in zip(selected, values):
            metrics_rows.append({"scenario": scenario, "seed": run["seed"], "accuracy": value["accuracy"],
                                 "macro_f1": value["macro_f1"], "train_accuracy": run["train"]["accuracy"],
                                 "train_macro_f1": run["train"]["macro_f1"], "training_seconds": run["training_seconds"],
                                 "parameters": run["parameters"], "epochs_run": run["epochs_run"],
                                 "selected_epoch": run["selected_epoch"],
                                 "medium_to_strong": value["confusion_matrix"][2][3],
                                 "strong_to_none": value["confusion_matrix"][3][0],
                                 "train_val_accuracy_gap_pp": 100*(run["train"]["accuracy"]-value["accuracy"]),
                                 "train_val_macro_f1_gap_pp": 100*(run["train"]["macro_f1"]-value["macro_f1"])})
            for label in NAMES:
                class_rows.append({"scenario": scenario, "seed": run["seed"], "class_name": label,
                                   **value["classification_report"][label]})
        matrices = np.stack([value["confusion_matrix"] for value in values]).astype(float)
        directory = output / "aggregate" / scenario
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrices.mean(0), index=pd.Index(NAMES, name="actual"), columns=NAMES).to_csv(directory / "confusion_mean.csv")
        if len(matrices) > 1:
            pd.DataFrame(matrices.std(0, ddof=1), index=pd.Index(NAMES, name="actual"), columns=NAMES).to_csv(directory / "confusion_std.csv")
        local = [row for row in metrics_rows if row["scenario"] == scenario]
        models[scenario] = {key: mean_sd([row[key] for row in local]) for key in
                            ("accuracy", "macro_f1", "train_accuracy", "train_macro_f1", "training_seconds", "parameters",
                             "epochs_run", "selected_epoch", "train_val_accuracy_gap_pp", "train_val_macro_f1_gap_pp",
                             "medium_to_strong", "strong_to_none")}
        per_class = {label: {key: mean_sd([v["classification_report"][label][key] for v in values])
                             for key in ("precision", "recall", "f1-score", "support")} for label in NAMES}
        models[scenario]["per_class"] = per_class
        write_json(directory / "statistics.json", models[scenario])
    pd.DataFrame(metrics_rows).to_csv(output / "metrics_by_seed.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class_by_seed.csv", index=False)
    pd.DataFrame([{"scenario": scenario, "class_name": label,
                   **{f"{key}_{stat}": value[stat] for key, value in item["per_class"][label].items() for stat in ("mean", "std")}}
                  for scenario, item in models.items() for label in NAMES]).to_csv(output / "per_class_mean_std.csv", index=False)
    paired = []
    for gru in [run for run in runs if run["kind"] == "gru"]:
        mlp = next(run for run in runs if run["kind"] == "mlp" and run["seed"] == gru["seed"])
        paired.append({"seed": gru["seed"], **{f"gru_minus_mlp_{key}_pp": 100*(gru["validation"][key]-mlp["validation"][key])
                        for key in ("accuracy", "macro_f1")},
                       **{f"chronological_minus_shuffled_{key}_pp": 100*(gru["validation"][key]-gru["shuffled_validation_diagnostic"][key])
                          for key in ("accuracy", "macro_f1")}})
    pd.DataFrame(paired).to_csv(output / "paired_deltas_by_seed.csv", index=False)
    delta_stats = {key: mean_sd([item[key] for item in paired]) for key in paired[0] if key != "seed"}
    visual = json.loads((visual_dir / "results.json").read_text(encoding="utf-8"))
    motion = json.loads((motion_dir / "results.json").read_text(encoding="utf-8"))
    historical = {"random_forest_single_seed": {key: motion["models"]["random_forest"][key] for key in ("accuracy", "macro_f1")},
                  "visual_linear_single_seed": {key: visual["validation"][key] for key in ("accuracy", "macro_f1")}}
    differences = {name: {reference: {metric: 100*(models[name][metric]["mean"]-values[metric]) for metric in ("accuracy", "macro_f1")}
                         for reference, values in historical.items()} for name in ("mlp", "gru")}
    no_hurt_seeds = [row["seed"] for row in paired if row["chronological_minus_shuffled_macro_f1_pp"] <= 0]
    summary = {"models": models, "paired_deltas_pp": delta_stats, "historical_single_seed": historical,
               "delta_pp_vs_historical": differences, "seeds_with_no_macro_f1_damage_from_shuffle": no_hurt_seeds,
               "order_benefit_mean_macro_f1_drop_pp": delta_stats["chronological_minus_shuffled_macro_f1_pp"]["mean"],
               "input_audit": audit, "standard_deviation_ddof": 1, "test_evaluated": False}
    write_json(output / "summary.json", summary)
    lines = ["# Experimento controlado: MLP versus GRU", "",
             f"Entradas: {audit['videos']} vídeos; shape {audit['shape']}; {audit['original_cache']} caches originais reutilizados; "
             f"{audit['new_extractions']} reextrações; leitura/verificação {audit['wall_seconds']:.2f} s.", "",
             "Média ± desvio-padrão amostral entre seeds (ddof=1). Com uma única seed de piloto, SD não é estimável.", "",
             "| Cenário | Acurácia | Macro-F1 | Gap treino−val macro-F1 (pp) | Treino (s) | Parâmetros |",
             "|---|---:|---:|---:|---:|---:|"]
    for scenario, item in models.items():
        lines.append(f"| {scenario} | {display(item['accuracy'])} | {display(item['macro_f1'])} | "
                     f"{display(item['train_val_macro_f1_gap_pp'])} | {display(item['training_seconds'])} | {item['parameters']['mean']:.0f} |")
    lines += ["", "`gru_shuffled` é uma ablação do mesmo checkpoint; o tempo de treino e os parâmetros repetem a GRU e não representam outro ajuste.",
              "", "## Treino e diferença para validação", "",
              "Métricas de treino usam os próprios exemplos de ajuste em eval; não medem generalização.", "",
              "| Cabeça | Acurácia treino | Macro-F1 treino | Gap acurácia (pp) | Gap macro-F1 (pp) |", "|---|---:|---:|---:|---:|"]
    for scenario in ("mlp", "gru"):
        item = models[scenario]
        lines.append(f"| {scenario} | {display(item['train_accuracy'])} | {display(item['train_macro_f1'])} | "
                     f"{display(item['train_val_accuracy_gap_pp'])} | {display(item['train_val_macro_f1_gap_pp'])} |")
    lines += ["", "## Confusões de interesse", "",
              "| Cenário | medium→strong (contagem média ± SD) | strong→none (contagem média ± SD) |", "|---|---:|---:|"]
    for scenario, item in models.items():
        lines.append(f"| {scenario} | {display(item['medium_to_strong'])} | {display(item['strong_to_none'])} |")
    lines += [
              "", "## Resultados por seed", "", "| Cabeça | Seed | Acurácia | Macro-F1 | Época selecionada / executadas | medium→strong | strong→none |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows:
        lines.append(f"| {row['scenario']} | {row['seed']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
                     f"{row['selected_epoch']}/{row['epochs_run']} | {row['medium_to_strong']} | {row['strong_to_none']} |")
    lines += ["", "## Diferenças pareadas", ""]
    lines += [f"- {key}: {display(value)} pp." for key, value in delta_stats.items()]
    lines += ["", "## Referências históricas (uma seed)", "", "| Referência | Acurácia | Macro-F1 |", "|---|---:|---:|"]
    lines += [f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} |" for name, value in historical.items()]
    for head, refs in differences.items():
        for reference, delta in refs.items():
            lines += ["", f"{head} − {reference}: acurácia {delta['accuracy']:+.3f} pp; macro-F1 {delta['macro_f1']:+.3f} pp."]
    lines += ["", "## Por classe (validação)", "", "| Cenário | Classe | Precisão | Recall | F1 |", "|---|---|---:|---:|---:|"]
    for scenario, item in models.items():
        for label in NAMES:
            c = item["per_class"][label]
            lines.append(f"| {scenario} | {label} | {display(c['precision'])} | {display(c['recall'])} | {display(c['f1-score'])} |")
    for scenario in models:
        lines += ["", f"## Matriz média: {scenario}", "", "Linhas reais e colunas previstas. Contagens médias por modelo, não previsão de ensemble.", "",
                  "| Real / prevista | none | weak | medium | strong |", "|---|---:|---:|---:|---:|"]
        matrix = pd.read_csv(output / "aggregate" / scenario / "confusion_mean.csv", index_col=0)
        for label, row in matrix.iterrows():
            lines.append(f"| {label} | " + " | ".join(f"{value:.2f}" for value in row) + " |")
    lines += ["", "## Interpretação e limites", "",
              f"GRU − MLP em macro-F1: {delta_stats['gru_minus_mlp_macro_f1_pp']['mean']:+.3f} pp em média; "
              f"GRU − linear histórica: {differences['gru']['visual_linear_single_seed']['macro_f1']:+.3f} pp."]
    wins = sum(row["gru_minus_mlp_macro_f1_pp"] > 0 for row in paired)
    lines.append(f"A GRU superou a MLP em macro-F1 em {wins} de {len(paired)} seeds. O ganho médio deve ser lido junto às diferenças pareadas e à sua variação, sem escolher a melhor seed.")
    if summary["order_benefit_mean_macro_f1_drop_pp"] <= 0:
        lines.append("Embaralhar não prejudicou o macro-F1 médio: não encontramos evidência clara de benefício da ordem neste experimento.")
    elif no_hurt_seeds:
        lines.append(f"A queda média após embaralhar foi positiva, mas nas seeds {no_hurt_seeds} não houve prejuízo de macro-F1; o resultado sobre ordem não foi consistente entre todos os ajustes.")
    else:
        lines.append("Embaralhar reduziu o macro-F1 nas três seeds, um efeito observado compatível com dependência da ordem; isso não demonstra compreensão de atividade alimentar nem isola movimento de preferência pelo último frame.")
    lines += ["", "A variação entre seeds mede instabilidade do ajuste, não incerteza entre fazendas ou dias independentes. Três seeds não sustentam uma conclusão estatística ampla.",
              "O mesmo dia de validação foi reutilizado em diversos experimentos e seleciona checkpoints/early stopping. Existe risco crescente de adaptação ao dia de validação, inclusive por decisões humanas; as métricas não são avaliação final independente.",
              "As referências históricas têm uma seed e procedimentos de treino diferentes. MLP e GRU compartilham dados, normalização, regularização e seleção, mas quantidade semelhante de parâmetros não iguala custo computacional, arquitetura ou dinâmica de otimização.",
              "O recorte central descarta regiões do tanque; exemplos efetivos estão em `input_examples/`. Nenhum rótulo foi alterado. Média temporal não usa ordem; GRU pode usar aparência do último frame, além de transições.",
              "O embaralhamento é uma única ablação fixa por vídeo, compartilhada pelas seeds, sem nova seleção ou treinamento. Não escolhemos permutação de melhor desempenho.",
              "Somente as cabeças foram treinadas. Cache e extrator congelado são os do experimento visual; nenhum dado de teste foi utilizado.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
