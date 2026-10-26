"""Treina somente uma transformação afim dos embeddings temporais médios."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CLASS_IDS
from .evaluation import metrics


def train_statistics(x: np.ndarray, y: np.ndarray, balance: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mean, std = x.mean(0), x.std(0)
    std = np.maximum(std, 1e-6)
    counts = np.bincount(y, minlength=4)
    if (counts == 0).any():
        raise ValueError("Treino precisa conter as quatro classes")
    if balance not in ("none", "inverse_frequency"):
        raise ValueError("Balanceamento desconhecido")
    weights = (len(y)/(4*counts)).astype(np.float32) if balance == "inverse_frequency" else None
    return mean, std, weights


@torch.inference_mode()
def probabilities(head: nn.Module, x: torch.Tensor) -> np.ndarray:
    head.eval()
    return head(x).softmax(1).cpu().numpy()


def load_probe(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["class_mapping"] != CLASS_IDS or bundle["input_dim"] != 576:
        raise ValueError("Checkpoint com mapeamento/dimensão incompatível")
    head = nn.Linear(bundle["input_dim"], 4).to(device)
    head.load_state_dict(bundle["state_dict"])
    head.eval()
    return head, bundle


def train_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                config: dict, seed: int, device: torch.device, output: Path, provenance: dict) -> dict:
    cfg = config
    if not 1 <= cfg["max_epochs"] <= 200 or cfg["batch_size"] < 1:
        raise ValueError("Épocas/batch_size inválidos")
    if not cfg["standardize_train_only"]:
        raise ValueError("Este experimento requer padronização estimada somente no treino")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("Embeddings inválidos; não há imputação por zero")
    mean, std, weights = train_statistics(train_x, train_y, cfg["balance"])
    x = torch.from_numpy((train_x-mean)/std).to(device)
    v = torch.from_numpy((val_x-mean)/std).to(device)
    y, vy = torch.from_numpy(train_y).long().to(device), torch.from_numpy(val_y).long().to(device)
    torch.manual_seed(seed)
    head = nn.Linear(train_x.shape[1], 4).to(device)
    loss_function = nn.CrossEntropyLoss(weight=None if weights is None else torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    started, best_score, best_epoch, history = time.perf_counter(), -1.0, None, []
    for epoch in range(1, cfg["max_epochs"]+1):
        epoch_started = time.perf_counter()
        head.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for indices in order.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            train_logits, val_logits = head(x), head(v)
            train_loss, val_loss = loss_function(train_logits, y).item(), loss_function(val_logits, vy).item()
            t = metrics(train_y, train_logits.argmax(1).cpu().numpy())
            a = metrics(val_y, val_logits.argmax(1).cpu().numpy())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": t["accuracy"], "train_macro_f1": t["macro_f1"],
                        "val_accuracy": a["accuracy"], "val_macro_f1": a["macro_f1"],
                        "epoch_seconds": time.perf_counter()-epoch_started})
        if a["macro_f1"] > best_score:
            best_score, best_epoch = a["macro_f1"], epoch
            bundle = {"state_dict": {k: tensor.detach().cpu().clone() for k, tensor in head.state_dict().items()},
                      "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "class_mapping": CLASS_IDS,
                      "input_dim": train_x.shape[1], "selected_epoch": epoch, "selection": "validation_macro_f1",
                      "validation_macro_f1": best_score, "training_config": cfg, "seed": seed,
                      "provenance": provenance, "extractor_trained": False, "test_evaluated": False}
            torch.save(bundle, output / "best_classifier.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg["max_epochs"]:
            print(f"Época {epoch}/{cfg['max_epochs']}: train_f1={t['macro_f1']:.4f}; val_f1={a['macro_f1']:.4f}; melhor={best_score:.4f} (época {best_epoch})", flush=True)
    training_seconds = time.perf_counter()-started
    loaded, bundle = load_probe(output / "best_classifier.pt", device)
    loaded_x = torch.from_numpy((train_x-bundle["mean"].numpy())/bundle["std"].numpy()).to(device)
    loaded_v = torch.from_numpy((val_x-bundle["mean"].numpy())/bundle["std"].numpy()).to(device)
    train_prob, val_prob = probabilities(loaded, loaded_x), probabilities(loaded, loaded_v)
    if abs(metrics(val_y, val_prob.argmax(1))["macro_f1"]-best_score) > 1e-12:
        raise ValueError("Checkpoint recarregado diverge do melhor macro-F1")
    return {"train_probabilities": train_prob, "val_probabilities": val_prob,
            "selected_epoch": best_epoch, "best_val_macro_f1": best_score, "training_seconds": training_seconds,
            "epochs": cfg["max_epochs"], "trainable_parameters": sum(p.numel() for p in loaded.parameters()),
            "class_weights": None if weights is None else weights.tolist()}


def plot_history(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = pd.read_csv(path / "history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"), ("Entropia cruzada", "Acurácia", "Macro-F1")):
        for split, label in (("train", "Treino (reutilização)"), ("val", "Validação")):
            axis.plot(history.epoch, history[f"{split}_{metric}"], label=label)
        axis.set(xlabel="Época", ylabel=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(path / "curves.png", dpi=160)
    fig.savefig(path / "curves.svg")
    plt.close(fig)
