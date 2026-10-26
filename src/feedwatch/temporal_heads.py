"""MLP de capacidade semelhante e GRU de uma camada sobre os mesmos embeddings."""

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CLASS_IDS, write_json
from .evaluation import metrics, save_evaluation
from .linear_probe import plot_history


class TemporalHead(nn.Module):
    def __init__(self, kind: str, config: dict):
        super().__init__()
        self.kind, self.config = kind, config
        self.dropout = nn.Dropout(config["dropout"])
        if kind == "mlp":
            self.hidden = nn.Linear(config["embedding_dim"], config["mlp_hidden_size"])
            self.classifier = nn.Linear(config["mlp_hidden_size"], 4)
        elif kind == "gru":
            self.gru = nn.GRU(config["embedding_dim"], config["gru_hidden_size"], num_layers=1,
                              batch_first=True, bidirectional=False, dropout=0.0)
            self.classifier = nn.Linear(config["gru_hidden_size"], 4)
        else:
            raise ValueError("Cabeça desconhecida")

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (16, self.config["embedding_dim"]):
            raise ValueError("Esperada sequência real (N,16,576), não média reconstruída")
        if self.kind == "mlp":
            hidden = torch.relu(self.hidden(sequence.mean(1)))
        else:
            _, last = self.gru(sequence)
            hidden = last[-1]
        return self.classifier(self.dropout(hidden))


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        if patience < 1 or min_delta != 0:
            raise ValueError("Use patience positivo e min_delta=0 para selecionar o maior macro-F1 exato")
        self.patience, self.best, self.stale = patience, -math.inf, 0

    def step(self, score: float) -> tuple[bool, bool]:
        if not math.isfinite(score):
            raise ValueError("Métrica de seleção não finita")
        improved = score > self.best
        if improved:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def validation_permutations(paths: list[str], seed: int, length: int = 16) -> np.ndarray:
    result = []
    for path in paths:
        digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        order = rng.permutation(length)
        if np.array_equal(order, np.arange(length)):
            order = np.roll(order, 1)
        result.append(order)
    return np.asarray(result, dtype=np.int64)


def apply_permutations(x: torch.Tensor, orders: np.ndarray) -> torch.Tensor:
    if orders.shape != x.shape[:2] or not np.all(np.sort(orders, axis=1) == np.arange(x.shape[1])):
        raise ValueError("Permutação deve preservar exatamente os mesmos frames")
    index = torch.from_numpy(orders).to(x.device)
    return x.gather(1, index.unsqueeze(-1).expand_as(x))


@torch.inference_mode()
def evaluate_head(model: nn.Module, x: torch.Tensor, y: np.ndarray, batch_size: int) -> tuple[np.ndarray, dict, float]:
    model.eval()
    outputs = [model(batch).cpu() for batch in x.split(batch_size)]
    logits = torch.cat(outputs)
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y).long()).item()
    probabilities = logits.softmax(1).numpy()
    return probabilities, metrics(y, probabilities.argmax(1)), loss


def load_temporal_checkpoint(path: Path, device: torch.device) -> tuple[TemporalHead, dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["class_mapping"] != CLASS_IDS or bundle["extractor_trained"]:
        raise ValueError("Checkpoint incompatível")
    model = TemporalHead(bundle["kind"], bundle["head_config"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def train_head(kind: str, seed: int, heads: dict, cfg: dict, train_x: torch.Tensor, train_y: np.ndarray,
               val_x: torch.Tensor, val_y: np.ndarray, train_rows: pd.DataFrame, val_rows: pd.DataFrame,
               mean: np.ndarray, std: np.ndarray, output: Path, provenance: dict, permutations: np.ndarray) -> dict:
    if cfg["balance"] != "none" or cfg["max_epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("Configuração de treino inválida; este controle não usa pesos de classes")
    device = train_x.device
    torch.manual_seed(seed)
    model = TemporalHead(kind, heads).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    stopping = EarlyStopping(cfg["patience"], cfg["min_delta"])
    generator = torch.Generator(device=device).manual_seed(seed)
    targets = torch.from_numpy(train_y).long().to(device)
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, history, best_epoch = time.perf_counter(), [], None
    for epoch in range(1, cfg["max_epochs"]+1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator, device=device)
        for batch in order.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[batch]), targets[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
        _, t, train_loss = evaluate_head(model, train_x, train_y, cfg["evaluation_batch_size"])
        _, v, val_loss = evaluate_head(model, val_x, val_y, cfg["evaluation_batch_size"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": t["accuracy"], "train_macro_f1": t["macro_f1"],
                        "val_accuracy": v["accuracy"], "val_macro_f1": v["macro_f1"],
                        "epoch_seconds": time.perf_counter()-epoch_started})
        improved, stop = stopping.step(v["macro_f1"])
        if improved:
            best_epoch = epoch
            torch.save({"state_dict": {k: p.detach().cpu().clone() for k, p in model.state_dict().items()},
                        "kind": kind, "seed": seed, "head_config": heads, "training_config": cfg,
                        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "class_mapping": CLASS_IDS,
                        "selected_epoch": epoch, "validation_macro_f1": v["macro_f1"], "parameters": parameters,
                        "selection": "maximum chronological validation macro-F1; earliest exact tie",
                        "provenance": provenance, "extractor_trained": False, "test_evaluated": False},
                       output / "best_head.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or stop:
            print(f"{kind} seed={seed} época {epoch}: train_f1={t['macro_f1']:.4f}; val_f1={v['macro_f1']:.4f}; melhor={stopping.best:.4f} (época {best_epoch})", flush=True)
        if stop:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter()-started
    memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else None)
    del model
    model, bundle = load_temporal_checkpoint(output / "best_head.pt", device)
    train_prob, _, _ = evaluate_head(model, train_x, train_y, cfg["evaluation_batch_size"])
    val_prob, replay, _ = evaluate_head(model, val_x, val_y, cfg["evaluation_batch_size"])
    if abs(replay["macro_f1"]-stopping.best) > 1e-12:
        raise ValueError("Checkpoint não reproduz a métrica selecionada")
    train_metrics = save_evaluation(train_rows, train_prob, output / "train", "train")
    val_metrics = save_evaluation(val_rows, val_prob, output / "val", "val")
    # Recarrega novamente depois de salvar previsões; comparação não depende do objeto em memória.
    reloaded, _ = load_temporal_checkpoint(output / "best_head.pt", device)
    actual, _, _ = evaluate_head(reloaded, val_x, val_y, cfg["evaluation_batch_size"])
    saved = pd.read_csv(output / "val" / "predictions.csv")
    np.testing.assert_array_equal(actual.argmax(1), saved.predicted_id)
    np.testing.assert_allclose(actual, saved[[f"prob_{name}" for name in CLASS_IDS]], atol=1e-7)
    shuffled = None
    if kind == "gru":
        shuffled_x = apply_permutations(val_x, permutations)
        ablation_started = time.perf_counter()
        shuffled_prob, _, _ = evaluate_head(reloaded, shuffled_x, val_y, cfg["evaluation_batch_size"])
        shuffled = save_evaluation(val_rows, shuffled_prob, output / "val_shuffled", "val_shuffled_diagnostic")
        shuffled["evaluation_seconds"] = time.perf_counter()-ablation_started
        again, _, _ = evaluate_head(reloaded, shuffled_x, val_y, cfg["evaluation_batch_size"])
        np.testing.assert_array_equal(again.argmax(1), pd.read_csv(output / "val_shuffled" / "predictions.csv").predicted_id)
        write_json(output / "val_shuffled" / "diagnostic.json", {
            "same_checkpoint": "../best_head.pt", "retrained": False, "used_for_selection": False,
            "permutation_sha256": provenance["permutation_sha256"], "checkpoint_replay_verified": True})
    plot_history(output)
    result = {"kind": kind, "seed": seed, "parameters": parameters, "training_seconds": training_seconds,
              "epochs_run": len(history), "selected_epoch": best_epoch,
              "stop_reason": "patience" if stop else "max_epochs", "gpu_memory": memory,
              "train": train_metrics, "validation": val_metrics, "shuffled_validation_diagnostic": shuffled,
              "train_val_accuracy_gap_pp": 100*(train_metrics["accuracy"]-val_metrics["accuracy"]),
              "train_val_macro_f1_gap_pp": 100*(train_metrics["macro_f1"]-val_metrics["macro_f1"]),
              "checkpoint_reproduces_predictions": True, "test_evaluated": False}
    write_json(output / "results.json", result)
    return result
