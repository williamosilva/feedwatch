"""GRU operacional congelada e janelas independentes; sem rótulos na inferência."""

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import CLASS_IDS, file_hash
from .final_evaluation import inference_setup
from .final_protocol import verify_protocol
from .motion import identity_hash
from .temporal_data import check_identity
from .temporal_heads import load_temporal_checkpoint
from .visual import encode_frames, load_extractor, preprocess_bgr, sample_visual_indices, state_hash, visual_identity

PORTUGUESE = {"none": "Ausente", "weak": "Fraca", "medium": "Média", "strong": "Forte"}


class OperationalPredictor:
    def __init__(self, protocol: Path = Path("artifacts/experiments/final-test"), device_name: str = "auto"):
        config = verify_protocol(protocol)  # Verifica hashes, sem ler o split/vídeos do teste.
        if config["operational_seed"] != 42 or config["primary_candidate"] != "gru":
            raise ValueError("A referência operacional deve continuar sendo GRU seed 42")
        spec = next(item for item in config["models"] if item["family"] == "gru" and item["seed"] == 42)
        self.device = inference_setup(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
        self.identity = config["extraction_identities"]["visual"]
        check_identity(self.identity)
        self.extraction = self.identity["parameters"]
        self.extractor, self.transform, weights = load_extractor(Path(self.identity["weights"]["weights_file"]).parent, self.device)
        runtime_identity = visual_identity(self.extraction, weights, self.device)
        if any(runtime_identity[key] != value for key, value in self.identity.items() if key != "device_type"):
            raise ValueError("Extrator/pré-processamento diferem do experimento")
        self.head, self.bundle = load_temporal_checkpoint(Path(spec["checkpoint"]), self.device)
        if self.bundle["kind"] != "gru" or self.bundle["seed"] != 42:
            raise ValueError("Checkpoint operacional inválido")
        if self.bundle["provenance"]["extraction_config_id"] != identity_hash(self.identity):
            raise ValueError("Checkpoint e extrator não correspondem")
        self.head.requires_grad_(False).eval()
        self.mean, self.std = self.bundle["mean"].numpy(), self.bundle["std"].numpy()
        if self.mean.shape != (576,) or self.std.shape != (576,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Padronização salva inválida")
        self.initial_extractor = state_hash(self.extractor)
        self.initial_head = state_hash(self.head)
        self.identification = {"checkpoint": str(Path(spec["checkpoint"]).resolve()),
            "checkpoint_sha256": file_hash(Path(spec["checkpoint"])), "seed": 42, "kind": "gru",
            "selected_epoch": self.bundle["selected_epoch"], "class_mapping": CLASS_IDS,
            "extractor_identity": self.identity, "extraction_config_id": identity_hash(self.identity),
            "runtime_device": str(self.device), "historical_device": self.identity["device_type"],
            "train_only_normalization_sha256": hashlib.sha256(self.mean.tobytes()+self.std.tobytes()).hexdigest(),
            "normalization": "mean/std loaded from original train-only checkpoint; no fitting",
            "gru_state": "implicit zero initial state on every independent forward; never carried between windows",
            "scores": "softmax outputs, not calibrated confidence", "training": False,
            "protocol_config_sha256": file_hash(protocol / "config.json")}

    @torch.inference_mode()
    def predict_frames(self, frames: torch.Tensor) -> dict:
        if frames.shape != (16, 3, 224, 224):
            raise ValueError("São necessários exatamente 16 frames transformados, sem preenchimento")
        embeddings = encode_frames(self.extractor, frames, self.device)
        return self.predict_embeddings(embeddings)

    @torch.inference_mode()
    def predict_embeddings(self, embeddings: np.ndarray) -> dict:
        if embeddings.shape != (16, 576) or not np.isfinite(embeddings).all():
            raise ValueError("Sequência inválida; não substituir por zeros")
        x = torch.from_numpy((embeddings-self.mean)/self.std).unsqueeze(0).to(self.device)
        # TemporalHead chama self.gru(sequence) sem h0: estado reiniciado a cada chamada.
        scores = self.head(x).cpu().softmax(1).numpy()[0]
        ident = int(scores.argmax())
        name = next(name for name, value in CLASS_IDS.items() if value == ident)
        return {"predicted_id": ident, "predicted_class": name, "legend_pt": PORTUGUESE[name],
                "scores": dict(zip(CLASS_IDS, scores.tolist())),
                "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest()}

    def verify_frozen(self):
        if state_hash(self.extractor) != self.initial_extractor or state_hash(self.head) != self.initial_head:
            raise ValueError("Pesos/buffers mudaram durante a inferência")
        if any(module.training for model in (self.extractor, self.head) for module in model.modules()):
            raise ValueError("Todos os módulos devem permanecer em eval")


def open_capture(path: Path):
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_N_THREADS, 1])


def inspect_video(cap) -> dict:
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo")
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not all(math.isfinite(value) and value > 0 for value in (fps, count, width, height)):
        raise ValueError("FPS, contagem ou resolução inválidos; sem valores inventados para exportar")
    return {"fps": fps, "frame_count": round(count), "width": round(width), "height": round(height),
            "duration_seconds": round(count)/fps, "duration_basis": "frame_count / FPS, checked by full decode",
            "backend": cap.getBackendName()}


def plan_windows(metadata: dict, mode: str) -> list[dict]:
    fps, count, duration = metadata["fps"], metadata["frame_count"], metadata["duration_seconds"]
    if mode not in ("sequential", "clip"):
        raise ValueError("Use sequential ou clip")
    windows = []
    count_windows = max(1, math.ceil(duration/2-1e-9)) if mode == "sequential" else 1
    for number in range(count_windows):
        start = 2*number if mode == "sequential" else 0.
        end = min(start+2, duration) if mode == "sequential" else duration
        first = math.ceil(start*fps-1e-9)
        stop = min(count, math.ceil(end*fps-1e-9))
        item = {"window_id": number, "start_seconds": start, "end_seconds": end,
                "frame_start": first, "frame_stop_exclusive": stop, "status": "pending", "error": "",
                "available_at_seconds": end if mode == "sequential" else 0.,
                "sampled_indices": [], "sampled_timestamps_seconds": [], "predicted_class": None,
                "predicted_id": None, "scores": None}
        if mode == "sequential" and end-start < 2-1e-9:
            item.update(status="incomplete", error="Segmento final menor que 2 s; não analisado, sem padding")
        elif mode == "clip" and duration > 2+1e-6:
            item.update(status="failed", error="Modo clipe completo limitado a até 2 s; use sequential para vídeos longos")
        else:
            try:
                selected = sample_visual_indices(fps, stop-first, 16, 2.)
                item["sampled_indices"] = [first+index for index in selected]
            except ValueError as error:
                item.update(status="failed", error=str(error))
        windows.append(item)
    return windows


def analyze_video(path: Path, predictor: OperationalPredictor, mode: str, progress=None) -> dict:
    started, cap = time.perf_counter(), open_capture(path)
    try:
        metadata = inspect_video(cap)
        if progress: progress('food', 0, metadata['frame_count'])
        windows = plan_windows(metadata, mode)
        index, window_index, tensors, fatal = 0, 0, [], ""
        while index < metadata["frame_count"]:
            if not cap.grab():
                fatal = f"Falha de decodificação no frame {index}; esperado {metadata['frame_count']} frames"
                break
            window = windows[window_index]
            if window["status"] == "pending" and index in window["sampled_indices"]:
                try:
                    ok, image = cap.retrieve()
                    if not ok or image is None or image.size == 0:
                        raise ValueError(f"Falha de leitura do frame {index}")
                    if image.shape[:2] != (metadata["height"], metadata["width"]):
                        raise ValueError(f"Resolução mudou no frame {index}")
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    previous = window["sampled_timestamps_seconds"]
                    if (not math.isfinite(timestamp) or abs(timestamp-index/metadata["fps"]) > max(1.5/metadata["fps"], .02)
                        or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-index) > .5
                        or timestamp < window["start_seconds"]-1e-6 or timestamp >= window["end_seconds"]+1e-6
                        or (previous and timestamp <= previous[-1])):
                        raise ValueError(f"Timestamp/posição incompatível com contrato CFR no frame {index}: {timestamp}")
                    tensors.append(preprocess_bgr(image, predictor.extraction, predictor.transform))
                    previous.append(timestamp)
                except (ValueError, cv2.error) as error:
                    window.update(status="failed", error=str(error))
                    tensors = []
            index += 1
            if progress: progress('food', index, metadata['frame_count'])
            if index == window["frame_stop_exclusive"]:
                if window["status"] == "pending":
                    try:
                        values = predictor.predict_frames(torch.stack(tensors))
                        window.update(status="ok", **values)
                    except (ValueError, RuntimeError) as error:
                        window.update(status="failed", error=f"Análise indisponível: {error}")
                tensors = []
                window_index = min(window_index+1, len(windows)-1)
        if not fatal and cap.grab():
            fatal = "Contagem declarada diverge da decodificação: existem frames adicionais; exportação recusada"
        if fatal:
            for window in windows:
                if window["status"] == "pending":
                    window.update(status="failed", error=fatal)
        predictor.verify_frozen()
        return {"metadata": metadata, "windows": windows, "decoded_frames": index, "fatal_error": fatal,
                "analysis_seconds": time.perf_counter()-started}
    finally:
        cap.release()


def display_state(windows: list[dict], mode: str, timestamp: float) -> dict:
    if mode == "clip":
        item = windows[0]
        return {"status": "prediction" if item["status"] == "ok" else "unavailable", "window": item, "tail": False, "last_valid": None}
    complete = [item for item in windows if item["status"] != "incomplete" and item["available_at_seconds"] <= timestamp+1e-9]
    tail = next((item for item in windows if item["status"] == "incomplete" and timestamp >= item["start_seconds"]-1e-9), None)
    if not complete:
        return {"status": "unavailable" if tail else "analyzing", "window": tail, "tail": bool(tail), "last_valid": None}
    latest = complete[-1]
    last_valid = next((item for item in reversed(complete) if item["status"] == "ok"), None)
    return {"status": "prediction" if latest["status"] == "ok" else "unavailable", "window": latest,
            "tail": bool(tail), "last_valid": last_valid}
