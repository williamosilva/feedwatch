"""Embeddings por frame: MobileNetV3 Small ImageNet congelada, sem gradientes."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .data import file_hash, video_path
from .motion import identity_hash

EMBEDDING_DIM = 576


def sample_visual_indices(fps: float, frame_count: int, count: int, seconds: float) -> list[int]:
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or count < 2 or not 0 < seconds <= 60:
        raise ValueError("Parâmetros temporais inválidos")
    last = min(frame_count - 1, math.ceil(seconds * fps - 1e-9) - 1)
    if last + 1 < count:
        raise ValueError(f"A janela contém menos de {count} frames distintos")
    indices = np.rint(np.linspace(0, last, count)).astype(int).tolist()
    if len(set(indices)) != count:
        raise ValueError("Amostra contém frames repetidos")
    return indices


def preprocess_bgr(image: np.ndarray, config: dict, transform) -> torch.Tensor:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = config["crop_xyxy"]
    cropped = image[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
    if cropped.size == 0:
        raise ValueError("Recorte vazio")
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    # O preset oficial usa PIL: resize bilinear proporcional, center crop, [0,1] e Normalize.
    return transform(Image.fromarray(rgb))


def frozen_model(model: nn.Module, device: torch.device) -> nn.Module:
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_extractor(weights_dir: Path, device: torch.device):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(weights_dir), progress=False,
                                               check_hash=True, map_location="cpu")
    path = weights_dir / weights.url.rsplit("/", 1)[-1]
    digest = file_hash(path)
    if not digest.startswith("047dcff4"):
        raise ValueError("Hash dos pesos oficiais difere do prefixo publicado")
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(state)
    model.classifier = nn.Identity()
    model = frozen_model(model, device)
    transform = weights.transforms()
    info = {"architecture": "mobilenet_v3_small", "weights": str(weights), "url": weights.url,
            "weights_file": str(path.resolve()), "weights_sha256": digest,
            "embedding": "flatten(avgpool(features(image)))", "embedding_dim": EMBEDDING_DIM,
            "resize_short_edge": transform.resize_size, "center_crop": transform.crop_size,
            "mean": transform.mean, "std": transform.std, "interpolation": str(transform.interpolation),
            "antialias": transform.antialias, "padding": "none", "color": "RGB", "dtype": "float32",
            "extractor_frozen": True, "eval_mode": True, "gradients": False}
    return model, transform, info


def visual_identity(config: dict, weights_info: dict, device: torch.device) -> dict:
    return {"algorithm": "mobilenet-small-frames-v1", "parameters": config,
            "weights": weights_info, "source_sha256": file_hash(Path(__file__)),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__,
            "numpy": np.__version__, "device_type": device.type, "tf32": False, "aggregation": "temporal_mean"}


def decode_visual(root: Path, relative_path: str, config: dict, transform) -> dict:
    started, cap = time.perf_counter(), None
    details = {"sampled_indices": [], "sampled_times_seconds": []}
    try:
        cap = cv2.VideoCapture(str(video_path(root, relative_path)), cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_N_THREADS, 1])
        if not cap.isOpened():
            raise ValueError("Não foi possível abrir o vídeo")
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count < 1:
            raise ValueError("Contagem de frames inválida")
        indices = sample_visual_indices(fps, round(count), config["frames_per_video"], config["window_seconds"])
        h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        details.update({"fps": fps, "frame_count": round(count), "width": w, "height": h, "requested_indices": indices})
        selected, tensors = set(indices), []
        for index in range(indices[-1]+1):
            if not cap.grab():
                raise ValueError(f"Falha de grab/decodificação no frame {index}")
            if index not in selected:
                continue
            ok, image = cap.retrieve()
            if not ok or image is None or image.size == 0:
                raise ValueError(f"Falha de retrieve no frame {index}")
            if image.shape[:2] != (round(h), round(w)) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-index) > .5:
                raise ValueError(f"Resolução/posição inesperada no frame {index}")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if (not math.isfinite(timestamp) or timestamp >= config["window_seconds"] + 1e-6
                    or abs(timestamp-index/fps) > max(1.5/fps, .02)
                    or (details["sampled_times_seconds"] and timestamp <= details["sampled_times_seconds"][-1])):
                raise ValueError(f"Timestamp incompatível no frame {index}: {timestamp}")
            tensors.append(preprocess_bgr(image, config, transform))
            details["sampled_indices"].append(index)
            details["sampled_times_seconds"].append(timestamp)
        frames = torch.stack(tensors)
        if len(frames) != config["frames_per_video"] or not torch.isfinite(frames).all():
            raise ValueError("Frames incompletos ou não finitos")
        return {"ok": True, "frames": frames, "details": details, "error": "", "seconds": time.perf_counter()-started}
    except (OSError, ValueError, cv2.error) as error:
        return {"ok": False, "frames": None, "details": details, "error": str(error), "seconds": time.perf_counter()-started}
    finally:
        if cap is not None:
            cap.release()


def cache_signature(root: Path, relative_path: str, config_id: str) -> dict:
    stat = video_path(root, relative_path).stat()
    return {"root": str(root), "relative_path": relative_path, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "config_id": config_id}


def load_embedding_cache(cache: Path, signature: dict, count: int):
    path = cache / signature["config_id"] / f"{identity_hash(signature)}.npz"
    if not path.exists():
        return None, ""
    try:
        with np.load(path, allow_pickle=False) as entry:
            meta = json.loads(str(entry["metadata"].item()))
            embeddings = entry["embeddings"]
        if (meta["signature"] != signature or embeddings.shape != (count, EMBEDDING_DIM)
                or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all()):
            raise ValueError("Cache de embeddings incompatível")
        return {"embeddings": embeddings, "details": meta["details"], "original_seconds": meta["seconds"]}, ""
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"Cache inválido, recalcular: {error}"


def save_embedding_cache(cache: Path, signature: dict, embeddings: np.ndarray, details: dict, seconds: float) -> None:
    path = cache / signature["config_id"] / f"{identity_hash(signature)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings.astype(np.float32),
                            metadata=np.asarray(json.dumps({"signature": signature, "details": details, "seconds": seconds})))
    temporary.replace(path)


@torch.inference_mode()
def encode_frames(model: nn.Module, frames: torch.Tensor, device: torch.device) -> np.ndarray:
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Extrator deve permanecer congelado e em eval, inclusive BatchNorm")
    embeddings = model(frames.to(device))
    if embeddings.requires_grad or embeddings.shape[1] != EMBEDDING_DIM or not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings inválidos")
    return embeddings.cpu().numpy().astype(np.float32)
