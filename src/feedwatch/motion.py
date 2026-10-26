"""Características de movimento e cache por configuração/arquivo; sem rótulos na entrada."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .data import video_path, write_json

ALGORITHM = "motion-v1"
PAIR_NAMES = [f"{family}_{stat}" for family in ("diff", "flow")
              for stat in ("mean", "std", "p50", "p90", "p99", "active_pixels", "active_regions")]
FEATURE_NAMES = [f"{name}_temporal_{stat}" for name in PAIR_NAMES for stat in ("mean", "std", "p90")]


def extraction_identity(config: dict) -> dict:
    return {"algorithm": ALGORITHM, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "opencv": cv2.__version__, "numpy": np.__version__, "parameters": config,
            "feature_names": FEATURE_NAMES}


def identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config: dict) -> None:
    if not 0 < config["sample_hz"] <= 30 or not 0 < config["window_seconds"] <= 60:
        raise ValueError("Use sample_hz entre 0 e 30 e window_seconds entre 0 e 60")
    if min(config["resize_width"], config["resize_height"]) < 16:
        raise ValueError("Resolução de extração deve ser pelo menos 16×16")
    x1, y1, x2, y2 = config["crop_xyxy"]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("crop_xyxy deve definir uma região normalizada válida")
    if config["blur_kernel"] < 1 or config["blur_kernel"] % 2 != 1:
        raise ValueError("blur_kernel deve ser positivo e ímpar")
    if not 0 <= config["difference_threshold"] <= 1 or config["flow_threshold_diagonals_per_second"] <= 0:
        raise ValueError("Limiares de movimento inválidos")
    if not 1 <= config["grid_size"] <= min(config["resize_width"], config["resize_height"]):
        raise ValueError("grid_size inválido")
    if not 0 <= config["region_active_fraction"] <= 1:
        raise ValueError("region_active_fraction deve estar em [0, 1]")


def sample_indices(fps: float, frame_count: int, config: dict) -> list[int]:
    if not math.isfinite(fps) or fps < config["sample_hz"] or frame_count < 1:
        raise ValueError("FPS/contagem inválidos ou FPS inferior à frequência de amostragem")
    end = min(config["window_seconds"], frame_count / fps)
    times = np.arange(0, end - 1e-9, 1 / config["sample_hz"])
    indices = np.rint(times * fps).astype(int)
    indices = indices[indices < frame_count].tolist()
    if len(indices) < 3 or len(set(indices)) != len(indices):
        raise ValueError("Vídeo não fornece ao menos três frames distintos na grade temporal")
    return indices


def prepare_frame(image: np.ndarray, config: dict) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = config["crop_xyxy"]
    image = image[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
    if image.size == 0:
        raise ValueError("Recorte de imagem vazio")
    gray = cv2.cvtColor(cv2.resize(image, (config["resize_width"], config["resize_height"]),
                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (config["blur_kernel"], config["blur_kernel"]), 0)


def spatial_stats(values: np.ndarray, threshold: float, config: dict) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("Valores não finitos no cálculo de movimento")
    active = values > threshold
    grid = config["grid_size"]
    regions = [block.mean() > config["region_active_fraction"]
               for row in np.array_split(active, grid, axis=0)
               for block in np.array_split(row, grid, axis=1)]
    return [float(values.mean()), float(values.std()), *np.percentile(values, [50, 90, 99]).tolist(),
            float(active.mean()), float(np.mean(regions))]


def pair_features(previous: np.ndarray, current: np.ndarray, dt: float, config: dict) -> list[float]:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Intervalo temporal inválido")
    difference = cv2.absdiff(previous, current).astype(np.float32) / 255.0
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **config["farneback"])
    # Magnitude em frações da diagonal da imagem por segundo, não pixels absolutos.
    magnitude = np.hypot(flow[..., 0], flow[..., 1]) / (dt * math.hypot(*previous.shape))
    return (spatial_stats(difference, config["difference_threshold"], config)
            + spatial_stats(magnitude, config["flow_threshold_diagonals_per_second"], config))


def aggregate_pairs(pairs: list[list[float]]) -> dict[str, float]:
    values = np.asarray(pairs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(PAIR_NAMES) or not np.isfinite(values).all():
        raise ValueError("Pares de movimento insuficientes ou inválidos")
    features = np.stack([values.mean(axis=0), values.std(axis=0), np.percentile(values, 90, axis=0)], axis=1).ravel()
    return dict(zip(FEATURE_NAMES, features.tolist()))


def extract_video(root: Path, relative_path: str, config: dict) -> dict:
    """Erro de abertura/decodificação invalida o vídeo inteiro, sem vetor zero."""
    started, cap = time.perf_counter(), None
    details = {"sampled_indices": [], "sampled_times_seconds": [], "frames_decoded": 0}
    try:
        cap = cv2.VideoCapture(str(video_path(root, relative_path)), cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_N_THREADS, 1,
        ])
        if not cap.isOpened():
            raise ValueError("Não foi possível abrir o vídeo")
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count < 1:
            raise ValueError("Contagem de frames inválida")
        indices = sample_indices(fps, round(count), config)
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        details.update({"fps": fps, "frame_count": round(count), "width": width, "height": height,
                        "requested_indices": indices, "backend": cap.getBackendName()})
        selected, previous, previous_time, pairs = set(indices), None, None, []
        for index in range(indices[-1] + 1):
            if not cap.grab():
                raise ValueError(f"Falha de decodificação/grab no frame {index}")
            if index not in selected:
                continue
            success, image = cap.retrieve()
            if not success or image is None or image.size == 0:
                raise ValueError(f"Falha de retrieve no frame {index}")
            if image.shape[:2] != (round(height), round(width)):
                raise ValueError(f"Resolução mudou no frame {index}")
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 - index) > 0.5:
                raise ValueError(f"Posição decodificada diverge do frame {index}")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or abs(timestamp - index / fps) > max(1.5 / fps, 0.02):
                raise ValueError(f"Timestamp incompatível com amostragem CFR no frame {index}: {timestamp}")
            gray = prepare_frame(image, config)
            details["sampled_indices"].append(index)
            details["sampled_times_seconds"].append(timestamp)
            details["frames_decoded"] += 1
            if previous is not None:
                pairs.append(pair_features(previous, gray, timestamp - previous_time, config))
            previous, previous_time = gray, timestamp
        features = aggregate_pairs(pairs)
        return {"ok": True, "features": features, "details": details, "error": "",
                "extraction_seconds": time.perf_counter() - started}
    except (OSError, ValueError, cv2.error) as error:
        return {"ok": False, "features": None, "details": details, "error": str(error),
                "extraction_seconds": time.perf_counter() - started}
    finally:
        if cap is not None:
            cap.release()


def cached_extract(root: Path, relative_path: str, config: dict, cache_dir: Path, config_id: str) -> dict:
    started = time.perf_counter()
    cache_warning = ""
    try:
        stat = video_path(root, relative_path).stat()
        signature = {"root": str(root), "relative_path": relative_path, "size_bytes": stat.st_size,
                     "mtime_ns": stat.st_mtime_ns, "config_id": config_id}
        key = identity_hash(signature)
        cache_file = cache_dir / config_id / f"{key}.json"
        if cache_file.exists():
            try:
                entry = json.loads(cache_file.read_text(encoding="utf-8"))
                features = entry["result"]["features"]
                if (entry["signature"] != signature or not entry["result"]["ok"]
                        or set(features) != set(FEATURE_NAMES)
                        or not np.isfinite([features[n] for n in FEATURE_NAMES]).all()):
                    raise ValueError("Assinatura ou características do cache inválidas")
                return {**entry["result"], "cache_hit": True, "cache_warning": "", "source_signature": signature,
                        "elapsed_seconds": time.perf_counter() - started}
            except (OSError, ValueError, KeyError, TypeError) as error:
                cache_warning = f"Cache inválido, extraído novamente: {error}"
        result = extract_video(root, relative_path, config)
        after = video_path(root, relative_path).stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Arquivo mudou durante a extração")
        if result["ok"]:
            temporary = cache_file.with_suffix(".tmp")
            write_json(temporary, {"signature": signature, "result": result})
            temporary.replace(cache_file)
        return {**result, "cache_hit": False, "cache_warning": cache_warning, "source_signature": signature,
                "elapsed_seconds": time.perf_counter() - started}
    except (OSError, ValueError) as error:
        return {"ok": False, "features": None, "details": {}, "error": str(error),
                "extraction_seconds": time.perf_counter() - started, "cache_hit": False,
                "cache_warning": cache_warning, "source_signature": None,
                "elapsed_seconds": time.perf_counter() - started}
