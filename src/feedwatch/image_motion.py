"""Optical flow measures apparent IMAGE movement, including water and reflections."""

import cv2
import numpy as np


class ImageMotion:
    def __init__(self, config: dict):
        self.config = config
        self.previous, self.previous_time, self.shape = None, None, None

    def update(self, image: np.ndarray, timestamp: float):
        cfg = self.config
        height, width = image.shape[:2]
        ratio = min(1., cfg["analysis_long_edge"]/max(width, height))
        small = cv2.resize(image, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if not np.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            self.previous, self.previous_time, self.shape = None, None, None
            raise ValueError("Timestamp inválido para fluxo óptico")
        if self.shape is not None and self.shape != image.shape:
            raise ValueError("Resolução mudou no fluxo óptico")
        previous, previous_time = self.previous, self.previous_time
        self.previous, self.previous_time, self.shape = gray, timestamp, image.shape
        if previous is None:
            return {"status": "initializing", "mean_source_px_per_second": None,
                "p90_source_px_per_second": None, "moving_fraction": None, "delta_seconds": None}, None
        delta = timestamp-previous_time
        flow = cv2.calcOpticalFlowFarneback(previous, gray, None, cfg["pyr_scale"], cfg["levels"],
            cfg["winsize"], cfg["iterations"], cfg["poly_n"], cfg["poly_sigma"], 0)
        flow[:, :, 0] *= width/gray.shape[1]/delta
        flow[:, :, 1] *= height/gray.shape[0]/delta
        magnitude = np.linalg.norm(flow, axis=2)
        if not np.isfinite(magnitude).all():
            raise ValueError("Fluxo óptico inválido; não substituir por movimento zero")
        return {"status": "ok", "mean_source_px_per_second": float(magnitude.mean()),
            "p90_source_px_per_second": float(np.percentile(magnitude, 90)),
            "moving_fraction": float(np.mean(magnitude > cfg["active_threshold_source_px_per_second"])),
            "delta_seconds": delta}, flow
