"""Causal association of detector observations; IDs are temporary, not fish identities."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


class ObservationTracker:
    def __init__(self, config: dict):
        self.config = config
        self.reset()

    def reset(self):
        self.tracks, self.next_id, self.last_timestamp = {}, 1, None

    def update(self, detections: list[dict], timestamp: float, frame_index: int) -> list[dict]:
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Timestamp inválido para rastreamento; reiniciar, sem velocidade inventada")
        self.last_timestamp = timestamp
        cfg = self.config
        self.tracks = {key: value for key, value in self.tracks.items()
                       if timestamp-value["time"] <= cfg["max_gap_seconds"]+1e-9}
        keys = sorted(self.tracks)
        cost = np.full((len(keys), len(detections)), 1e6)
        for row, key in enumerate(keys):
            track = self.tracks[key]
            displacement = track["velocity"]*(timestamp-track["time"])
            predicted_box = np.asarray(track["box"])+np.tile(displacement, 2)
            for col, det in enumerate(detections):
                box = det["xyxy"]
                diagonal = max(np.linalg.norm(np.asarray(box[2:])-box[:2]),
                               np.linalg.norm(predicted_box[2:]-predicted_box[:2]), 1.)
                distance = np.linalg.norm(center(box)-center(predicted_box))/diagonal
                overlap = iou(predicted_box, box)
                if overlap >= cfg["min_iou"] or distance <= cfg["max_center_distance_in_box_diagonals"]:
                    cost[row, col] = (1-overlap)+distance
        matches = {}
        if cost.size:
            rr, cc = linear_sum_assignment(cost)
            matches = {int(c): keys[int(r)] for r, c in zip(rr, cc) if cost[r, c] < 1e6}
        observations = []
        for number, detection in enumerate(detections):
            box, position = detection["xyxy"], center(detection["xyxy"])
            ident = matches.get(number)
            speed, trail, velocity, count = None, [], np.zeros(2), 1
            if ident is None:
                ident = self.next_id
                self.next_id += 1
            else:
                old = self.tracks[ident]
                count = old["observations"]+1
                # Only consecutive observations define displayed speed/trail. Never bridge missing detections.
                if frame_index == old["frame"]+1:
                    velocity = (position-center(old["box"]))/ (timestamp-old["time"])
                    speed = float(np.linalg.norm(velocity))
                    trail = [point for point in old["trail"] if timestamp-point[0] <= cfg["trail_seconds"]]
            trail.append([timestamp, float(position[0]), float(position[1])])
            self.tracks[ident] = {"box": box, "time": timestamp, "frame": frame_index,
                "velocity": velocity, "trail": trail, "observations": count}
            observations.append({**detection, "track_id": ident, "track_observations": count,
                "center_xy": position.tolist(), "speed_source_px_per_second": speed,
                "trail_txy": trail})
        # Unmatched tracks are retained only for association; no ghost boxes or counts.
        return observations
