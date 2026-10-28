"""Causal fish associations with short occlusion memory and visual descriptors.

Appearance here is a crop descriptor, not a trained re-identification model.
Predictions are association hypotheses only: returned boxes always come from the
current detector, and missing observations never become displayed fish.
"""

import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .dynamic_tracking import center, iou


DEFAULTS = {
    'max_gap_seconds': 1.0,
    'trail_seconds': .5,
    'confirmation_frames': 3,
    'min_speed_observations': 6,
    'min_speed_history_seconds': .2,
    'speed_window_seconds': .24,
    'max_area_ratio': 3.0,
    'max_center_speed_diagonals_per_second': 8.0,
    'occlusion_overlap_threshold': .35,
    'max_prediction_seconds': .75,
    'max_prediction_distance_diagonals': .65,
    'prediction_uncertainty_diagonals_per_second': .65,
    'max_recovery_distance_without_appearance': .4,
    'appearance_max_distance': .72,
    'recovery_appearance_max_distance': .52,
    'appearance_weight': .65,
    'appearance_update_alpha': .10,
    'min_assignment_margin': .10,
}


def _histogram(values, channels, bins, ranges, mask):
    hist = cv2.calcHist([values], channels, mask, bins, ranges).ravel().astype(float)
    total = hist.sum()
    return np.sqrt(hist / total) if total else np.zeros_like(hist)


def appearance_descriptor(image, box):
    """Unit crop signature from central HSV distributions and coarse texture."""
    if image is None:
        return None
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Appearance requires an original uint8 BGR image')
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = cv2.resize(image[y1:y2, x1:x2], (64, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros((32, 64), np.uint8)
    cv2.ellipse(mask, (32, 16), (27, 13), 0, 0, 360, 255, -1)
    parts = [np.sqrt(.75) * _histogram(hsv, [0, 1], [12, 8], [0, 180, 0, 256], mask),
             np.sqrt(.10) * _histogram(hsv, [2], [16], [0, 256], mask)]
    # Top/bottom color provides a weak spatial cue without making left/right
    # orientation an identity (a fish can turn around).
    for start, stop in ((0, 16), (16, 32)):
        parts.append(np.sqrt(.05) * _histogram(hsv[start:stop], [0, 1], [12, 8],
                                              [0, 180, 0, 256], mask[start:stop]))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))
    parts.append(np.sqrt(.05) * _histogram(texture, [0], [8], [0, 256], mask))
    descriptor = np.concatenate(parts)
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm else None


def appearance_distance(first, second):
    if first is None or second is None:
        return None
    return float(np.clip(np.linalg.norm(first - second) / np.sqrt(2), 0, 1))


class OcclusionAwareTracker:
    def __init__(self, config):
        self.config = {**DEFAULTS, **config}
        cfg = self.config
        for name in ('max_gap_seconds', 'trail_seconds', 'max_area_ratio',
                     'max_center_speed_diagonals_per_second', 'max_prediction_seconds',
                     'max_prediction_distance_diagonals', 'max_recovery_distance_without_appearance',
                     'min_speed_history_seconds', 'speed_window_seconds'):
            if not math.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError('Invalid tracking parameter: ' + name)
        if cfg['max_area_ratio'] < 1:
            raise ValueError('Invalid tracking parameter: max_area_ratio')
        uncertainty = cfg['prediction_uncertainty_diagonals_per_second']
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError('Invalid tracking parameter: prediction_uncertainty_diagonals_per_second')
        for name in ('confirmation_frames', 'min_speed_observations'):
            if (not isinstance(cfg[name], (int, np.integer)) or isinstance(cfg[name], bool)
                    or cfg[name] < 1):
                raise ValueError('Invalid tracking parameter: ' + name)
        for name in ('occlusion_overlap_threshold', 'appearance_max_distance',
                     'recovery_appearance_max_distance', 'appearance_weight',
                     'appearance_update_alpha', 'min_assignment_margin'):
            if not math.isfinite(cfg[name]) or not 0 <= cfg[name] <= 1:
                raise ValueError('Invalid tracking parameter: ' + name)
        self.reset()

    def reset(self):
        self.next_id = 1
        self.clear_history()

    def clear_history(self):
        """Forget uncertain associations after errors without recycling IDs."""
        self.tracks, self.suspended_ids = {}, []
        self.last_timestamp = self.last_frame = None

    def _prediction(self, track, timestamp):
        elapsed = min(timestamp - track['anchor_time'], self.config['max_prediction_seconds'])
        velocity = np.median(track['velocities'][-5:], axis=0) if track['velocities'] else np.zeros(2)
        return track['anchor_box'] + np.tile(velocity * elapsed, 2)

    def _cost(self, track, detection, descriptor, timestamp, frame_index):
        cfg = self.config
        box = np.asarray(detection['xyxy'], dtype=float)
        predicted = self._prediction(track, timestamp)
        old_size = track['anchor_box'][2:] - track['anchor_box'][:2]
        size = box[2:] - box[:2]
        area_ratio = float(np.prod(size) / np.prod(old_size))
        if not 1 / cfg['max_area_ratio'] <= area_ratio <= cfg['max_area_ratio']:
            return None
        diagonal = max(1., math.sqrt(np.linalg.norm(old_size) * np.linalg.norm(size)))
        distance = float(np.linalg.norm(center(box) - center(predicted)) / diagonal)
        dt = timestamp - track['time']
        rate = np.linalg.norm(center(box) - center(track['box'])) / diagonal / dt
        if rate > cfg['max_center_speed_diagonals_per_second']:
            return None
        gap = frame_index != track['frame'] + 1
        age = min(timestamp - track['anchor_time'], cfg['max_gap_seconds'])
        radius = cfg['max_prediction_distance_diagonals'] + cfg['prediction_uncertainty_diagonals_per_second'] * age
        appearance = appearance_distance(track['appearance'], descriptor)
        limit = cfg['recovery_appearance_max_distance'] if gap else cfg['appearance_max_distance']
        if appearance is not None and appearance > limit:
            return None
        if appearance is None and gap:
            radius = min(radius, cfg['max_recovery_distance_without_appearance'])
        if distance > radius:
            return None
        geometry = .75 * distance / radius + .15 * (1 - iou(box, predicted))
        geometry += .10 * abs(math.log(area_ratio)) / math.log(max(cfg['max_area_ratio'], 1.0001))
        weight = cfg['appearance_weight'] if appearance is not None else 0.
        cost = (1 - weight) * geometry + weight * (appearance or 0.)
        if appearance is None:
            cost += .08
        return float(cost), appearance, gap

    def update(self, detections, timestamp, frame_index, image=None):
        if (not math.isfinite(timestamp) or timestamp < 0
                or (self.last_timestamp is not None and timestamp <= self.last_timestamp)
                or not isinstance(frame_index, (int, np.integer)) or frame_index < 0
                or (self.last_frame is not None and frame_index <= self.last_frame)):
            raise ValueError('Timestamp/frame invalid for tracking')
        boxes = [np.asarray(d['xyxy'], dtype=float) for d in detections]
        if any(box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]) for box in boxes):
            raise ValueError('Invalid detector box')
        descriptors = [appearance_descriptor(image, box) for box in boxes]
        cfg = self.config
        self.tracks = {key: track for key, track in self.tracks.items()
                       if timestamp - track['time'] <= cfg['max_gap_seconds'] + 1e-9}
        self.last_timestamp, self.last_frame = timestamp, int(frame_index)
        dirty = [False] * len(boxes)
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                intersection = np.prod(np.maximum(0, np.minimum(boxes[a][2:], boxes[b][2:])
                                                   - np.maximum(boxes[a][:2], boxes[b][:2])))
                minimum = min(np.prod(boxes[a][2:] - boxes[a][:2]), np.prod(boxes[b][2:] - boxes[b][:2]))
                if intersection / minimum >= cfg['occlusion_overlap_threshold']:
                    dirty[a] = dirty[b] = True
        established = [key for key, track in self.tracks.items()
                       if track['clean_observations'] >= cfg['confirmation_frames']]
        # One box covering multiple known predictions may be a merged observation.
        for number, box in enumerate(boxes):
            covers = sum(np.all(center(self._prediction(self.tracks[key], timestamp)) >= box[:2])
                         and np.all(center(self._prediction(self.tracks[key], timestamp)) <= box[2:])
                         for key in established)
            dirty[number] |= covers > 1
        matches = {}
        # Established identities get first refusal. Temporary boxes created during
        # an ambiguous overlap must not steal the recovery of those identities.
        groups = (established, [key for key in self.tracks if key not in established])
        for keys in groups:
            columns = [i for i in range(len(detections)) if i not in matches]
            if not keys or not columns:
                continue
            cost = np.full((len(keys), len(columns)), 1e6)
            details = {}
            for row, key in enumerate(keys):
                for col, number in enumerate(columns):
                    candidate = self._cost(self.tracks[key], detections[number], descriptors[number], timestamp, frame_index)
                    if candidate is not None:
                        cost[row, col] = candidate[0]
                        details[row, col] = candidate
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                value = cost[row, col]
                if value >= 1e6:
                    continue
                alternatives = np.concatenate((np.delete(cost[row], col), np.delete(cost[:, col], row)))
                margin = float(alternatives.min() - value) if alternatives.size else 1e6
                if margin < cfg['min_assignment_margin']:
                    continue
                number, key = columns[col], keys[row]
                matches[number] = (key, float(value), min(margin, 1e6), details[row, col][1])
        observations = []
        observed_ids = set()
        for number, detection in enumerate(detections):
            box, position = boxes[number], center(boxes[number])
            match = matches.get(number)
            old = self.tracks[match[0]] if match else None
            if match:
                ident, cost, margin, distance = match
            else:
                ident = self.next_id
                self.next_id += 1
                cost = margin = distance = None
            consecutive = old is not None and old['frame'] == frame_index - 1
            clean = not dirty[number]
            clean_consecutive = consecutive and clean and old['clean']
            streak = old['streak'] + 1 if consecutive else 1
            clean_streak = old['clean_streak'] + 1 if clean_consecutive else int(clean)
            total = old['observations'] + 1 if old else 1
            clean_total = (old['clean_observations'] if old else 0) + int(clean)
            points = [p for p in old['points'] if timestamp - p[0] <= cfg['trail_seconds']] if clean_consecutive else []
            if clean:
                points.append([float(timestamp), *position.tolist()])
            velocities = list(old['velocities']) if old else []
            if clean_consecutive:
                velocities = (velocities + [(position - center(old['box'])) / (timestamp - old['time'])])[-5:]
            descriptor = old['appearance'] if old else None
            if clean and descriptors[number] is not None:
                if descriptor is None:
                    descriptor = descriptors[number]
                else:
                    alpha = cfg['appearance_update_alpha']
                    descriptor = (1 - alpha) * descriptor + alpha * descriptors[number]
                    descriptor /= np.linalg.norm(descriptor)
            anchor_box = box.copy() if clean or old is None else old['anchor_box']
            anchor_time = timestamp if clean or old is None else old['anchor_time']
            confirmed = streak >= cfg['confirmation_frames']
            speed = None
            recent = [p for p in points if timestamp - p[0] <= cfg['speed_window_seconds'] + 1e-9]
            if (confirmed and clean_streak >= cfg['min_speed_observations']
                    and len(recent) >= cfg['min_speed_observations']
                    and timestamp - recent[0][0] >= cfg['min_speed_history_seconds'] - 1e-9):
                speed = float(np.linalg.norm(position - np.asarray(recent[0][1:])) / (timestamp - recent[0][0]))
            self.tracks[ident] = {'box': box.copy(), 'time': timestamp, 'frame': int(frame_index),
                'anchor_box': anchor_box, 'anchor_time': anchor_time, 'appearance': descriptor,
                'streak': streak, 'clean_streak': clean_streak, 'clean': clean,
                'observations': total, 'clean_observations': clean_total,
                'points': points, 'velocities': velocities}
            observed_ids.add(ident)
            status = ('ambiguous' if old is None and dirty[number] else 'tentative')
            if old is not None:
                status = 'recovered' if not consecutive else 'occluded_observed' if not clean else 'confirmed' if confirmed else 'tentative'
            observations.append({**detection, 'track_id': ident, 'track_observations': total,
                'consecutive_observations': streak, 'track_confirmed': confirmed,
                'tracking_status': status, 'association_cost': cost, 'association_margin': margin,
                'appearance_distance': distance, 'appearance_updated': clean and descriptors[number] is not None,
                'occlusion_observed': not clean, 'recovered_after_gap': old is not None and not consecutive,
                'observed_current_frame': True, 'position_predicted_only': False,
                'center_xy': position.tolist(), 'speed_source_px_per_second': speed,
                'speed_basis': 'net displacement over consecutive clean detector observations only',
                'trail_txy': points if confirmed and clean_streak >= cfg['confirmation_frames'] else []})
        self.suspended_ids = sorted(set(self.tracks) - observed_ids)
        return observations
