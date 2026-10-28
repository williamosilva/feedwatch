"""Native ByteTrack/BoT-SORT associations with strictly observed display geometry."""

import importlib.util
import math
import os
from types import SimpleNamespace

import numpy as np

from .conservative_tracking import ConservativeTracker
from .dynamic_tracking import center


class NativeAssociationTracker:
    """Use Kalman association IDs, while never rendering predicted boxes or gap trails."""

    def __init__(self, config, fps):
        if importlib.util.find_spec('lap') is None:
            raise RuntimeError('Rastreamento requer lap==0.5.13; instale requirements-dynamic-lock.txt no ambiente Python ativo.')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        args = SimpleNamespace(track_high_thresh=config.get('track_high_thresh', .15),
            track_low_thresh=config.get('track_low_thresh', .10), new_track_thresh=config.get('new_track_thresh', .15),
            track_buffer=max(1, round(config['max_gap_seconds'] * fps)),
            match_thresh=config.get('match_thresh', .8), fuse_score=config.get('fuse_score', False),
            gmc_method=config.get('gmc_method', 'none'), proximity_thresh=.5, appearance_thresh=.8,
            with_reid=False, model='auto')
        self.backend = (BOTSORT if config['kind'] == 'botsort' else BYTETracker)(args, frame_rate=30)
        self.config, self.fps = config, fps
        self.tracks, self.native_ids, self.next_id, self.last_timestamp = {}, {}, 1, None
        self.last_frame, self.suspended_ids = None, []

    def clear_history(self):
        """Drop associations after a decode error; keep displayed ID counter monotonic."""
        self.backend.reset()
        self.tracks.clear(); self.native_ids.clear()
        self.last_timestamp = self.last_frame = None

    def update(self, detections, timestamp, frame_index, image=None):
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError('Timestamp inválido para acompanhamento')
        if self.last_frame is not None and frame_index != self.last_frame + 1:
            self.clear_history()
        self.last_timestamp, self.last_frame = timestamp, frame_index
        from ultralytics.engine.results import Boxes
        values = np.array([[*item['xyxy'], item['score'], 0] for item in detections], dtype=np.float32).reshape(-1, 6)
        self.backend.update(Boxes(values, image.shape[:2] if image is not None else (1, 1)), img=image)
        # Internal tracks include not-yet-activated observations. Their geometry is never used.
        high = np.flatnonzero(values[:, 4] >= self.backend.args.track_high_thresh)
        low = np.flatnonzero((values[:, 4] > self.backend.args.track_low_thresh)
                             & (values[:, 4] < self.backend.args.track_high_thresh))
        current = {}
        for track in self.backend.tracked_stracks:
            if track.frame_id != self.backend.frame_id:
                continue
            # Ultralytics numbers detections again inside each confidence subset.
            # Recover the original row instead of attaching the high-score ID to
            # a low-score fish that happened to occupy that row in the full input.
            subset = high if track.score >= self.backend.args.track_high_thresh else low
            index = int(track.idx)
            if not 0 <= index < len(subset):
                raise ValueError('Índice de associação incompatível com o subconjunto de detecções')
            current[int(subset[index])] = track
        self.tracks = {key: track for key, track in self.tracks.items()
                       if timestamp - track['time'] <= self.config['max_gap_seconds'] + 1e-9}
        observed_ids = set()
        observations = []
        for number, detection in enumerate(detections):
            native = current.get(number)
            native_id = int(native.track_id) if native is not None else None
            if native_id is not None and native_id in self.native_ids:
                ident = self.native_ids[native_id]
            else:
                ident = self.next_id; self.next_id += 1
                if native_id is not None:
                    self.native_ids[native_id] = ident
            position = center(detection['xyxy'])
            old = self.tracks.get(ident)
            consecutive = old is not None and old['frame'] == frame_index - 1
            points = [point for point in old['points'] if timestamp - point[0] <= self.config['trail_seconds'] + 1e-9] if consecutive else []
            points.append([float(timestamp), *position.tolist()])
            streak = old['streak'] + 1 if consecutive else 1
            count = old['observations'] + 1 if old is not None else 1
            confirmed = streak >= self.config['confirmation_frames']
            recent = [point for point in points if timestamp - point[0] <= self.config['speed_window_seconds'] + 1e-9]
            speed = None
            if (confirmed and len(recent) >= self.config['min_speed_observations']
                    and timestamp - recent[0][0] >= self.config['min_speed_history_seconds'] - 1e-9):
                speed = float(np.linalg.norm(position - np.asarray(recent[0][1:])) / (timestamp - recent[0][0]))
            self.tracks[ident] = {'box': detection['xyxy'], 'time': timestamp, 'frame': frame_index,
                                  'points': points, 'streak': streak, 'observations': count}
            observed_ids.add(ident)
            observations.append({**detection, 'track_id': ident, 'track_observations': count,
                'consecutive_observations': streak, 'track_confirmed': confirmed,
                'tracking_status': 'confirmed' if confirmed else 'tentative',
                'association_cost': None, 'association_margin': None,
                'observed_current_frame': True, 'position_predicted_only': False,
                'center_xy': position.tolist(), 'speed_source_px_per_second': speed,
                'speed_basis': 'net observed center displacement / elapsed time over recent consecutive history',
                'trail_txy': points if confirmed else []})
        self.suspended_ids = sorted(set(self.tracks) - observed_ids)
        return observations


def create_tracker(config, fps):
    kind = config.get('kind', 'conservative-v2')
    if kind == 'occlusion-aware-v1':
        from .occlusion_tracking import OcclusionAwareTracker
        return OcclusionAwareTracker(config)
    if kind == 'conservative-v2':
        return ConservativeTracker(config)
    if kind in {'bytetrack', 'botsort'}:
        return NativeAssociationTracker(config, fps)
    raise ValueError('Tipo de rastreador desconhecido: ' + kind)
