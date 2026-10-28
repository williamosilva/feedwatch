"""Re-track saved detections and render them over the verified original recording."""

import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
from torchvision.models import MobileNet_V3_Small_Weights

from .association_tracking import create_tracker
from .data import file_hash, utc_now, write_json
from .dynamic_video import visual_timestamp
from .simple_overlay import SimpleOverlay
from .video_export import H264Writer, media_info, verify_export
from .video_inference import display_state, inspect_video, open_capture


DETECTION_KEYS = ('xyxy', 'score', 'detector_class_id', 'detector_class')
CSV_FIELDS = ('frame_index', 'timestamp_seconds', 'food_status', 'food_window_id',
              'food_class', 'detector_status', 'detected_count', 'motion_status',
              'motion_mean_source_px_per_second', 'motion_p90_source_px_per_second',
              'moving_fraction', 'error')


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _geometry_predictor(run):
    """Construct only a preprocessing preset. No network or checkpoint is loaded."""
    identity = run['food_model']['extractor_identity']
    weights = identity['weights']
    preset = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    expected = {'architecture': 'mobilenet_v3_small',
                'resize_short_edge': preset.resize_size, 'center_crop': preset.crop_size,
                'interpolation': str(preset.interpolation), 'antialias': preset.antialias}
    if any(weights.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved preprocessing geometry differs from the installed preset')
    return SimpleNamespace(extraction=deepcopy(identity['parameters']), transform=preset)


def _detections(row, width, height):
    values = []
    for observation in row.get('detections', []):
        if any(key not in observation for key in DETECTION_KEYS):
            raise ValueError('Saved detector observation is incomplete')
        detection = {key: deepcopy(observation[key]) for key in DETECTION_KEYS}
        box, score = detection['xyxy'], detection['score']
        if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                or not 0 <= box[0] < box[2] <= width
                or not 0 <= box[1] < box[3] <= height
                or not math.isfinite(score) or not 0 <= score <= 1
                or detection['detector_class_id'] != 0 or detection['detector_class'] != 'Fish'):
            raise ValueError('Saved detector observation is invalid')
        values.append(detection)
    if row['detector_status'] == 'ok':
        if row['detected_count'] != len(values):
            raise ValueError('Saved detector count differs from the observations')
    elif values or row['detected_count'] is not None:
        raise ValueError('Unavailable detector frames cannot contain observations or a numeric count')
    return values


def _configuration(value, saved):
    if isinstance(value, dict):
        config, origin = deepcopy(value), None
    else:
        path = Path(value).resolve()
        config = json.loads(path.read_text(encoding='utf-8'))
        origin = {'path': str(path), 'sha256': file_hash(path)}
    if config.get('schema') != 1:
        raise ValueError('Unsupported video configuration schema')
    original = deepcopy(saved['dynamic_config'])
    # run_simple records this effective default even when omitted in the file.
    original['presentation'].setdefault('show_speed', False)
    config['presentation'].setdefault('show_speed', False)
    left, right = deepcopy(original), deepcopy(config)
    left.pop('tracker', None); right.pop('tracker', None)
    if left != right:
        raise ValueError('A tracking replay may change only the tracker configuration')
    return config, origin


def replay_tracking_video(source, saved_analysis, output, config, *, progress=None):
    """Reassign IDs without rerunning detection, feeding inference or image motion.

    ``saved_analysis`` contains a simple, non-diagnostic run.json and frames.jsonl.
    ``config`` is a config path or dictionary, differing only in its tracker section.
    ``output`` must not exist. Every cached box/score/class and feeding window is
    preserved, including unavailable observations. Only tracking fields and drawing
    change. A failure leaves diagnostic files but no published annotated.mp4.
    """
    source, saved_analysis, output = map(lambda path: Path(path).resolve(),
                                         (source, saved_analysis, output))
    if output.exists():
        raise ValueError('Replay output already exists; choose a fresh directory')
    if source.is_relative_to(output) or saved_analysis.is_relative_to(output):
        raise ValueError('Replay output cannot contain the original inputs')
    run_path, frames_path = saved_analysis / 'run.json', saved_analysis / 'frames.jsonl'
    run_bytes, frames_bytes = run_path.read_bytes(), frames_path.read_bytes()
    run_identity, frames_identity = map(lambda value: hashlib.sha256(value).hexdigest(),
                                        (run_bytes, frames_bytes))
    saved = json.loads(run_bytes)
    if saved.get('presentation_profile') != 'simple' or saved.get('diagnostic', False):
        raise ValueError('Tracking replay requires a non-diagnostic simple analysis')
    signature = (source.stat().st_size, source.stat().st_mtime_ns, file_hash(source))
    if signature[2] != saved['input_signature']['sha256']:
        raise ValueError('Original recording differs from the saved input hash')
    rows = [json.loads(line) for line in frames_bytes.decode('utf-8').splitlines()]
    metadata = saved['metadata']
    if len(rows) != metadata['frame_count']:
        raise ValueError('Cached frame count differs from the saved video')
    config, config_origin = _configuration(config, saved)
    inputs = []
    for index, row in enumerate(rows):
        if row['frame_index'] != index:
            raise ValueError('Cached frames must be consecutive from frame zero')
        timestamp = row['timestamp_seconds']
        if timestamp is not None and (not math.isfinite(timestamp)
                or abs(timestamp-index/metadata['fps']) > max(1.5/metadata['fps'], .02)):
            raise ValueError('Cached timestamp differs from the original frame')
        if timestamp is None and row['detector_status'] == 'ok':
            raise ValueError('A valid detector frame requires a compatible timestamp')
        inputs.append(_detections(row, metadata['width'], metadata['height']))
    windows_identity = _identity(saved['windows'])
    detections_identity = _identity(inputs)
    cap = open_capture(source)
    writer = None
    started = time.perf_counter()
    try:
        actual = inspect_video(cap)
        for key in ('width', 'height', 'frame_count'):
            if actual[key] != metadata[key]:
                raise ValueError('Original recording metadata differs from the saved analysis')
        if not math.isclose(actual['fps'], metadata['fps'], rel_tol=1e-6):
            raise ValueError('Original recording FPS differs from the saved analysis')
        renderer = SimpleOverlay(metadata['width'], metadata['height'], _geometry_predictor(saved),
                                 config, False, saved.get('source_note', ''))
        for current, original in ((renderer.roi, saved.get('food_roi_xyxy')),
                                  (renderer.crop, saved.get('food_effective_crop_xyxy'))):
            if original is not None and (len(current) != len(original)
                    or any(abs(a-b) > 1e-6 for a, b in zip(current, original))):
                raise ValueError('Saved crop geometry differs from the renderer')
        export_metadata = renderer.output_metadata(metadata)
        tracker = create_tracker(config['tracker'], metadata['fps'])
        output.mkdir(parents=True, exist_ok=False)
        source_media = media_info(source)
        pending = output / 'annotated.pending.mp4'
        writer = H264Writer(pending, source, export_metadata, source_media['has_audio'], output / 'encoder.log')
        new_rows, previous_time, tracking_seconds = [], None, 0.
        if progress: progress('render', 0, len(rows))
        with (output / 'frames.jsonl').open('w', encoding='utf-8') as details, \
                (output / 'frames.csv').open('w', encoding='utf-8', newline='') as csv_file:
            table = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            table.writeheader()
            for index, (row, detections) in enumerate(zip(rows, inputs)):
                ok, image = cap.read()
                if not ok or image is None or image.shape[:2] != (metadata['height'], metadata['width']):
                    raise ValueError(f'Original recording cannot be decoded at frame {index}')
                decoded = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                visual_time, compatible, _ = visual_timestamp(decoded, index, metadata['fps'], previous_time)
                previous_time = visual_time if compatible else previous_time
                timestamp = row['timestamp_seconds']
                if timestamp is not None and (not compatible
                        or abs(timestamp-visual_time) > max(1e-6, .5/metadata['fps'])):
                    raise ValueError(f'Original and cached clocks differ at frame {index}')
                food = display_state(saved['windows'], saved['mode'], index/metadata['fps'])
                item = food.get('window')
                expected_food = (food['status'], item['window_id'] if item else None,
                                 item['predicted_class'] if food['status'] == 'prediction' else None)
                if expected_food != (row['food_status'], row['food_window_id'], row['food_class']):
                    raise ValueError(f'Cached feeding display differs from its windows at frame {index}')
                mark = time.perf_counter()
                if timestamp is None or row['detector_status'] != 'ok':
                    if hasattr(tracker, 'clear_history'):
                        tracker.clear_history()
                    else:
                        tracker.tracks.clear()
                        tracker.last_timestamp = None
                        tracker.suspended_ids = []
                    observations = []
                else:
                    observations = tracker.update(deepcopy(detections), timestamp, index, image=image)
                tracking_seconds += time.perf_counter()-mark
                replayed = [{key: observation[key] for key in DETECTION_KEYS} for observation in observations]
                if replayed != detections:
                    raise ValueError('Tracker changed detection order, geometry, score or count')
                movement = deepcopy(row.get('image_motion', {'status': row['motion_status']}))
                annotated = renderer.render(image, food, saved['mode'], observations,
                    row['detector_status'], movement, None, timestamp if timestamp is not None else index/metadata['fps'])
                writer.write(annotated)
                if index in (0, len(rows)//2, len(rows)-1):
                    cv2.imwrite(str(output / ('poster.jpg' if index == 0 else f'frame_{index:05d}.jpg')), annotated)
                updated = {**deepcopy(row), 'detections': observations,
                           'confirmed_track_count': sum(value.get('track_confirmed', False) for value in observations),
                           'suspended_track_ids': list(getattr(tracker, 'suspended_ids', [])),
                           'layout': deepcopy(renderer.last_layout)}
                new_rows.append(updated)
                details.write(json.dumps(updated, ensure_ascii=False, allow_nan=False)+'\n')
                table.writerow({key: updated.get(key) for key in CSV_FIELDS})
                if progress: progress('render', index+1, len(rows))
            if cap.read()[0]:
                raise ValueError('Original recording contains additional frames')
    finally:
        cap.release()
        if writer is not None: writer.close()
    verification = verify_export(pending, export_metadata)
    if signature != (source.stat().st_size, source.stat().st_mtime_ns, file_hash(source)):
        raise ValueError('Original recording changed during replay')
    if run_identity != file_hash(run_path) or frames_identity != file_hash(frames_path):
        raise ValueError('Saved analysis changed during replay')
    if detections_identity != _identity([[{key: item[key] for key in DETECTION_KEYS}
                                         for item in row['detections']] for row in new_rows]):
        raise ValueError('Detector observations changed during replay')
    final = output / 'annotated.mp4'
    pending.replace(final)
    report = deepcopy(saved)
    # Do not present prior model replay or final-card verification as a new run.
    for key in ('presentation', 'reload_verification', 'gpu_peak_allocated_bytes_since_process_start',
                'reproduce_command', 'total_processing_seconds', 'effective_input_fps'):
        report.pop(key, None)
    report.update(created_at_utc=utc_now(), input_video=str(source), dynamic_config=config,
                  input_signature={'size_bytes': signature[0], 'mtime_ns': signature[1], 'sha256': signature[2]},
                  processing='Saved detections and feeding windows; tracking and overlay replayed over original frames',
                  output_video=str(final), output_sha256=file_hash(final), rendered_frames=len(new_rows),
                  export_verification=verification, source_media=source_media,
                  render_geometry={'source_width': metadata['width'], 'source_height': metadata['height'],
                      'output_width': export_metadata['width'], 'output_height': export_metadata['height'],
                      'source_image_scaled': False, 'footer_added': True},
                  training=False, realtime_claim=False, analysis_seconds=0.,
                  cached_analysis_seconds=saved.get('analysis_seconds'),
                  processing_seconds=time.perf_counter()-started)
    report['timings'] = {'detector_seconds': 0., 'tracking_seconds': tracking_seconds,
                         'motion_seconds': 0., 'render_encode_seconds': report['processing_seconds']-tracking_seconds}
    report['counters'] = {'frames_detector_ok': sum(row['detector_status'] == 'ok' for row in new_rows),
        'frames_with_detections': sum(bool(row['detections']) for row in new_rows),
        'detection_observations': sum(len(row['detections']) for row in new_rows),
        'observations_with_speed': sum(item.get('speed_source_px_per_second') is not None
                                      for row in new_rows for item in row['detections']),
        'frames_motion_ok': sum(row['motion_status'] == 'ok' for row in new_rows),
        'max_detections_in_frame': max((len(row['detections']) for row in new_rows), default=0)}
    report['effective_processing_fps'] = len(new_rows)/report['processing_seconds']
    report['tracks'] = config['tracker'].get('kind', 'conservative-v2')+'; observed boxes only; no gap trails'
    if _identity(report['windows']) != windows_identity:
        raise ValueError('Feeding windows changed during replay')
    audit = {'saved_analysis': str(saved_analysis), 'saved_run_sha256': run_identity,
             'saved_frames_sha256': frames_identity, 'source_sha256': signature[2],
             'config_source': config_origin, 'effective_config_sha256': _identity(config),
             'before_detections_sha256': detections_identity, 'after_detections_sha256': detections_identity,
             'before_windows_sha256': windows_identity, 'after_windows_sha256': _identity(report['windows']),
             'detector_inference_run': False, 'feeding_inference_run': False,
             'image_motion_recomputed': False, 'source_image_only': True,
             'frame_count_preserved': True, 'timestamps_preserved': True,
             'detection_geometry_scores_counts_preserved': True, 'feeding_windows_preserved': True,
             'previous_tracker': saved['dynamic_config']['tracker'], 'new_tracker': config['tracker'],
             'render_seconds': report['processing_seconds']}
    report['tracking_replay'] = audit
    write_json(output / 'config.json', config)
    write_json(output / 'tracking-replay.json', audit)
    write_json(output / 'analysis.json', report)
    write_json(output / 'run.json', report)
    with (output / 'windows.csv').open('w', encoding='utf-8', newline='') as handle:
        if report['windows']:
            table = csv.DictWriter(handle, fieldnames=list(report['windows'][0]))
            table.writeheader()
            for window in report['windows']:
                table.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                for key, value in window.items()})
    return report
