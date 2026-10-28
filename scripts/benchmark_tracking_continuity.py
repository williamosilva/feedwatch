"""Association-only assisted regression on frozen original-image anchors.

Run from the workspace root with .venv/Scripts/python.exe. No model inference.
Anchors are deliberately sparse and not independent identity ground truth.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from feedwatch.association_tracking import create_tracker
from feedwatch.occlusion_tracking import DEFAULTS




def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def metrics(anchors):
    """Count changes along a target and observed transfers between targets.

    A changed ID is fragmentation, not proof of assignment to another fish.
    A shared track between two annotated fish is evidence of a wrong identity
    transfer. Unannotated fish and intermediate frames cannot be adjudicated.
    """
    previous = {}
    owners = {}
    ids_by_identity = defaultdict(set)
    changes, transfers, events = 0, 0, []
    for frame in anchors:
        for ref in frame['reference']:
            identity, track = ref['identity'], ref['track_id']
            ids_by_identity[identity].add(track)
            if identity in previous and previous[identity] != track:
                changes += 1
                events.append({'type': 'changed_id', 'frame_index': frame['frame_index'],
                               'identity': identity, 'before': previous[identity], 'after': track})
            if track in owners and owners[track] != identity:
                transfers += 1
                events.append({'type': 'cross_reference_transfer', 'frame_index': frame['frame_index'],
                               'track_id': track, 'before': owners[track], 'after': identity})
            previous[identity], owners[track] = track, identity
    return {'reference_observations': sum(len(f['reference']) for f in anchors),
            'reference_identities': len(previous), 'id_changes_along_reference': changes,
            'observed_cross_reference_id_transfers': transfers,
            'unique_ids_per_reference': {key: sorted(value) for key, value in ids_by_identity.items()},
            'events': events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=ROOT / 'evidence/tracking-references.json')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'artifacts/local-app-runs')
    parser.add_argument('--media-dir', type=Path, help='Directory containing <scene>-original.mp4 recordings')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh output directory')
    args = parser.parse_args()
    OUT, REFERENCE = args.output_dir.resolve(), args.references.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    # Baseline remains explicit even after the selected default is promoted.
    legacy = {'kind': 'conservative-v2', 'max_gap_seconds': .1, 'trail_seconds': .5,
              'min_iou': .2, 'max_innovation_diagonals': .3,
              'max_center_speed_diagonals_per_second': 8., 'max_area_ratio': 2.,
              'min_assignment_margin': .12, 'confirmation_frames': 3,
              'min_speed_observations': 6, 'min_speed_history_seconds': .2, 'speed_window_seconds': .24}
    configs = {'conservative_0p1s': legacy,
               'conservative_1s': {**legacy, 'max_gap_seconds': 1.},
               'bytetrack_1s': {**legacy, 'kind': 'bytetrack', 'max_gap_seconds': 1.},
               'botsort_1s': {**legacy, 'kind': 'botsort', 'max_gap_seconds': 1.},
               'occlusion_1s': {'kind': 'occlusion-aware-v1', **DEFAULTS}}
    results = {'schema': 1, 'reference_sha256': sha(REFERENCE),
               'annotation_status': reference['annotation_status'],
               'limitations': reference['limitations'] + [
                   'ID changes are fragmentation or reassignment, not automatically a wrong-fish transfer.',
                   'Cross-reference transfer counts only observable reuse among annotated targets; zero does not establish absence of wrong associations to other fish.',
                   'Temporal development/verification labels are descriptive partitions, not independent blinded validation.',
                   'Source detections, confidence scores, order, counts and feeding outputs are unchanged; no detector or feeding model was run.'],
               'configs': configs, 'module_sha256': {name: sha(ROOT / 'src/feedwatch' / name)
                   for name in ('occlusion_tracking.py', 'association_tracking.py', 'conservative_tracking.py')},
               'sequences': []}
    for sequence in reference['sequences']:
        run = args.runs_dir.resolve() / sequence['run']
        rows = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines()]
        trackers = {name: create_tracker(config, sequence['fps']) for name, config in configs.items()}
        anchors = {f['frame_index']: f for f in sequence['frames']}
        observed = {name: [] for name in configs}
        assignments = {name: [] for name in configs}
        last_seen = {name: {} for name in configs}
        summary = {name: {'total_detected_observations': 0, 'allocated_ids': set(),
                         'recoveries_after_missing_frame': 0} for name in configs}
        source = (args.media_dir / (sequence['scene'] + '-original.mp4') if args.media_dir else ROOT / sequence['source'])
        saved_run = json.loads((run / 'run.json').read_text(encoding='utf-8'))
        source_hash = sha(source)
        assert source_hash == saved_run['input_signature']['sha256'], 'Source differs from cached detector input'
        cap = cv2.VideoCapture(str(source))
        assert cap.isOpened(), source
        assert abs(cap.get(cv2.CAP_PROP_FPS) - sequence['fps']) < 1e-6
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == saved_run['metadata']['frame_count']
        seq_out = {'scene': sequence['scene'], 'source_sha256': source_hash,
                   'frames_sha256': sha(run / 'frames.jsonl'), 'warmup_start_frame': 0,
                   'last_frame': sequence['last_frame'], 'methods': {}}
        for frame_index, row in enumerate(rows[:sequence['last_frame'] + 1]):
            ok, image = cap.read()
            assert ok and row['frame_index'] == frame_index
            timestamp = row['timestamp_seconds']
            pure = [{key: d[key] for key in ('xyxy', 'score', 'detector_class_id', 'detector_class')}
                    for d in row['detections']]
            for name, tracker in trackers.items():
                output = tracker.update(pure, timestamp, frame_index, image=image)
                assert len(output) == len(pure)
                for old, new in zip(pure, output):
                    assert all(new[key] == old[key] for key in old)
                s = summary[name]
                s['total_detected_observations'] += len(output)
                s['allocated_ids'].update(d['track_id'] for d in output)
                for observation in output:
                    track_id = observation['track_id']
                    previous = last_seen[name].get(track_id)
                    s['recoveries_after_missing_frame'] += int(previous is not None and previous < frame_index - 1)
                    last_seen[name][track_id] = frame_index
                assignments[name].append({'frame_index': frame_index,
                    'track_ids': [d['track_id'] for d in output],
                    'recovered_detection_indices': [i for i, d in enumerate(output) if d.get('recovered_after_gap')],
                    'status': [d['tracking_status'] for d in output]})
                if frame_index in anchors:
                    anchor = anchors[frame_index]
                    refs = []
                    for ref in anchor['reference']:
                        number = ref['detection_index']
                        assert ref['xyxy'] == pure[number]['xyxy']
                        refs.append({**ref, 'track_id': output[number]['track_id'],
                                     'tracking_status': output[number]['tracking_status']})
                    observed[name].append({'frame_index': frame_index, 'split': anchor['split'],
                                           'reference': refs})
        cap.release()
        for name in configs:
            s = summary[name]
            s['allocated_id_count'] = len(s.pop('allocated_ids'))
            seq_out['methods'][name] = {'all': metrics(observed[name]),
                'development': metrics([f for f in observed[name] if f['split'] == 'development']),
                'verification': metrics([f for f in observed[name] if f['split'] == 'verification']),
                'all_frame_summary': s, 'anchors': observed[name]}
        (OUT / (sequence['scene'] + '-assignments-v1.json')).write_text(
            json.dumps(assignments, indent=2) + '\n', encoding='utf-8')
        results['sequences'].append(seq_out)
    results['overall'] = {name: {key: sum(s['methods'][name]['all'][key] for s in results['sequences'])
                                for key in ('reference_observations', 'reference_identities',
                                            'id_changes_along_reference', 'observed_cross_reference_id_transfers')}
                           for name in configs}
    (OUT / 'results-v1.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results['overall'], indent=2))
    for seq in results['sequences']:
        print(seq['scene'], {name: {k: data['all'][k] for k in ('id_changes_along_reference', 'observed_cross_reference_id_transfers')}
                             for name, data in seq['methods'].items()})


if __name__ == '__main__':
    main()
