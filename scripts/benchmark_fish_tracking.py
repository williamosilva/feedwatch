"""Compare association methods on cached identical detections and identity references."""

import argparse
import json
from pathlib import Path

from feedwatch.association_tracking import create_tracker
from feedwatch.data import file_hash
from feedwatch.detection_evaluation import identity_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/simple_video.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.sequences.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))['tracker']
    result = {'schema': 1, 'sequences_sha256': file_hash(args.sequences),
              'annotation_status': manifest['annotation_status'],
              'comparison': 'Identical detector outputs, source timestamps, no predicted boxes. BoT-SORT without ReID/GMC.',
              'results': {}}
    for kind in ['conservative-v2', 'bytetrack', 'botsort']:
        association_config = {**config, 'kind': kind}
        if kind != 'conservative-v2':
            # The fixed ensemble accepts scores from .15; let native trackers start
            # those observations rather than creating a fresh untracked ID each frame.
            association_config.update(track_high_thresh=.15, track_low_thresh=.10,
                                      new_track_thresh=.15, fuse_score=False)
        summaries = []
        for sequence in manifest['sequences']:
            tracker = create_tracker(association_config, sequence['fps'])
            frames = []
            for frame in sequence['frames']:
                observations = tracker.update(frame['detections'], frame['timestamp'], frame['frame_index'])
                frames.append({**frame, 'predicted': observations})
            summaries.append({'scene': sequence['scene'], 'metrics': identity_metrics(frames), 'frames': frames})
        total = {key: sum(sequence['metrics'][key] for sequence in summaries)
                 for key in ['matched_observations', 'missed_observations', 'identity_switches']}
        result['results'][kind] = {'config': association_config, 'total': total, 'sequences': summaries}
        print(kind, total, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
