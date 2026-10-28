"""Replay tracker identities over a verified source with cached detections and feeding results."""

import argparse
import json
from pathlib import Path

from feedwatch.tracking_replay import replay_tracking_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='Original recording, matching the saved input hash')
    parser.add_argument('--analysis-dir', type=Path, required=True, help='Saved simple run containing run.json and frames.jsonl')
    parser.add_argument('--config', type=Path, default=Path('config/simple_video.json'))
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    args = parser.parse_args()
    result = replay_tracking_video(args.video, args.analysis_dir, args.output_dir, args.config)
    print(json.dumps({'video': result['output_video'], 'sha256': result['output_sha256'],
                      'frames': result['rendered_frames'], 'feeding_inference_run': False,
                      'detector_inference_run': False}))


if __name__ == '__main__':
    main()
