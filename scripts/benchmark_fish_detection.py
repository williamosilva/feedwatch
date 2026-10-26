"""Compare registered detector configurations on an explicit annotated frame manifest."""

import argparse
import json
from pathlib import Path

import cv2

from feedwatch.data import file_hash
from feedwatch.detection_evaluation import detection_metrics
from feedwatch.fish_detector import FishDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'holdout'], required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    configurations = json.loads(args.candidates.read_text(encoding='utf-8'))
    selected = [frame for frame in annotations['frames'] if frame['split'] == args.split]
    if not selected:
        raise ValueError('Nenhum quadro na divisão selecionada')
    result = {'schema': 1, 'annotations_sha256': file_hash(args.annotations),
              'candidates_sha256': file_hash(args.candidates), 'split': args.split,
              'annotation_status': annotations['annotation_status'], 'iou_threshold': .5,
              'results': {}}
    for name, config in configurations.items():
        detector = FishDetector(config, args.device)
        frames = []
        for frame in selected:
            image = cv2.imread(str(args.annotations.parent / frame['image']))
            if image is None:
                raise ValueError('Imagem não encontrada: ' + frame['image'])
            image_path = args.annotations.parent / frame['image']
            frames.append({**frame, 'image_sha256': file_hash(image_path),
                           'image_dimensions': {'width': image.shape[1], 'height': image.shape[0]},
                           'predicted': detector.detect(image)})
        result['results'][name] = {'detector': detector.identification, 'frames': frames,
                                  'overall': detection_metrics(frames),
                                  'by_scene': {scene: detection_metrics([frame for frame in frames
                                             if frame['frame'].startswith(scene + '_')])
                                               for scene in sorted({f['frame'].rsplit('_', 1)[0] for f in frames})}}
        print(name, {key: value for key, value in result['results'][name]['overall'].items()
                     if key != 'by_frame'}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
