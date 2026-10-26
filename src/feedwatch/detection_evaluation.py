"""One-to-one box and identity evaluation for explicitly annotated video frames."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dynamic_tracking import iou


def match_boxes(reference, predicted, threshold=.5):
    """Maximize valid IoU matches; a detection can never match two fish."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU de avaliação inválido")
    overlaps = np.array([[iou(a, b) for b in predicted] for a in reference], dtype=float)
    if not reference or not predicted:
        return []
    # A valid match always outweighs any improvement in total IoU.
    cost = np.where(overlaps >= threshold, -1 - overlaps / (max(overlaps.shape) + 1), 0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(a), int(b), float(overlaps[a, b])) for a, b in zip(rows, cols)
            if overlaps[a, b] >= threshold]


def detection_metrics(frames, threshold=.5):
    """Evaluate all visible reference boxes, including frames with zero predictions."""
    tp = fp = fn = hand_errors = 0
    frame_results = []
    for frame in frames:
        truth = [item['xyxy'] for item in frame['reference']]
        predictions = [item['xyxy'] for item in frame['predicted']]
        matched = match_boxes(truth, predictions, threshold)
        matched_predictions = {b for _, b, _ in matched}
        errors = sum(any(iou(box, hand) >= .3 for hand in frame.get('hands', []))
                     for n, box in enumerate(predictions) if n not in matched_predictions)
        values = {'frame': frame['frame'], 'tp': len(matched),
                  'fp': len(predictions) - len(matched), 'fn': len(truth) - len(matched),
                  'unmatched_hand_boxes': errors, 'matches': matched}
        tp += values['tp']; fp += values['fp']; fn += values['fn']; hand_errors += errors
        frame_results.append(values)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'frames': len(frames), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'unmatched_hand_boxes': hand_errors, 'by_frame': frame_results}


def identity_metrics(frames, threshold=.5):
    """Count identity switches only on matched, provisionally identified fish."""
    previous = {}
    switches = matched_total = misses = 0
    for frame in frames:
        matches = match_boxes([x['xyxy'] for x in frame['reference']],
                              [x['xyxy'] for x in frame['predicted']], threshold)
        misses += len(frame['reference']) - len(matches)
        for reference_index, predicted_index, _ in matches:
            truth_id = frame['reference'][reference_index]['identity']
            track_id = frame['predicted'][predicted_index].get('track_id')
            if track_id is None:
                misses += 1
                continue
            matched_total += 1
            if truth_id in previous and previous[truth_id] != track_id:
                switches += 1
            previous[truth_id] = track_id
    return {'matched_observations': matched_total, 'missed_observations': misses,
            'identity_switches': switches,
            'scope': 'Annotated fish only; misses and switches are separate, not MOTA/HOTA/IDF1.'}
