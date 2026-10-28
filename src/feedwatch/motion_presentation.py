"""Synchronized optical-flow display, separate from feeding predictions and detections.

The input must be the original recording. Color represents apparent image motion,
including water, reflections and camera movement; it is neither temperature nor a
feeding-model attention map. All frames share one fixed source-pixel/s scale.
"""

import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .data import file_hash, write_json
from .image_motion import ImageMotion
from .video_export import H264Writer, verify_export
from .video_inference import inspect_video, open_capture


NEUTRAL_BGR = (232, 236, 240)


def motion_heatmap(flow, scale_max_source_px_per_second, size):
    """Render BGR TURBO colors on a fixed scale; gray denotes no measurement.

    ``size`` is (width, height). Blue means less apparent motion, red means more;
    magnitudes at or above the configured maximum use the same saturated color.
    """
    maximum = float(scale_max_source_px_per_second)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Motion display maximum must be finite and positive")
    width, height = size
    if min(width, height) < 2:
        raise ValueError("Motion display dimensions must be at least two pixels")
    if flow is None:
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = NEUTRAL_BGR
        return image
    if flow.ndim != 3 or flow.shape[2] != 2 or not np.isfinite(flow).all():
        raise ValueError("Invalid optical-flow field")
    magnitude = np.linalg.norm(flow, axis=2)
    values = np.uint8(np.clip(magnitude / maximum, 0., 1.) * 255)
    # Interpolate color after applying a common scale, never normalize a frame.
    colors = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    return cv2.resize(colors, (width, height), interpolation=cv2.INTER_LINEAR)


def _output_size(width, height, long_edge):
    if not isinstance(long_edge, int) or long_edge < 16:
        raise ValueError("Motion analysis long edge must be an integer >= 16")
    ratio = min(1., long_edge / max(width, height))
    # H.264 uses even dimensions. Rounding is at most one pixel per dimension.
    return tuple(max(2, 2 * round(value * ratio / 2)) for value in (width, height))


def export_motion_presentation(source: Path, output: Path, motion_config: dict, *, progress=None):
    """Export a silent, frame-aligned heatmap and small scalar JSON records.

    No detector or feeding inference is run. ``motion_config`` is the existing
    config's ``motion`` section. The original recording is decoded once, and the
    exported video is fully decoded to verify frame count, FPS and duration.
    Existing motion exports are never overwritten. Other files may coexist in
    ``output`` so this can be called from a completed local analysis directory.
    """
    from .dynamic_video import visual_timestamp

    source, output = Path(source).resolve(), Path(output).resolve()
    maximum = float(motion_config["heatmap_max_source_px_per_second"])
    threshold = float(motion_config["active_threshold_source_px_per_second"])
    if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Motion scale and threshold must be finite and positive")
    names = ("motion.mp4", "motion.pending.mp4", "motion.json", "motion-audit.json", "motion-encoder.log")
    if any((output / name).exists() for name in names):
        raise ValueError("Motion output already exists; choose a fresh output directory")
    if source in (output / name for name in names):
        raise ValueError("Motion output must not replace the source recording")
    before = (source.stat().st_size, source.stat().st_mtime_ns, file_hash(source))
    started = time.perf_counter()
    cap = open_capture(source)
    writer = None
    try:
        metadata = inspect_video(cap)
        width, height = _output_size(metadata["width"], metadata["height"], motion_config["analysis_long_edge"])
        export_metadata = {**metadata, "width": width, "height": height}
        output.mkdir(parents=True, exist_ok=True)
        pending = output / "motion.pending.mp4"
        writer = H264Writer(pending, source, export_metadata, False, output / "motion-encoder.log")
        movement = ImageMotion(dict(motion_config))
        previous_time = None
        frames, errors, clocks = [], [], Counter()
        if progress:
            progress("motion", 0, metadata["frame_count"])
        for index in range(metadata["frame_count"]):
            ok, image = cap.read()
            if not ok or image is None or image.shape[:2] != (metadata["height"], metadata["width"]):
                raise ValueError(f"Original recording cannot be decoded at frame {index}")
            decoder_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp, compatible, clock = visual_timestamp(decoder_time, index, metadata["fps"], previous_time)
            previous_time = timestamp if compatible else previous_time
            clocks[clock] += 1
            try:
                if not compatible:
                    raise ValueError("Decoder clock is incompatible with the source frame")
                metrics, flow = movement.update(image, timestamp)
            except (ValueError, RuntimeError, cv2.error) as error:
                metrics, flow = {"status": "unavailable"}, None
                errors.append({"frame_index": index, "error": str(error)})
                movement = ImageMotion(dict(motion_config))
            row = {"frame_index": index, "timestamp_seconds": index / metadata["fps"],
                   "status": metrics["status"],
                   "mean_source_px_per_second": metrics.get("mean_source_px_per_second"),
                   "p90_source_px_per_second": metrics.get("p90_source_px_per_second"),
                   "moving_fraction": metrics.get("moving_fraction")}
            frames.append(row)
            writer.write(motion_heatmap(flow, maximum, (width, height)))
            if index in (0, metadata["frame_count"] // 2, metadata["frame_count"] - 1):
                cv2.imwrite(str(output / f"motion-frame-{index:05d}.jpg"),
                            motion_heatmap(flow, maximum, (width, height)))
            if progress:
                progress("motion", index + 1, metadata["frame_count"])
        if cap.read()[0]:
            raise ValueError("Original recording contains more frames than its metadata")
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    verification = verify_export(pending, export_metadata)
    if (source.stat().st_size, source.stat().st_mtime_ns, file_hash(source)) != before:
        raise ValueError("Source recording changed during motion export")
    final = output / "motion.mp4"
    pending.replace(final)
    result = {"schema": 1, "status": "ok" if not errors else "completed_with_unavailable_frames",
              "video": str(final), "frames": frames, "fps": metadata["fps"],
              "frame_count": metadata["frame_count"], "duration_seconds": metadata["duration_seconds"],
              "width": width, "height": height, "colormap": "TURBO",
              "scale_max_source_px_per_second": maximum,
              "active_threshold_source_px_per_second": threshold,
              "measurement": "apparent image motion from consecutive original frames",
              "scope": "Includes water, reflections and camera movement; not temperature, fish physical speed or feeding-model attention.",
              "missing_display": "neutral gray; status and scalar values indicate unavailable measurements",
              "video_sha256": file_hash(final)}
    write_json(output / "motion.json", result)
    write_json(output / "motion-audit.json", {
        "source": str(source), "source_sha256": before[2], "source_metadata": metadata,
        "motion_config": motion_config, "export_verification": verification,
        "timestamp_sources": dict(clocks), "errors": errors,
        "source_image_only": True, "feeding_or_detector_inference_run": False,
        "source_frame_count_preserved": True, "processing_seconds": time.perf_counter() - started,
    })
    return result
