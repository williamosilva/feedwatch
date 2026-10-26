"""Exportação local H.264, rastreabilidade e verificação integral da cópia."""

import json
import math
import os
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from .data import file_hash, utc_now, write_json
from .video_inference import OperationalPredictor, analyze_video, display_state, inspect_video, open_capture
from .video_overlay import OverlayRenderer, font_file, unavailable_image


def hidden_process():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def media_info(path: Path) -> dict:
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([executable, "-hide_banner", "-i", str(path)], capture_output=True, timeout=30, **hidden_process())
    text = result.stderr.decode("utf-8", errors="replace")
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    return {"ffmpeg": executable, "has_audio": "Audio:" in text,
            "container_duration_seconds": (int(duration[1])*3600+int(duration[2])*60+float(duration[3])) if duration else None}


class H264Writer:
    def __init__(self, path: Path, source: Path, metadata: dict, has_audio: bool, log: Path):
        self.log_handle = log.open("wb")
        fps = str(Fraction(metadata["fps"]).limit_denominator(100000))
        args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{metadata['width']}x{metadata['height']}",
                "-framerate", fps, "-i", "pipe:0"]
        if has_audio:
            args += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac"]
        else:
            args += ["-an"]
        self.pad = bool(metadata["width"] % 2 or metadata["height"] % 2)
        if self.pad:
            args += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2",
                 "-pix_fmt", "yuv420p", "-t", f"{metadata['duration_seconds']:.12f}",
                 "-movflags", "+faststart", str(path)]
        self.command = args
        try:
            self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log_handle, **hidden_process())
        except Exception:
            self.log_handle.close()
            raise

    def write(self, frame: np.ndarray):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        try:
            if not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            code = self.process.wait(timeout=120)
            if code:
                raise ValueError(f"FFmpeg terminou com código {code}; consulte encoder.log")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            raise ValueError("FFmpeg excedeu o tempo de finalização; exportação não concluída")
        finally:
            self.log_handle.close()


def verify_export(path: Path, expected: dict, progress=None) -> dict:
    cap = open_capture(path)
    try:
        actual = inspect_video(cap)
        frames = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image is None or image.shape[:2] != (actual["height"], actual["width"]):
                raise ValueError("Frame exportado inválido")
            frames += 1
            if progress: progress('verify', frames, expected['frame_count'])
        width, height = expected["width"]+expected["width"] % 2, expected["height"]+expected["height"] % 2
        if frames != expected["frame_count"] or actual["frame_count"] != frames:
            raise ValueError(f"Exportação perdeu frames: {frames} / {expected['frame_count']}")
        if (actual["width"], actual["height"]) != (width, height) or not math.isclose(actual["fps"], expected["fps"], rel_tol=1e-6):
            raise ValueError("Resolução/FPS da exportação divergem do esperado")
        if abs(actual["duration_seconds"]-expected["duration_seconds"]) > 1/expected["fps"]+1e-6:
            raise ValueError("Duração exportada diverge mais de um frame")
        return {"full_decode_verified": True, "decoded_frames": frames, **actual,
                "odd_dimension_padding_pixels": [width-expected["width"], height-expected["height"]]}
    finally:
        cap.release()


def prepare_output(output: Path, predictor: OperationalPredictor):
    target = output.resolve()
    roots = [Path("artifacts/experiments"), Path("artifacts/analyses"), Path("artifacts/pretrained"),
             Path("artifacts/visual-cache"), Path("artifacts/motion-cache"), Path("artifacts/final-test-cache")]
    indexed = Path("artifacts/index_report.json")
    if indexed.exists():
        roots.append(Path(json.loads(indexed.read_text(encoding="utf-8"))["data_dir"]))
    if any(target == root.resolve() or root.resolve() in target.parents for root in roots):
        raise ValueError("A saída deve ficar fora dos originais, experimentos e caches históricos")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diretório de saída não está vazio; use uma nova pasta para preservar as exportações")
    output.mkdir(parents=True, exist_ok=True)


def export_video(source: Path, output: Path, predictor: OperationalPredictor, mode="sequential") -> dict:
    prepare_output(output, predictor)
    started = time.perf_counter()
    source = source.resolve()
    report = {"created_at_utc": utc_now(), "input_video": str(source), "mode": mode,
              "model": predictor.identification, "status": "running", "training": False,
              "continuous_windows_evaluated_for_accuracy": False, "smoothing": False,
              "processing": "offline two-pass export; sequential display follows window availability, not measured live inference latency",
              "incomplete_policy": "sequential tail <2s never classified/padded; last valid prediction retains its source interval",
              "display_events": []}
    try:
        stat = source.stat()
        report["input_signature"] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_hash(source)}
        report.update(analyze_video(source, predictor, mode))
        if report["fatal_error"]:
            raise ValueError(report["fatal_error"]+"; nenhuma cópia truncada publicada como vídeo completo")
        metadata, windows = report["metadata"], report["windows"]
        renderer = OverlayRenderer(metadata["width"], metadata["height"], predictor.extraction, predictor.transform)
        report["source_roi_xyxy"] = renderer.roi
        report["source_effective_crop_xyxy"] = renderer.crop
        media = media_info(source)
        report["source_media"] = media
        cap, writer = open_capture(source), None
        render_started, frames, previous = time.perf_counter(), 0, None
        pending = output / "annotated.pending.mp4"
        try:
            writer = H264Writer(pending, source, metadata, media["has_audio"], output / "encoder.log")
            for index in range(metadata["frame_count"]):
                ok, image = cap.read()
                if not ok or image is None:
                    raise ValueError(f"Falha na segunda leitura no frame {index}; exportação não concluída")
                if image.shape[:2] != (metadata["height"], metadata["width"]):
                    raise ValueError("Resolução variável na segunda leitura")
                state = display_state(windows, mode, index/metadata["fps"])
                item = state.get("window")
                key = (state["status"], item["window_id"] if item else None, state["tail"])
                if key != previous:
                    report["display_events"].append({"first_frame": index, "at_seconds": index/metadata["fps"],
                        "status": state["status"], "window_id": item["window_id"] if item else None,
                        "window_start_seconds": item["start_seconds"] if item else None,
                        "window_end_seconds": item["end_seconds"] if item else None,
                        "tail_incomplete": state["tail"],
                        "last_valid_window_id": state["last_valid"]["window_id"] if state.get("last_valid") else None})
                    previous = key
                annotated = renderer.render(image, state, mode)
                writer.write(annotated)
                if frames == 0:
                    cv2.imwrite(str(output / "poster.jpg"), annotated)
                frames += 1
            if cap.read()[0]:
                raise ValueError("Número de frames mudou entre as duas leituras")
        finally:
            cap.release()
            if writer is not None:
                writer.close()
        report["render_encode_seconds"] = time.perf_counter()-render_started
        report["encoding"] = {"codec": "libx264", "pixel_format": "yuv420p", "preset": "veryfast", "crf": 20,
                              "audio": "AAC transcode of first source audio stream" if media["has_audio"] else "source has no audio",
                              "ffmpeg_command": writer.command}
        report["export_verification"] = verify_export(pending, metadata)
        if (source.stat().st_size, source.stat().st_mtime_ns, file_hash(source)) != (stat.st_size, stat.st_mtime_ns, report["input_signature"]["sha256"]):
            raise ValueError("O arquivo original mudou durante a exportação")
        predictor.verify_frozen()
        final = output / "annotated.mp4"
        pending.replace(final)
        report.update(output_video=str(final.resolve()), output_sha256=file_hash(final),
                      status="ok" if all(window["status"] == "ok" for window in windows) else "completed_with_unavailable_intervals")
    except (OSError, ValueError, RuntimeError, cv2.error, subprocess.SubprocessError) as error:
        report.update(status="unavailable", error=str(error), output_video=None)
        unavailable_image(output / "analysis_unavailable.png", str(error))
    report["processing_seconds"] = time.perf_counter()-started
    duration = report.get("metadata", {}).get("duration_seconds")
    report["processing_seconds_per_video_second"] = report["processing_seconds"]/duration if duration else None
    report["realtime_claim"] = False
    report["font_files"] = {"regular": font_file(), "bold": font_file(True)}
    windows = report.get("windows", [])
    pd.DataFrame([{**item, "sampled_indices": json.dumps(item["sampled_indices"]),
                   "sampled_timestamps_seconds": json.dumps(item["sampled_timestamps_seconds"]),
                   "scores": json.dumps(item["scores"])} for item in windows],
                 columns=list(windows[0]) if windows else ["window_id", "status", "error", "predicted_class", "scores"]).to_csv(output / "windows.csv", index=False)
    write_json(output / "analysis.json", report)
    return report
