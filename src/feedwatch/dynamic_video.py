"""Independent visual analysis alongside the unchanged operational feeding pipeline."""

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import file_hash, utc_now, write_json
from .dynamic_overlay import DynamicOverlay
from .dynamic_tracking import ObservationTracker
from .final_evaluation import inference_setup
from .fish_detector import FishDetector
from .image_motion import ImageMotion
from .video_export import H264Writer, media_info, prepare_output, verify_export
from .video_inference import analyze_video, display_state, open_capture
from .video_overlay import unavailable_image


def visual_timestamp(decoded, frame_index, fps, previous=None):
    """Accept decoder clocks; repair a near-CFR duplicate without losing a frame."""
    expected = frame_index / fps
    compatible = math.isfinite(decoded) and abs(decoded - expected) <= max(1.5 / fps, .02)
    if compatible and frame_index == 0 and decoded != 0:
        return 0., True, 'frame_index_fallback_initial_decoder_offset'
    if compatible and (previous is None or decoded > previous):
        return decoded, True, 'decoder'
    if compatible and (previous is None or expected > previous):
        return expected, True, 'frame_index_fallback_nonmonotonic_decoder'
    return expected, False, 'unavailable_decoder_clock'


def hardware():
    return {"platform":platform.platform(),"cpu":platform.processor(),"python":platform.python_version(),
            "torch":torch.__version__,"opencv":cv2.__version__,
            "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_memory_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}


def load_dynamic(config_path, predictor, motion_only=False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("Configuração dinâmica incompatível")
    detector = None if motion_only else FishDetector(config["detector"], str(predictor.device))
    inference_setup(str(predictor.device))
    return config, detector


def export_dynamic(source: Path, output: Path, predictor, config: dict, detector,
                   mode="sequential", clip_context="", *, simple=False, diagnostic=False, source_note="", progress=None) -> dict:
    prepare_output(output,predictor)
    started = time.perf_counter()
    source = source.resolve()
    report = {"created_at_utc":utc_now(),"input_video":str(source),"mode":mode,"food_model":predictor.identification,
        "dynamic_config":config,"detector":detector.identification if detector else None,"hardware":hardware(),
        "status":"running","training":False,"realtime_claim":False,"smoothing":False,
        "processing":"offline two-pass; food display gated by original display_state; visual analysis uses current/past frames only",
        "incomplete_policy":"original plan_windows: tail <2s never padded/classified; last completed food result retains source interval",
        "tracks":"temporary clip-local observation associations, not permanent identities; missing boxes never drawn",
        "speed":"consecutive detected box center displacement / actual timestamp delta, source px/s, not fish physical speed",
        "image_motion":"full-frame image motion includes water/reflections; not detected fish or feeding GRU output",
        "visual_errors":[],"display_events":[]}
    frame_count = 0
    counters = {"frames_detector_ok":0,"frames_with_detections":0,"detection_observations":0,
        "observations_with_speed":0,"frames_motion_ok":0,"max_detections_in_frame":0}
    timings = {"detector_seconds":0.,"tracking_seconds":0.,"motion_seconds":0.,"render_encode_seconds":0.}
    try:
        before = (source.stat().st_size,source.stat().st_mtime_ns,file_hash(source))
        report["input_signature"] = {"size_bytes":before[0],"mtime_ns":before[1],"sha256":before[2]}
        # Exactly the original decoder, temporal sampling, transform, zero GRU state and checkpoint.
        inference_setup(str(predictor.device))
        report.update(analyze_video(source,predictor,mode,progress=progress))
        if report["fatal_error"]:
            raise ValueError(report["fatal_error"])
        metadata, windows = report["metadata"], report["windows"]
        if simple:
            from .simple_overlay import SimpleOverlay
            renderer=SimpleOverlay(metadata['width'],metadata['height'],predictor,config,diagnostic,source_note)
        else:
            renderer = DynamicOverlay(metadata["width"],metadata["height"],predictor,config)
        export_metadata=renderer.output_metadata(metadata) if simple else metadata
        report['render_geometry']={'source_width':metadata['width'],'source_height':metadata['height'],
            'output_width':export_metadata['width'],'output_height':export_metadata['height'],
            'source_image_scaled':False,'footer_added':simple}
        report["food_roi_xyxy"],report["food_effective_crop_xyxy"] = renderer.roi,renderer.crop
        media = media_info(source)
        report["source_media"] = media
        if simple:
            from .association_tracking import create_tracker
            tracker=create_tracker(config['tracker'], metadata['fps'])
        else:
            tracker=ObservationTracker(config['tracker'])
        motion=ImageMotion(config['motion'])
        report.update(presentation_profile='simple' if simple else 'legacy',diagnostic=diagnostic if simple else True,
                      source_note=source_note)
        if simple:
            report['speed']='net observed center displacement over >= configured 0.2 s consecutive valid history, source px/s'
            report['tracks']=config['tracker'].get('kind', 'conservative-v2') + '; confirmed observed trails only; no predicted boxes or bridged trails'
        cap,writer = open_capture(source),None
        previous_event = None
        previous_visual_timestamp = None
        pending = output/"annotated.pending.mp4"
        csv_fields = ["frame_index","timestamp_seconds","food_status","food_window_id","food_class",
            "detector_status","detected_count","motion_status","motion_mean_source_px_per_second",
            "motion_p90_source_px_per_second","moving_fraction","error"]
        try:
            writer = H264Writer(pending,source,export_metadata,media["has_audio"],output/"encoder.log")
            if progress: progress('render', 0, metadata['frame_count'])
            with (output/"frames.jsonl").open("w",encoding="utf-8") as details, (output/"frames.csv").open("w",encoding="utf-8",newline="") as summary:
                import csv
                csv_writer = csv.DictWriter(summary,fieldnames=csv_fields)
                csv_writer.writeheader()
                for index in range(metadata["frame_count"]):
                    ok, image = cap.read()
                    if not ok or image is None or image.shape[:2] != (metadata["height"],metadata["width"]):
                        raise ValueError(f"Falha de leitura/resolução no frame {index}; não publicar vídeo truncado")
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    errors = []
                    visual_time, compatible, timestamp_source = visual_timestamp(timestamp, index, metadata['fps'], previous_visual_timestamp)
                    previous_visual_timestamp = visual_time if compatible else previous_visual_timestamp
                    food = display_state(windows,mode,index/metadata["fps"])
                    item = food.get("window")
                    event = (food["status"],item["window_id"] if item else None,food["tail"])
                    if event != previous_event:
                        report["display_events"].append({"first_frame":index,"at_seconds":index/metadata["fps"],
                            "status":event[0],"window_id":event[1],"tail":event[2]})
                        previous_event = event
                    observations,detector_status = [],"disabled" if detector is None else "ok"
                    if detector is not None:
                        mark = time.perf_counter()
                        try:
                            detections = detector.detect(image)
                            timings["detector_seconds"] += time.perf_counter()-mark
                            mark = time.perf_counter()
                            if not compatible:
                                raise ValueError("Timestamp incompatível; rastreamento indisponível neste frame")
                            observations = (tracker.update(detections,visual_time,index,image=image) if simple
                                            else tracker.update(detections,visual_time,index))
                            timings["tracking_seconds"] += time.perf_counter()-mark
                            counters["frames_detector_ok"] += 1
                            counters["detection_observations"] += len(observations)
                            counters["frames_with_detections"] += bool(observations)
                            counters["observations_with_speed"] += sum(obs["speed_source_px_per_second"] is not None for obs in observations)
                            counters["max_detections_in_frame"] = max(counters["max_detections_in_frame"],len(observations))
                        except (ValueError,RuntimeError,cv2.error) as error:
                            detector_status = "unavailable"
                            # Keep ID counter monotonic after errors; never reuse an old identity within a clip.
                            if hasattr(tracker, 'clear_history'):
                                tracker.clear_history()
                            else:
                                tracker.tracks.clear()
                                tracker.last_timestamp = None
                            errors.append("detector/tracker: "+str(error))
                    mark = time.perf_counter()
                    try:
                        if not compatible:
                            raise ValueError("Timestamp incompatível; fluxo indisponível")
                        movement,flow = (motion.update(image,visual_time) if not simple or diagnostic
                                         else ({'status':'disabled'},None))
                        counters["frames_motion_ok"] += movement["status"] == "ok"
                    except (ValueError,RuntimeError,cv2.error) as error:
                        movement,flow = {"status":"unavailable","error":str(error)},None
                        motion = ImageMotion(config["motion"])
                        errors.append("motion: "+str(error))
                    timings["motion_seconds"] += time.perf_counter()-mark
                    mark = time.perf_counter()
                    annotated = renderer.render(image,food,mode,observations,detector_status,movement,flow,visual_time,clip_context)
                    writer.write(annotated)
                    if index in (0,metadata["frame_count"]//2,metadata["frame_count"]-1):
                        cv2.imwrite(str(output/("poster.jpg" if index == 0 else f"frame_{index:05d}.jpg")),annotated)
                    timings["render_encode_seconds"] += time.perf_counter()-mark
                    current_class = item["predicted_class"] if food["status"] == "prediction" else None
                    row = {"frame_index":index,"timestamp_seconds":visual_time if compatible else None,
                        "food_status":food["status"],"food_window_id":item["window_id"] if item else None,
                        "food_class":current_class,"detector_status":detector_status,
                        "detected_count":len(observations) if detector_status == "ok" else None,
                        "motion_status":movement["status"],"motion_mean_source_px_per_second":movement.get("mean_source_px_per_second"),
                        "motion_p90_source_px_per_second":movement.get("p90_source_px_per_second"),
                        "moving_fraction":movement.get("moving_fraction"),"error":"; ".join(errors)}
                    extra=({'confirmed_track_count':sum(o.get('track_confirmed',False) for o in observations),
                            'suspended_track_ids':getattr(tracker,'suspended_ids',[]),
                            'detector_postprocess':getattr(detector,'last_audit',None),
                            'layout':renderer.last_layout} if simple else {})
                    details.write(json.dumps({**row,"decoder_timestamp_seconds":timestamp if math.isfinite(timestamp) else None,
                        'timestamp_source':timestamp_source,"detections":observations,"image_motion":movement,**extra},ensure_ascii=False,allow_nan=False)+"\n")
                    csv_writer.writerow(row)
                    if errors:
                        report["visual_errors"].append({"frame_index":index,"errors":errors})
                    frame_count += 1
                    if progress: progress('render', frame_count, metadata['frame_count'])
            if cap.read()[0]:
                raise ValueError("Contagem de frames mudou entre leituras")
        finally:
            cap.release()
            if writer is not None:
                writer.close()
        report["encoding"] = {"ffmpeg_command":writer.command,"audio":"source first stream AAC" if media["has_audio"] else "no source audio"}
        if progress: progress('verify', 0, export_metadata['frame_count'])
        report["export_verification"] = verify_export(pending,export_metadata,progress=progress)
        if (source.stat().st_size,source.stat().st_mtime_ns,file_hash(source)) != before:
            raise ValueError("Original mudou durante execução")
        predictor.verify_frozen()
        final = output/"annotated.mp4"
        pending.replace(final)
        report.update(status="ok" if not report["visual_errors"] and all(w["status"]=="ok" for w in windows) else "completed_with_unavailable_intervals",
                      output_video=str(final.resolve()),output_sha256=file_hash(final))
    except (ValueError,OSError,RuntimeError,cv2.error,subprocess.SubprocessError) as error:
        report.update(status="unavailable",error=str(error),output_video=None)
        unavailable_image(output/"analysis_unavailable.png",str(error))
    report.update(counters=counters,timings=timings,rendered_frames=frame_count,
                  processing_seconds=time.perf_counter()-started)
    report["effective_processing_fps"] = frame_count/report["processing_seconds"]
    if torch.cuda.is_available() and predictor.device.type == "cuda":
        report["gpu_peak_allocated_bytes_since_process_start"] = torch.cuda.max_memory_allocated()
    pd.DataFrame([{**window,"scores":json.dumps(window["scores"]),
        "sampled_indices":json.dumps(window["sampled_indices"]),
        "sampled_timestamps_seconds":json.dumps(window["sampled_timestamps_seconds"])}
        for window in report.get("windows",[])]).to_csv(output/"windows.csv",index=False)
    write_json(output/"analysis.json",report)
    return report
