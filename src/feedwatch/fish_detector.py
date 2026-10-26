"""Fishial external detector. No feeding model dependency and no implicit downloads."""

import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import cv2

from .data import file_hash
from .final_evaluation import inference_setup
from .dynamic_tracking import iou


def letterbox_tensor(image, size, device):
    """Fixed square RGB input; record actual rounded resize and integer padding."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    scaled_width, scaled_height = round(width * ratio), round(height * ratio)
    left, top = (size - scaled_width) // 2, (size - scaled_height) // 2
    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top:top + scaled_height, left:left + scaled_width] = resized
    tensor = torch.from_numpy(padded[:, :, ::-1].transpose(2, 0, 1).copy()).to(device).float()[None] / 255.
    return tensor, (ratio, left, top)


def decode_torchscript_boxes(prediction, original_shape, geometry, config):
    """Decode the verified one-class YOLOv12 export, then restore source pixels once."""
    values = prediction.detach().cpu().numpy()
    if values.shape != (1, 5, 8400) or not np.isfinite(values).all():
        raise ValueError("Saída TorchScript diferente do contrato verificado [1,5,8400]")
    raw = values[0].T
    raw = raw[raw[:, 4] >= config['conf']]
    if not len(raw):
        return np.empty((0, 6), dtype=float)
    xyxy = np.column_stack([raw[:, 0] - raw[:, 2] / 2, raw[:, 1] - raw[:, 3] / 2,
                            raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2])
    selected = []
    for index in np.argsort(-raw[:, 4], kind='stable'):
        if all(iou(xyxy[index], xyxy[kept]) <= config['iou'] for kept in selected):
            selected.append(int(index))
            if len(selected) >= config['max_det']:
                break
    boxes = xyxy[selected]
    ratio, left, top = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    height, width = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    values = np.column_stack([boxes, raw[selected, 4], np.zeros(len(selected))])
    return values[(values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])]


def postprocess_boxes(values, width, height, config):
    """SDK boxes are ALREADY in source coordinates; optional class-specific duplicate removal."""
    detections = []
    for box in values:
        if len(box) != 6 or not np.isfinite(box).all():
            raise ValueError("Saída inválida do detector")
        x1,y1,x2,y2,score,class_id = map(float,box)
        if class_id != 0 or not 0 <= score <= 1:
            raise ValueError("Classe/score diferente do contrato Fish verificado")
        x1,x2 = np.clip([x1,x2],0,width).tolist()
        y1,y2 = np.clip([y1,y2],0,height).tolist()
        if x2 <= x1 or y2 <= y1 or score < config["conf"]:
            raise ValueError("Caixa inválida; não desenhar detecção inventada")
        detections.append({"xyxy":[x1,y1,x2,y2],"detector_class_id":0,"detector_class":"Fish","score":score})
    threshold = config.get("post_nms_iou")
    if threshold is None:
        return detections,[]
    if not 0 < threshold < 1:
        raise ValueError("IoU de remoção de duplicatas inválido")
    retained,removed = [],[]
    for index in sorted(range(len(detections)),key=lambda i:(-detections[i]["score"],i)):
        candidate = detections[index]
        if any(iou(candidate["xyxy"],detections[kept]["xyxy"]) > threshold for kept in retained):
            removed.append(index)
        else:
            retained.append(index)
    return [detections[i] for i in retained],removed


def prepare_detector(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))["detector"]
    return prepare_detector_weights(config)


def prepare_detector_weights(config):
    """Explicitly download every registered ensemble member and verify each digest."""
    extra = [prepare_detector_weights(item) for item in config.get('additional_detectors', [])]
    weights = Path(config["weights"])
    if weights.exists():
        if file_hash(weights) != config["weights_sha256"]:
            raise ValueError("Pesos existentes diferem do hash registrado; não sobrescrever")
        return {"weights": str(weights), "sha256": file_hash(weights), "downloaded": False,
                'additional_detectors': extra}
    weights.parent.mkdir(parents=True, exist_ok=True)
    archive = weights.parent / config.get('archive_filename', 'detector_v26_n3.zip')
    if archive.parent != weights.parent or Path(config.get('archive_member', 'model.pt')).name != config.get('archive_member', 'model.pt'):
        raise ValueError('Nomes de arquivo do detector inválidos')
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urlopen(config["download_url"], timeout=60) as response, temporary.open("wb") as handle:
            while block := response.read(1024*1024):
                handle.write(block)
        if file_hash(temporary) != config["archive_sha256"]:
            raise ValueError("Hash do pacote baixado diverge")
        temporary.replace(archive)
    if file_hash(archive) != config["archive_sha256"]:
        raise ValueError("Hash do pacote do detector diverge")
    # Extract only the known basename; no arbitrary paths or vendor Python execution.
    with zipfile.ZipFile(archive) as package:
        members = [name for name in package.namelist() if Path(name).name == config.get('archive_member', 'model.pt') and "__MACOSX" not in name]
        if len(members) != 1:
            raise ValueError("Pacote sem um único model.pt")
        payload = package.read(members[0])
    import hashlib
    if hashlib.sha256(payload).hexdigest() != config["weights_sha256"]:
        raise ValueError("Hash dos pesos extraídos diverge")
    pending_weights = weights.with_name(weights.name + '.pending')
    pending_weights.write_bytes(payload)
    if file_hash(pending_weights) != config['weights_sha256']:
        raise ValueError('Hash dos pesos gravados diverge')
    pending_weights.replace(weights)
    return {"weights": str(weights), "sha256": file_hash(weights), "downloaded": True,
            'additional_detectors': extra}


def validate_fish_classes(names: dict, expected: dict):
    normalized = {str(key): value for key, value in names.items()}
    if normalized != expected or normalized != {"0": "Fish"}:
        raise ValueError(f"Detector não corresponde à classe Fish verificada: {normalized}")


class FishDetector:
    def __init__(self, config: dict, device: str):
        from ultralytics import YOLO, __version__
        path = Path(config["weights"])
        if not path.is_file() or file_hash(path) != config["weights_sha256"]:
            raise ValueError("Detector ausente/incompatível; execute prepare-fish-detector ou use --motion-only")
        self.backend = config.get('backend', 'ultralytics')
        self.export_metadata = None
        if self.backend == 'fishial-torchscript':
            metadata = {'config.txt': ''}
            self.model = torch.jit.load(str(path), map_location=device, _extra_files=metadata).eval()
            self.export_metadata = json.loads(metadata['config.txt'])
            validate_fish_classes(self.export_metadata['names'], config['class_mapping'])
            if (self.export_metadata.get('task') != 'detect' or self.export_metadata.get('imgsz') != [640, 640]
                    or self.export_metadata.get('batch') != 1 or config['imgsz'] != 640
                    or self.export_metadata.get('args', {}).get('nms') is not False):
                raise ValueError('Metadados TorchScript incompatíveis com o adaptador verificado')
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        elif self.backend == 'ultralytics':
            self.model = YOLO(str(path), task="detect")
            validate_fish_classes(self.model.names, config["class_mapping"])
            self.model.model.requires_grad_(False).eval()
        else:
            raise ValueError('Backend de detecção desconhecido')
        self.config, self.device = config, device
        self.identification = {**config, "weights": str(path.resolve()), "runtime_ultralytics": __version__,
            "checkpoint_ultralytics": self.export_metadata['version'] if self.export_metadata else self.model.ckpt.get("version"),
            "classes_verified": self.export_metadata['names'] if self.export_metadata else dict(self.model.names),
            "trained_in_feedwatch": False, "scores": "detector outputs; not calibrated confidence",
            "scope": "full frame; no crop from the GRU is used",
            "format": "TorchScript one-class YOLOv12 export with embedded metadata" if self.export_metadata else "Ultralytics DetectionModel checkpoint (not TorchScript)",
            "runtime_device":device,"sdk_end2end":False if self.export_metadata else bool(getattr(self.model.model,"end2end",False)),
            "box_coordinates":("fixed square RGB/114 letterbox; xywh decoded; NMS; source coordinates restored once"
                               if self.export_metadata else "SDK construct_result scales letterboxed boxes to original pixels; no second scaling"),
            "explicit_duplicate_nms":config.get("post_nms_iou")}
        self.additional = [FishDetector(item, device) for item in config.get('additional_detectors', [])]
        if self.additional:
            self.identification['additional_detectors'] = [item.identification for item in self.additional]
            self.identification['combination'] = 'Fixed detectors on every full frame; union with class-specific duplicate NMS. No scene or position rules.'
        # Ultralytics initialization must not change the feeding inference contract.
        inference_setup(device)

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> list[dict]:
        cfg = self.config
        height, width = image.shape[:2]
        if self.backend == 'fishial-torchscript':
            tensor, geometry = letterbox_tensor(image, cfg['imgsz'], self.device)
            values = decode_torchscript_boxes(self.model(tensor), (height, width), geometry, cfg)
        else:
            result = self.model.predict(image, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                max_det=cfg["max_det"], half=False, augment=False, classes=[0], device=self.device,
                verbose=False, save=False, stream=False)[0]
            if tuple(result.orig_shape) != (height,width) or tuple(result.boxes.orig_shape) != (height,width):
                raise ValueError("Coordenadas do SDK não correspondem ao frame original")
            values = result.boxes.data.detach().cpu().numpy()
        inference_setup(self.device)
        additional = [box for detector in self.additional for box in detector.detect(image)]
        if additional:
            # Each detector already applies its registered score threshold.
            values = np.concatenate([values, np.array([[*box['xyxy'], box['score'], 0] for box in additional])])
        postprocess_config = {**cfg, 'conf': min([cfg['conf']] + [d.config['conf'] for d in self.additional])}
        detections,removed = postprocess_boxes(values,width,height,postprocess_config)
        self.last_audit = {"sdk_boxes":len(values),"retained_boxes":len(detections),"duplicate_indices_removed":removed}
        return detections
