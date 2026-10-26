"""Diagnóstico observado, incluindo operação real em CUDA sem carregar modelos."""

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .data import utc_now, write_json


def executable_info(name: str, arguments: list[str]) -> dict:
    executable = shutil.which(name)
    if not executable:
        return {"available": False, "reason": "Não encontrado no PATH"}
    try:
        result = subprocess.run([executable, *arguments], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=20, check=False)
        return {"available": result.returncode == 0, "path": executable,
                "returncode": result.returncode,
                "output": (result.stdout or result.stderr).strip().splitlines()[:6]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "path": executable, "reason": str(error)}


def diagnose(output: Path) -> dict:
    report = {"created_at_utc": utc_now(), "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "in_venv": sys.prefix != sys.base_prefix, "versions": {}, "import_errors": {},
              "ffmpeg": executable_info("ffmpeg", ["-version"]),
              "ffprobe": executable_info("ffprobe", ["-version"]),
              "nvidia_smi": executable_info("nvidia-smi", ["--query-gpu=name,memory.total,driver_version", "--format=csv"])}
    modules = {}
    for name in ["torch", "torchvision", "cv2", "numpy", "pandas", "sklearn"]:
        try:
            modules[name] = importlib.import_module(name)
            report["versions"][name] = modules[name].__version__
        except Exception as error:
            report["import_errors"][name] = str(error)
    if "cv2" in modules:
        report["opencv_ffmpeg"] = [line.strip() for line in modules["cv2"].getBuildInformation().splitlines() if "FFMPEG:" in line]
    torch = modules.get("torch")
    report["cuda"] = {"available": False, "operation": "não executada"}
    if torch is not None:
        report["cuda"]["runtime"] = torch.version.cuda
        try:
            report["cpu_tensor_operation"] = float((torch.arange(4, dtype=torch.float32) ** 2).sum()) == 14.0
            available = torch.cuda.is_available()
            report["cuda"]["available"] = available
            if available:
                prop = torch.cuda.get_device_properties(0)
                report["cuda"].update({"device": prop.name, "memory_bytes": prop.total_memory,
                                         "compute_capability": f"{prop.major}.{prop.minor}"})
                value = (torch.arange(4, dtype=torch.float32, device="cuda") ** 2).sum().item()
                torch.cuda.synchronize()
                if value != 14.0:
                    raise RuntimeError(f"Resultado inesperado na GPU: {value}")
                report["cuda"]["operation"] = "ok"
                if "torchvision" in modules:
                    boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
                    keep = modules["torchvision"].ops.nms(boxes, torch.tensor([1.], device="cuda"), 0.5)
                    report["cuda"]["torchvision_nms"] = "ok" if keep.cpu().tolist() == [0] else "falhou"
            else:
                report["cuda"]["reason"] = "torch.cuda.is_available() retornou False; veja runtime, imports e nvidia_smi"
        except Exception as error:
            report["cuda"]["error"] = str(error)
    write_json(output, report)
    return report
