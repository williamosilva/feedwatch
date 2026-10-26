"""Figuras estáticas dos 16 frames, após o pré-processamento real do extrator."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from .data import video_path, write_json
from .visual import decode_visual, encode_frames, load_extractor, state_hash


def effective_crop_box(width: int, height: int, config: dict, transform) -> tuple[tuple, tuple]:
    x1, y1, x2, y2 = config["crop_xyxy"]
    left, top, right, bottom = round(x1*width), round(y1*height), round(x2*width), round(y2*height)
    # Obtém a mesma geometria de resize do preset instalado, inclusive arredondamento.
    dummy = Image.new("RGB", (right-left, bottom-top))
    resized = F.resize(dummy, transform.resize_size, interpolation=transform.interpolation, antialias=transform.antialias)
    rw, rh = resized.size
    crop = transform.crop_size[0]
    cx, cy = int(round((rw-crop)/2)), int(round((rh-crop)/2))
    box = (left+cx*(right-left)/rw, top+cy*(bottom-top)/rh,
           left+(cx+crop)*(right-left)/rw, top+(cy+crop)*(bottom-top)/rh)
    return (left, top, right, bottom), box


def create_input_examples(root: Path, sequences: np.ndarray, rows, visual_config: dict,
                          device: torch.device, output: Path, seed: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rng = random.Random(seed)
    chosen = []
    for label in ("none", "weak", "medium", "strong"):
        candidates = rows[(rows.split == "val") & (rows.class_name == label)].index.tolist()
        chosen.append(rng.choice(candidates))
    medium_train = rows[(rows.split == "train") & (rows.class_name == "medium")].index.tolist()
    chosen += rng.sample(medium_train, min(2, len(medium_train)))
    ecfg = visual_config["extraction_identity"]["parameters"]
    weights = visual_config["extraction_identity"]["weights"]
    model, transform, _ = load_extractor(Path(weights["weights_file"]).parent, device)
    before = state_hash(model)
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for number, index in enumerate(chosen, start=1):
        row = rows.loc[index].to_dict()
        decoded = decode_visual(root, row["relative_path"], ecfg, transform)
        if not decoded["ok"]:
            raise ValueError(f"Falha ao mostrar entrada real: {row['relative_path']}: {decoded['error']}")
        encoded = encode_frames(model, decoded["frames"], device)
        np.testing.assert_allclose(encoded, sequences[index], rtol=1e-4, atol=2e-5)
        cap = cv2.VideoCapture(str(video_path(root, row["relative_path"])), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            ok, first = cap.read()
            if not ok:
                raise ValueError("Não foi possível ler o frame de contexto")
        finally:
            cap.release()
        h, w = first.shape[:2]
        roi, effective = effective_crop_box(w, h, ecfg, transform)
        mean = torch.tensor(transform.mean).view(1, 3, 1, 1)
        std = torch.tensor(transform.std).view(1, 3, 1, 1)
        images = (decoded["frames"]*std+mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        grid = fig.add_gridspec(5, 4, height_ratios=[1.25, 1, 1, 1, 1])
        context = fig.add_subplot(grid[0, :3])
        context.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
        for box, color, label in ((roi, "deepskyblue", "Recorte das bordas"), (effective, "red", "Área efetiva do center crop")):
            l, t, r, b = box
            context.add_patch(Rectangle((l, t), r-l, b-t, fill=False, edgecolor=color, linewidth=2, label=label))
        context.legend(fontsize=7, loc="lower left")
        context.set_title("Frame original e área efetivamente recebida pela CNN", fontsize=9)
        context.axis("off")
        first_input = fig.add_subplot(grid[0, 3])
        first_input.imshow(images[0])
        first_input.set_title("Entrada CNN 224×224\n(normalização desfeita só para exibir)", fontsize=8)
        first_input.axis("off")
        for frame in range(16):
            axis = fig.add_subplot(grid[1+frame//4, frame % 4])
            axis.imshow(images[frame])
            axis.set_title(f"{frame:02d}: frame {decoded['details']['sampled_indices'][frame]}, "
                           f"t={decoded['details']['sampled_times_seconds'][frame]:.2f}s", fontsize=8)
            axis.axis("off")
        fig.suptitle(f"Rótulo do dataset: {row['class_name']} (ID {row['class_id']}) | {row['split']}\n{row['relative_path']}", fontsize=10)
        filename = f"input_{number:02d}_{row['class_name']}.jpg"
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
        examples.append({**row, "figure": filename, "indices": decoded["details"]["sampled_indices"],
                         "times_seconds": decoded["details"]["sampled_times_seconds"], "source_roi_xyxy": roi,
                         "source_effective_center_crop_xyxy": effective,
                         "embedding_max_abs_difference": float(np.max(np.abs(encoded-sequences[index]))),
                         "clip_path": str(video_path(root, row["relative_path"]))})
    if state_hash(model) != before:
        raise ValueError("Extrator mudou durante a conferência visual")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {"seed": seed, "examples": examples, "extractor_state_unchanged": True,
              "caption": "Frames são entradas da CNN; as cabeças recebem embeddings. Rótulos do dataset preservados, não previsões."}
    write_json(output / "examples.json", report)
    (output / "README.md").write_text("# Frames efetivamente usados\n\n"
        "Cada figura mostra contexto, área do recorte central e os 16 frames pós-transformação. "
        "A normalização foi desfeita apenas na exibição RGB. Os embeddings foram recalculados nestes seis exemplos "
        "e conferidos contra o cache; os demais vídeos não foram decodificados novamente. "
        "As cabeças MLP/GRU recebem embeddings destes frames, não imagens. Rótulos não foram alterados.\n\n"
        + "\n".join(f"- [{e['relative_path']}]({e['figure']}) — {e['class_name']}, {e['split']}." for e in examples) + "\n", encoding="utf-8")
    return report
