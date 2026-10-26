"""Sobreposição de apresentação: não altera os frames recebidos pelo extrator."""

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .temporal_examples import effective_crop_box

NAVY = (20, 45, 57)
INK = (235, 246, 247)
MINT = (170, 221, 211)
MUTED = (190, 210, 216)


def font_file(bold=False) -> str:
    candidates = (["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                  ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise OSError("Fonte Unicode não encontrada; instale Segoe UI ou DejaVu Sans para preservar os acentos")


def interval(item: dict) -> str:
    return f"{item['start_seconds']:.2f}–{item['end_seconds']:.2f} s".replace(".", ",")


class OverlayRenderer:
    def __init__(self, width: int, height: int, extraction: dict, transform):
        self.width, self.height = width, height
        self.scale = max(.4, min(width/1280, height/720))
        self.regular, self.bold = font_file(), font_file(True)
        self.roi, self.crop = effective_crop_box(width, height, extraction, transform)
        self.key, self.layer = None, None

    def font(self, size, bold=False):
        return ImageFont.truetype(self.bold if bold else self.regular, max(9, round(size*self.scale)))

    def _layer(self, state, mode):
        layer = Image.new("RGBA", (self.width, self.height))
        draw, s = ImageDraw.Draw(layer), self.scale
        left, top = round(22*s), round(22*s)
        panel_width, panel_height = round(600*s), round(211*s)
        draw.rounded_rectangle((left, top, left+panel_width, top+panel_height), radius=round(12*s), fill=(*NAVY, 235))
        x, y = left+round(21*s), top+round(14*s)
        draw.text((x, y), "FeedWatch", font=self.font(23, True), fill=INK)
        draw.text((x+round(145*s), y+round(9*s)), "PROTÓTIPO EXPERIMENTAL", font=self.font(11), fill=MINT)
        window = state.get("window")
        if state["status"] == "prediction":
            title = f"{window['predicted_class']}  ·  {window['legend_pt']}"
        elif state["status"] == "unavailable":
            title = "Análise indisponível"
        else:
            title = "Analisando"
        draw.text((x, top+round(48*s)), title, font=self.font(33, True), fill=INK)
        mode_text = "Análise do clipe completo" if mode == "clip" else "Janelas de 2 s · extensão experimental"
        draw.text((x, top+round(100*s)), mode_text, font=self.font(17, True), fill=MINT)
        if window is not None:
            prefix = "Intervalo indisponível: " if state["status"] == "unavailable" else ("Intervalo analisado: " if mode == "clip" else "Janela de origem: ")
            caption = prefix+interval(window)
        else:
            caption = "Primeira janela: 0,00–2,00 s · em análise"
        draw.text((x, top+round(132*s)), caption, font=self.font(15), fill=INK)
        if state["tail"]:
            note = "Trecho final incompleto: não analisado; sem preenchimento."
        elif state["status"] == "prediction" and mode == "sequential":
            note = "Última previsão concluída · sem suavização"
        elif state["status"] == "unavailable":
            previous = state.get("last_valid")
            note = (f"Última válida: {previous['predicted_class']} ({interval(previous)})" if previous else
                    "Sem resultado válido para este intervalo. Veja o registro.")
        elif mode == "clip":
            note = "Modo offline: usa o clipe antes da reprodução."
        else:
            note = "Sem antecipar a classificação de frames futuros."
        draw.text((x, top+round(170*s)), note, font=self.font(14), fill=MUTED)
        # Quatro cantos do recorte efetivo: referência de imagem, nunca detecção de peixe.
        l, t, r, b = self.crop
        segment, line = round(21*s), max(1, round(1.2*s))
        for cx, cy, dx, dy in ((l, t, 1, 1), (r, t, -1, 1), (l, b, 1, -1), (r, b, -1, -1)):
            draw.line([(cx+dx*segment, cy), (cx, cy), (cx, cy+dy*segment)], fill=(*MINT, 210), width=line)
        crop_text = "Recorte da imagem · área usada pelo modelo"
        font = self.font(12)
        box = draw.textbbox((0, 0), crop_text, font=font)
        tx, ty = max(8, int((l+r-(box[2]-box[0]))/2)), min(self.height-round(84*s), int(b-26*s))
        draw.rectangle((tx-5*s, ty-2*s, tx+box[2]+5*s, ty+19*s), fill=(*NAVY, 190))
        draw.text((tx, ty), crop_text, font=font, fill=INK)
        footer_top = self.height-round(61*s)
        draw.rectangle((0, footer_top, self.width, self.height), fill=(*NAVY, 233))
        draw.text((round(24*s), footer_top+round(6*s)), "Intensidade alimentar observada — protótipo experimental",
                  font=self.font(15, True), fill=INK)
        draw.text((round(24*s), footer_top+round(32*s)), "Não comprova fome ou saciedade · sem controle automático de alimentador",
                  font=self.font(12), fill=MUTED)
        return layer

    def render(self, bgr: np.ndarray, state: dict, mode: str) -> np.ndarray:
        window = state.get("window")
        previous = state.get("last_valid")
        key = (state["status"], state["tail"], mode, window["window_id"] if window else None,
               previous["window_id"] if previous else None)
        if key != self.key:
            self.layer, self.key = self._layer(state, mode), key
        base = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        return cv2.cvtColor(np.asarray(Image.alpha_composite(base, self.layer).convert("RGB")), cv2.COLOR_RGB2BGR)


def unavailable_image(path: Path, error: str):
    image = Image.new("RGB", (1280, 720), NAVY)
    draw = ImageDraw.Draw(image)
    draw.text((64, 74), "FeedWatch", fill=INK, font=ImageFont.truetype(font_file(True), 40))
    draw.text((64, 215), "Análise indisponível", fill=MINT, font=ImageFont.truetype(font_file(True), 55))
    draw.multiline_text((64, 310), textwrap.fill(error, 85), fill=INK, spacing=12,
                        font=ImageFont.truetype(font_file(), 23))
    draw.text((64, 600), "Nenhuma classe atribuída. O arquivo original permanece preservado.", fill=MUTED,
              font=ImageFont.truetype(font_file(), 22))
    image.save(path)
