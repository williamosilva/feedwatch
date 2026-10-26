"""English labels for saved learning charts; preserve every non-text SVG byte."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from xml.etree import ElementTree


NAMES = (
    "operational-learning", "operational-learning-mobile", "class-recall",
    "class-recall-mobile", "diverse-learning", "diverse-learning-mobile",
)
TEXT_NODES = re.compile(r"(<text\b[^>]*>)(.*?)(</text>)", re.DOTALL)
LABELS = {
    "Não observada": "Not observed",
    "Baixa": "Low",
    "Moderada": "Moderate",
    "Alta": "High",
    "Treino": "Training",
    "Validação": "Validation",
    "Vídeos de treino": "Training clips",
    "Vídeos de validação (outro dia)": "Validation clips (separate day)",
    "Etapa de treinamento (época)": "Training epoch",
    "Etapa de treinamento": "Training epoch",
    "Classificações corretas (%)": "Correct classifications (%)",
    "Acerto (%)": "Accuracy (%)",
    "Vídeos do nível corretamente identificados (%)": "Clips in each level correctly classified (%)",
    "Identificações corretas (%)": "Correctly classified (%)",
}


def english_label(value):
    if value in LABELS:
        return LABELS[value]
    for pattern, replacement in (
        (r"Versão escolhida: etapa (\d+)", r"Selected checkpoint: epoch \1"),
        (r"Etapa escolhida: (\d+)", r"Selected epoch: \1"),
        (r"Etapa (\d+)", r"Epoch \1"),
    ):
        if re.fullmatch(pattern, value):
            return re.sub(pattern, replacement, value)
    if re.fullmatch(r"[−+\-]?\d+(?:[,.]\d+)?%?", value):
        return value.replace(",", ".")
    raise ValueError(f"Learning-chart label needs an English translation: {value!r}")


def translate_svg(payload):
    source = payload.decode("utf-8")
    original_tree = ElementTree.fromstring(source)
    expected = len(original_tree.findall(".//{http://www.w3.org/2000/svg}text"))
    nodes = list(TEXT_NODES.finditer(source))
    if not expected or len(nodes) != expected:
        raise ValueError("Expected plain text nodes from Matplotlib with svg.fonttype=none")
    original_numbers, translated_numbers = [], []
    def replace(match):
        original = html.unescape(match.group(2))
        if "<" in match.group(2):
            raise ValueError("Nested text elements require a reviewed translation")
        translated = english_label(original)
        original_numbers.extend(re.findall(r"\d+(?:[,.]\d+)?%?", original.replace(",", ".")))
        translated_numbers.extend(re.findall(r"\d+(?:[,.]\d+)?%?", translated))
        return match.group(1)+html.escape(translated, quote=False)+match.group(3)
    translated = TEXT_NODES.sub(replace, source)
    ElementTree.fromstring(translated)
    # This covers paths, points, axis positions, scales, colors, annotations and
    # all metadata. Only the visible content of existing text nodes can differ.
    stripped = lambda value: TEXT_NODES.sub(lambda m: m.group(1)+m.group(3), value)
    if stripped(source) != stripped(translated) or original_numbers != translated_numbers:
        raise ValueError("Localization changed chart geometry or displayed values")
    return translated.encode("utf-8")


def write_english_svg(path):
    path = Path(path)
    if path.suffix != ".svg" or path.name.endswith(".en.svg"):
        raise ValueError("Supply an original Portuguese SVG")
    original = path.read_bytes()
    translated = translate_svg(original)
    target = path.with_name(path.stem+".en.svg")
    target.write_bytes(translated)
    if path.read_bytes() != original:
        raise ValueError("Original Portuguese chart changed")
    return {"source": path.name, "english": target.name,
            "portuguese_sha256": hashlib.sha256(original).hexdigest(),
            "english_sha256": hashlib.sha256(translated).hexdigest(),
            "only_text_changed": True, "displayed_values_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1]/"sites/feedwatch/dist/learning")
    args = parser.parse_args()
    print(json.dumps([write_english_svg(args.directory/(name+".svg")) for name in NAMES], indent=2))


if __name__ == "__main__":
    main()
