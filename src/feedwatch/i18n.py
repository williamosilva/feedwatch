"""English-first presentation with explicit Portuguese source strings.

Only display text and accessibility attributes are translated. Model outputs,
identifiers, URLs and form values retain their original contracts.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from html import escape
from html.parser import HTMLParser
from pathlib import Path

ASSETS = Path(__file__).parent / 'assets'
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
SKIP = {'script', 'style', 'code', 'pre', 'noscript', 'textarea', 'svg'}
TEXT_ONLY = {'title', 'option'}
TEXT_ATTRIBUTES = {'alt', 'aria-label', 'aria-description', 'title', 'placeholder'}
GRAPH_NAMES = ('operational-learning', 'operational-learning-mobile', 'class-recall',
               'class-recall-mobile', 'diverse-learning', 'diverse-learning-mobile',
               'finetune-learning', 'finetune-learning-mobile')
_expanded = {}


def normalize_text(value):
    return re.sub(r'\s+', ' ', value).strip()


@lru_cache(maxsize=1)
def _catalog():
    value = json.loads((ASSETS / 'translations.json').read_text(encoding='utf-8'))
    for name in ('dynamic-translations.json', 'local-translations.json'):
        path = ASSETS / name
        if path.exists():
            extra = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(extra, dict):
                raise ValueError('Additional translations must be a dictionary')
            value.update(extra)
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError('Translations must map Portuguese strings to English strings')
    return value


def translation_catalog():
    return {**_catalog(), **_expanded}


def _number(value):
    if ',' in value:
        return value.replace('.', '').replace(',', '.')
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+', value):
        return value.replace('.', ',')
    return value


@lru_cache(maxsize=1)
def _templates():
    result = []
    for source, target in _catalog().items():
        if '{' not in source:
            continue
        pieces = re.split(r'(\{[a-zA-Z_][a-zA-Z_0-9]*\})', source)
        expression = ''.join('(?P<' + piece[1:-1] + r'>[-+]?\d[\d.,]*%?)' if piece.startswith('{')
                             else re.escape(piece) for piece in pieces)
        result.append((re.compile(expression), target))
    return result


def translate_text(value):
    source = normalize_text(value)
    if source in _expanded:
        return _expanded[source]
    if source in _catalog():
        return _catalog()[source]
    for pattern, target in _templates():
        match = pattern.fullmatch(source)
        if match:
            translated = target.format(**{key: _number(number) for key, number in match.groupdict().items()})
            _expanded[source] = translated
            return translated
    if re.fullmatch(r'[-+]?\d[\d.,]*(?:%|\s+s)?', source):
        translated = _number(source)
        if translated != source:
            _expanded[source] = translated
        return translated
    return source


def _tag(tag, attrs, close=False):
    content = ''.join(' ' + name if value is None else ' ' + name + '="' + escape(value, quote=True) + '"'
                      for name, value in attrs)
    return '<' + tag + content + (' />' if close else '>')


class _Localizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output, self.stack, self.untranslated = [], [], []
        self.translated_fields = 0

    def handle_decl(self, decl):
        self.output.append('<!' + decl + '>')

    def handle_comment(self, data):
        self.output.append('<!--' + data + '-->')

    def handle_pi(self, data):
        self.output.append('<?' + data + '>')

    def _start(self, tag, attrs, self_closing=False):
        values = dict(attrs)
        inherited = self.stack[-1]['skip'] if self.stack else False
        skip = inherited or tag in SKIP or values.get('translate') == 'no' or 'data-fw-i18n' in values
        if tag == 'html':
            attrs = [(key, 'en' if key == 'lang' else value) for key, value in attrs]
        attr_sources = {}
        updated = []
        for key, value in attrs:
            should_translate = key in TEXT_ATTRIBUTES or tag == 'meta' and key == 'content' and values.get('name') == 'description'
            if value and should_translate and not skip and 'data-fw-i18n-attrs' not in values:
                source = normalize_text(value)
                translated = translate_text(source)
                if translated != source or source in _catalog():
                    attr_sources[key] = value
                    value = translated
                    self.translated_fields += 1
                else:
                    self.untranslated.append(source)
            updated.append((key, value))
        for key, marker in (('src', 'data-fw-graph-pt'), ('srcset', 'data-fw-graph-srcset-pt')):
            if tag not in {'img', 'source'} or not values.get(key) or marker in values:
                continue
            source = values[key]
            pattern = r'(' + '|'.join(re.escape(name) for name in GRAPH_NAMES) + r')\.svg(?=[\s,?#]|$)'
            translated = re.sub(pattern, r'\1.en.svg', source)
            if translated != source:
                updated = [(attr, translated if attr == key else item) for attr, item in updated]
                updated.append((marker, source))
        if attr_sources:
            updated.append(('data-fw-i18n-attrs', json.dumps(attr_sources, ensure_ascii=False, separators=(',', ':'))))
        frame = dict(tag=tag, skip=skip, attrs=updated, text_only=tag in TEXT_ONLY and not skip,
                     output_index=len(self.output), text=[])
        self.output.append(_tag(tag, updated, self_closing))
        if tag not in VOID and not self_closing:
            self.stack.append(frame)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, True)

    def handle_endtag(self, tag):
        frame = self.stack.pop() if self.stack and self.stack[-1]['tag'] == tag else None
        if frame and frame['text_only']:
            original = ''.join(frame['text'])
            source = normalize_text(original)
            translated = translate_text(source)
            if translated != source or source in _catalog():
                attrs = frame['attrs'] + [('data-fw-i18n', source)]
                self.output[frame['output_index']] = _tag(tag, attrs)
                self.translated_fields += 1
                self.output.append(escape(translated, quote=False))
            else:
                self.output.append(escape(original, quote=False))
                if source:
                    self.untranslated.append(source)
        self.output.append('</' + tag + '>')

    def handle_data(self, data):
        frame = self.stack[-1] if self.stack else None
        if frame and frame['text_only']:
            frame['text'].append(data)
            return
        if frame and frame['tag'] in {'script', 'style'}:
            self.output.append(data)
            return
        source = normalize_text(data)
        if not source or frame and frame['skip']:
            self.output.append(escape(data, quote=False))
            return
        translated = translate_text(source)
        if translated != source or source in _catalog():
            leading = data[:len(data) - len(data.lstrip())]
            trailing = data[len(data.rstrip()):]
            self.output.append(escape(leading, quote=False) + '<span data-fw-i18n="' + escape(source, quote=True)
                               + '">' + escape(translated, quote=False) + '</span>' + escape(trailing, quote=False))
            self.translated_fields += 1
        else:
            self.output.append(escape(data, quote=False))
            if re.search(r'\w', source):
                self.untranslated.append(source)


def localize_html(markup):
    """Render English once, retaining Portuguese sources for the language toggle."""
    parser = _Localizer()
    parser.feed(markup)
    parser.close()
    return ''.join(parser.output)


def translation_coverage(markup):
    """Report unregistered display fields without examining URLs or model data."""
    parser = _Localizer()
    parser.feed(markup)
    parser.close()
    return dict(translated_fields=parser.translated_fields, untranslated=list(dict.fromkeys(parser.untranslated)))


def i18n_javascript():
    payload = json.dumps(translation_catalog(), ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    return (ASSETS / 'i18n.js').read_text(encoding='utf-8').replace('__FEEDWATCH_TRANSLATIONS__', payload)
