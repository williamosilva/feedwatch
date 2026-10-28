"""Check the final static release links, media, ZIPs and download hashes."""
import hashlib
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT/'sites/feedwatch/dist'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        self.refs.extend(value for key, value in attrs if key in ('href', 'src'))


def main():
    missing = []
    checked = 0
    for path in DIST.rglob('*'):
        if path.suffix == '.html':
            parser = Links()
            parser.feed(path.read_text(encoding='utf-8'))
            refs = parser.refs
        elif path.suffix == '.md':
            refs = re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8'))
        else:
            continue
        for ref in refs:
            parts = urlsplit(ref)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            checked += 1
            if not (path.parent/unquote(parts.path)).resolve().is_file():
                missing.append([str(path.relative_to(DIST)), ref])
    for case in json.loads((DIST/'data.json').read_text(encoding='utf-8'))['cases']:
        for key in ('original', 'annotated', 'poster'):
            assert (DIST/case[key]).is_file(), case[key]
        motion=case.get('motion')
        if motion and motion.get('video'):
            path=DIST/motion['video']
            assert path.is_file(), motion['video']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==motion['video_sha256'], motion['video']
            assert len(motion['frames'])==motion['frame_count'], case['id']
            assert abs(motion['duration_seconds']-case['duration'])<=1/case['fps'], case['id']
    large = [(str(p.relative_to(DIST)), p.stat().st_size) for p in DIST.rglob('*')
             if p.is_file() and p.stat().st_size > 25*1024*1024]
    for name, meta in json.loads((DIST/'downloads.json').read_text()).items():
        path = DIST/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256'], name
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, name
    for path in (DIST/'evidence').rglob('*'):
        assert 'C:\\Users\\' not in path.read_text(encoding='utf-8'), path
    result = {'local_links_checked': checked, 'broken_links': missing,
              'assets_over_25_mib': large, 'download_hashes_and_zip_crc': 'pass'}
    print(json.dumps(result))
    assert not missing and not large


if __name__ == '__main__':
    main()
