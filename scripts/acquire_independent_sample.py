"""Blind, stratified sample from an AV-FFIA archive using verified HTTP byte ranges.

Only selected ZIP members are transferred; their CRC is checked by zipfile.
The full archive MD5 is recorded but not claimed to be verified.
"""
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/independent-evaluation'
URL = 'https://zenodo.org/api/records/11060195/files/video_6.27-6.29.zip/content'
SIZE = 13049105990
SEED = 'feedwatch-independent-20261022-v1'


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.transferred = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position + offset if whence == 1 else SIZE + offset)
        if self.position < 0: raise ValueError('Negative offset')
        return self.position

    def read(self, size=-1):
        size = min(SIZE - self.position, size if size >= 0 else SIZE - self.position)
        if size <= 0: return b''
        if size > 24 * 1024 * 1024: raise ValueError('Unexpectedly large remote member; abort instead of downloading the archive')
        start = self.position
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{start+size-1}', 'User-Agent': 'FeedWatch research'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{start+size-1}/{SIZE}':
                        raise ValueError('Server did not honor the exact requested byte range')
                    data = response.read(size+1)
                if len(data) != size: raise ValueError('Truncated byte range')
                self.position += size
                self.transferred += size
                return data
            except Exception:
                if attempt == 2: raise
                time.sleep(1)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        strata = defaultdict(list)
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace('\\', '/')).parts
            if not parts or not parts[-1].lower().endswith('.mp4'): continue
            date_position = next((i for i, p in enumerate(parts) if p in ('2022_6_27','2022_6_28','2022_6_29')), None)
            if date_position is None: continue
            relative = PurePosixPath(*parts[date_position:])
            if len(relative.parts) != 4: continue
            date, group, label, _ = relative.parts
            if label not in ('none','weak','medium','strong'): continue
            strata[date, group, label].append((relative.as_posix(), member))
        selected = []
        for key, entries in sorted(strata.items()):
            entries.sort(key=lambda item: hashlib.sha256(f'{SEED}:{item[0]}'.encode()).hexdigest())
            selected.extend(entries[:2])
        if not selected: raise ValueError('No labeled videos found')
        selection = {'source_url': URL, 'record': 'https://zenodo.org/records/11060195', 'license': 'CC-BY-4.0',
                     'archive_bytes': SIZE, 'archive_publisher_md5': '5c238cad9dc717135692daeb429ebeab',
                     'full_archive_checksum_verified': False,
                     'seed': SEED, 'policy': 'two members per day/group/class, ranked by SHA256(seed:path), before inference',
                     'expected_strata': len(strata), 'selected_count': len(selected),
                     'by_date_class': dict(Counter(f'{Path(path).parts[0]}/{Path(path).parts[2]}' for path, _ in selected)),
                     'members': [{'relative_path': path, 'zip_member': member.filename, 'crc32': member.CRC,
                                  'size': member.file_size, 'compressed_size': member.compress_size} for path, member in selected]}
        manifest = OUTPUT / 'selection.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != selection: raise ValueError('Selection changed')
        else: manifest.write_text(json.dumps(selection, indent=2), encoding='utf-8')
        print(f'Fixed {len(selected)} clips across {len(strata)} strata before inference', flush=True)
        hashes = {}
        for i, (relative, member) in enumerate(selected, 1):
            target = OUTPUT / 'videos' / relative
            if not target.exists():
                payload = archive.read(member)  # ZipFile verifies CRC, including decompression errors.
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_suffix('.pending')
                pending.write_bytes(payload)
                pending.replace(target)
            payload = target.read_bytes()
            import zlib
            if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC: raise ValueError(relative)
            hashes[relative] = hashlib.sha256(payload).hexdigest()
            if i % 10 == 0 or i == len(selected): print(f'Downloaded and CRC verified {i}/{len(selected)}; transfer {remote.transferred/1e6:.1f} MB', flush=True)
        (OUTPUT / 'acquisition.json').write_text(json.dumps({'status':'complete','sha256':hashes,
            'transferred_bytes_this_run':remote.transferred,'all_members_crc_verified':True,'full_archive_md5_verified':False},indent=2), encoding='utf-8')


if __name__ == '__main__': main()
