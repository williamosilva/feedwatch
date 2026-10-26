"""Verified, bounded HTTP range access to published research ZIP members."""

import io
import http.client
import time
import urllib.request
import zipfile
from pathlib import Path


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, size: int):
        if not url.startswith("https://") or size <= 0:
            raise ValueError("An HTTPS archive and a positive published size are required")
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        target = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if not 0 <= target <= self.size:
            raise ValueError("Seek outside published archive")
        self.position = target
        return target

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0: return b""
        if size > 24*1024*1024:
            raise ValueError("Refusing a large range instead of accidentally downloading an archive")
        start = self.position
        for attempt in range(4):
            try:
                request = urllib.request.Request(self.url, headers={
                    "Range": f"bytes={start}-{start+size-1}", "User-Agent": "FeedWatch research"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{start+size-1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact requested byte range")
                    payload = response.read(size+1)
                if len(payload) != size: raise ValueError("Truncated or oversized range")
                self.position += size
                self.transferred += size
                return payload
            except (OSError, ValueError, http.client.HTTPException):
                if attempt == 3: raise
                time.sleep(1+attempt)


def verified_member(url: str, size: int, member: zipfile.ZipInfo, target: Path) -> dict:
    """Read a sealed ZipInfo without fetching the central directory again."""
    import hashlib
    import struct
    import zlib
    transferred = 0
    if target.exists():
        payload = target.read_bytes()
    else:
        remote = RemoteZip(url, size)
        if member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("Only unencrypted stored/deflated research members are supported")
        if not 0 < member.file_size <= 24*1024*1024:
            raise ValueError("Unexpected member size")
        remote.seek(member.header_offset)
        header = struct.unpack("<4s5H3I2H", remote.read(30))
        signature, _, flags, compression, _, _, _, _, _, name_length, extra_length = header
        if signature != b"PK\x03\x04" or compression != member.compress_type or flags != member.flag_bits:
            raise ValueError("Local ZIP header differs from the sealed directory")
        raw = remote.read(name_length+extra_length+member.compress_size)
        local_name = raw[:name_length].decode("utf-8" if flags & 0x800 else "cp437")
        if local_name != member.filename:
            raise ValueError("ZIP local filename differs from the sealed directory")
        compressed = raw[name_length+extra_length:]
        if compression == zipfile.ZIP_STORED:
            payload = compressed
        else:
            decoder = zlib.decompressobj(-15)
            payload = decoder.decompress(compressed, member.file_size+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("Invalid or oversized compressed member")
        transferred = remote.transferred
    if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
        raise ValueError(f"Size/CRC mismatch: {member.filename}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(".pending")
        pending.write_bytes(payload)
        pending.replace(target)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "crc_verified": True, "transferred_bytes": transferred}
