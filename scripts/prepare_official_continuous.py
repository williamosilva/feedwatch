"""Download the fixed official demonstration and trim a real continuous 12 s interval."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from feedwatch.data import file_hash,write_json
from feedwatch.video_export import H264Writer,verify_export
from feedwatch.video_inference import open_capture,inspect_video

URL='https://fishmaster93.github.io/UFFIA_demopage/FFIA/Weak.mp4'
EXPECTED='d034775ce4df494e44edc5493ef134389d01b689225e1a6413959b4320efba05'


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    source=output/'Weak.mp4';target=output/'weak_continuous_12s.mp4'
    if not source.exists():
        with urlopen(URL,timeout=60) as response:payload=response.read()
        import hashlib
        if hashlib.sha256(payload).hexdigest()!=EXPECTED:raise ValueError('Official source changed; inspect before using')
        source.write_bytes(payload)
    if file_hash(source)!=EXPECTED:raise ValueError('Source hash differs')
    cap=open_capture(source);meta=inspect_video(cap)
    expected={**meta,'frame_count':360,'duration_seconds':12.}
    if target.exists():
        cap.release()
        record=json.loads((output/'selection_protocol.json').read_text(encoding='utf-8'))
        if record['trimmed_sha256']!=file_hash(target):raise ValueError('Existing trim changed')
        verify_export(target,expected)
        print('Existing source/trim verified; no overwrite:',target)
        return
    pending=output/'trim.pending.mp4'
    writer=H264Writer(pending,source,expected,False,output/'trim-encoder.log')
    try:
        for index in range(540):
            ok,image=cap.read()
            if not ok:raise ValueError(f'Decode failed at {index}')
            if index>=180:writer.write(image)
    finally:
        cap.release();writer.close()
    verification=verify_export(pending,expected);pending.replace(target)
    write_json(output/'selection_protocol.json',{'official_page':'https://fishmaster93.github.io/UFFIA_demopage/',
        'source_url':URL,'source_sha256':EXPECTED,'source_start_seconds':6.,'source_stop_seconds':18.,
        'source_frames_half_open':[180,540],'visible_date':'2022-06-14','reserved_test_date':'2022-06-22',
        'selection_basis':'fixed 6-18 s after the title, before feeding predictions; no concatenation',
        'feeding_usage':'exploratory; day, framing and author video encoding differ from evaluated conditions',
        'output':str(target),'trimmed_sha256':file_hash(target),'verification':verification})
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/refinement-sources'))
    prepare(parser.parse_args().output_dir)
