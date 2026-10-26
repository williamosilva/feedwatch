"""Fixed official example selection, independent of food predictions or source labels."""

import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data import file_hash, utc_now, write_json
from .video_export import hidden_process, media_info, verify_export
from .video_inference import inspect_video, open_capture

PAGE = 'https://fishmaster93.github.io/UFFIA_demopage/'
URL = PAGE + 'FFIA/Medium.mp4'
SHA = '9e0ca5419b337109785a3b9fc28b997a92b3ba491d873beb25b1532a5984a6f1'
DIRECTORY = Path('artifacts/examples/local-app')
EXAMPLE = DIRECTORY / 'exemplo-continuo-12s.mp4'


def prepare_example(directory=DIRECTORY):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    original=directory/'original-official.mp4'
    target=directory/EXAMPLE.name
    record_path=directory/'source.json'
    if not original.exists():
        available=Path('artifacts/refinement-sources/Medium.mp4')
        pending=directory/'download.pending'
        if available.exists() and file_hash(available)==SHA:
            shutil.copy2(available,pending)
            origin='verified existing download from official URL'
        else:
            with urlopen(URL,timeout=30) as response,pending.open('wb') as output:
                shutil.copyfileobj(response,output)
            origin='downloaded from official URL'
        if file_hash(pending)!=SHA:raise ValueError('O arquivo oficial mudou; requer nova inspeção')
        pending.replace(original)
    else:origin='preserved original, SHA-256 verified'
    if file_hash(original)!=SHA:raise ValueError('Original do exemplo foi alterado')
    if target.exists():
        record=json.loads(record_path.read_text(encoding='utf-8'))
        if file_hash(target)!=record['trimmed_sha256']:raise ValueError('Exemplo preparado foi alterado')
        return target,record_path
    cap=open_capture(original)
    meta=inspect_video(cap)
    count=0;previous=None;differences=[];tiles=[]
    chosen={0,120,165,210,255,300,345,390,435,479,480,510}
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if image.shape[:2]!=(1080,1920):raise ValueError('Resolução inesperada no original')
            small=cv2.resize(image,(480,270))
            if previous is not None and 120<=count<480:
                differences.append({'frame':count,'mean_absolute_pixel_difference':float(np.abs(small.astype(float)-previous).mean())})
            if count in chosen:
                tile=Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
                ImageDraw.Draw(tile).text((8,8),f'Original: {count/30:.2f} s',fill='white',stroke_width=1,stroke_fill='black')
                tiles.append(tile)
            previous=small.astype(float);count+=1
    finally:cap.release()
    if count!=511 or meta['frame_count']!=511 or meta['fps']!=30.:raise ValueError('Duração/FPS/decodificação inesperados')
    sheet=Image.new('RGB',(480*3,270*4))
    for i,tile in enumerate(tiles):sheet.paste(tile,(i%3*480,i//3*270))
    sheet.save(directory/'source-contact-sheet.jpg')
    write_json(directory/'adjacent-frame-differences.json',differences)
    media=media_info(original)
    pending=directory/'trim.pending.mp4'
    command=[media['ffmpeg'],'-hide_banner','-loglevel','warning','-n','-i',str(original),
             '-ss','4','-t','12','-map','0:v:0','-map','0:a:0?','-c:v','libx264','-preset','veryfast',
             '-crf','20','-threads','2','-r','30','-fps_mode','cfr','-c:a','aac','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(pending)]
    with (directory/'trim-encoder.log').open('wb') as log:
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=log,timeout=60,**hidden_process())
    expected={**meta,'frame_count':360,'duration_seconds':12.}
    verification=verify_export(pending,expected)
    pending.replace(target)
    write_json(record_path,{'created_at_utc':utc_now(),'official_page':PAGE,'source_url':URL,
        'acquisition':origin,'original_file':str(original.resolve()),'source_sha256':SHA,
        'original_metadata':meta,'original_full_decoded_frames':count,
        'source_start_seconds':4.,'source_stop_seconds':16.,'source_frames_half_open':[120,480],
        'visible_date':'2022-06-14','reserved_test_date':'2022-06-22',
        'selection_basis':'fixed 4–16 s requested before inference; no outcome filtering, no concatenation',
        'content_review':'contact sheet and full chronological playback required; adjacent differences are diagnostics, not proof of no cuts',
        'feeding_usage':'demonstration only; filename and author labels are not model inputs or verified per-window ground truth',
        'trim_command':command,'audio':'first original stream re-encoded AAC if available',
        'trimmed_sha256':file_hash(target),'output':str(target.resolve()),'verification':verification})
    return target,record_path
