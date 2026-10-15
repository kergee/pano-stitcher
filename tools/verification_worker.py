"""Isolated, two-thread stitching measurement, including Windows peak working set."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import importlib.metadata

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--options', default='{}')
ap.add_argument('--engine', default='pro')
args = ap.parse_args()
sys.path.insert(0, str(args.source.resolve()))
import cv2
import numpy as np
from app.stitcher.pipeline import stitch_pro
from app.stitcher.baseline import stitch_baseline

cv2.setNumThreads(2)
cv2.setRNGSeed(0)
manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
images = []
for entry in manifest['inputs']:
    path = args.manifest.parent / entry['path']
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法解码 {path}')
    images.append(image)
options = json.loads(args.options)
row = {'engine': args.engine, 'options': options, 'inputs': manifest,
       'runtime': {'python':sys.version, 'opencv':cv2.__version__, 'numpy':np.__version__, 'scipy':importlib.metadata.version('scipy'), 'threads':2},
       'source':str(args.source.resolve()), 'source_hashes':{str(p.relative_to(args.source.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source.resolve()/'app/stitcher').glob('*.py')}, 'visual_quality':'pending human inspection','geometry_pass':None}
t0 = time.perf_counter()
try:
    res = stitch_pro(images, backend_name='sift', **options) if args.engine=='pro' else stitch_baseline(images, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.jpg', res.image, [cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok:
        raise OSError('encode failed')
    args.output.write_bytes(data.tobytes())
    row.update(generated=True, output=str(args.output), size=list(res.image.shape[1::-1]),
               focal=getattr(res,'focal',None), used=getattr(res,'used',None),
               dropped=getattr(res,'dropped',None), crop=res.autocrop,
               diagnostics=getattr(res,'diagnostics',{}))
    if 'focal' in manifest and row['focal'] is not None:
        row['focal_error']=abs(row['focal']-manifest['focal'])/manifest['focal']
        row['geometry_pass']=row['focal_error']<0.05 and len(row['used'])==len(images)
except Exception as e:
    row.update(generated=False, error=f'{type(e).__name__}: {e}', geometry_pass=False)
row['seconds']=time.perf_counter()-t0
if sys.platform=='win32':
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        row['peak_working_set_bytes']=counters.PeakWorkingSetSize
print(json.dumps(row,ensure_ascii=False))
raise SystemExit(0 if row['generated'] else 1)
