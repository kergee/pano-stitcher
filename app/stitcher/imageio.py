"""Unicode-safe file I/O and immutable input manifests; no Web dependency."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray, params=()) -> None:
    path = Path(path)
    ok, data = cv2.imencode(path.suffix, image, list(params))
    if not ok or not data.size:
        raise OSError(f"图片编码失败：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a failed write leaving an apparently valid result.
    fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent)
    temp=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            if stream.write(data.tobytes()) != data.nbytes:
                raise OSError(f"图片写入不完整：{path}")
        temp.replace(path)
        if path.stat().st_size != data.nbytes:
            raise OSError(f"图片写入大小不一致：{path}")
    finally:
        temp.unlink(missing_ok=True)


def save_manifest(folder: Path, paths: list[Path], **metadata) -> Path:
    entries = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    target = folder / 'inputs.json'
    target.write_text(json.dumps({**metadata, 'inputs': entries}, ensure_ascii=False, indent=2), encoding='utf-8')
    return target


def manifest_paths(path: Path) -> list[Path]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    result = []
    for entry in doc['inputs']:
        p = path.parent / entry['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"输入哈希不一致：{p}")
        result.append(p)
    if not result or len(set(p.resolve() for p in result)) != len(result):
        raise ValueError('输入清单为空或重复')
    return result
