"""FastAPI 后端：托管单页前端 + /api/stitch 拼接接口。

拼接核心在 app.stitcher 包内，与本层严格解耦（迁移移动端时本层整体替换即可）。
启动：uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .stitcher import StitchError, stitch_baseline, stitch_pro

app = FastAPI(title="Pano")
STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()  # CPU 密集，同一时间只跑一个拼接任务

MAX_SIDE = 3000  # 上传图片超过此边长先降采样，控制内存与耗时


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


def _decode(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _run(engine: str, backend: str, projection: str, images: list[np.ndarray]) -> dict:
    with _lock:
        t0 = time.time()
        if engine == "baseline":
            res = stitch_baseline(images)
            meta = {}
        else:
            res = stitch_pro(images, backend_name=backend, projection=projection)
            meta = {"backend": res.backend, "focal": round(res.focal), "dropped": res.dropped}
        if res.autocrop:
            x, y, w, h = res.autocrop
            meta["autocrop"] = {"x": x, "y": y, "w": w, "h": h}
        ok, buf = cv2.imencode(".jpg", res.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise StitchError("结果编码失败")
        return {
            "ok": True,
            "engine": engine,
            "seconds": round(time.time() - t0, 1),
            "width": res.image.shape[1],
            "height": res.image.shape[0],
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
            **meta,
        }


@app.post("/api/stitch")
async def api_stitch(
    files: list[UploadFile] = File(...),
    engine: str = Form("pro"),
    backend: str = Form("auto"),
    projection: str = Form("spherical"),
):
    images = []
    for f in files:
        img = _decode(await f.read())
        if img is None:
            return JSONResponse({"ok": False, "error": f"无法解码图片：{f.filename}"}, status_code=400)
        images.append(img)
    if len(images) < 2:
        return JSONResponse({"ok": False, "error": "至少上传 2 张图片"}, status_code=400)
    try:
        return await run_in_threadpool(_run, engine, backend, projection, images)
    except StitchError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
    except Exception as e:  # 不让内部异常裸奔到前端
        return JSONResponse({"ok": False, "error": f"内部错误：{e}"}, status_code=500)
