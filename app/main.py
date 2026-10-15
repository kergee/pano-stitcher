"""FastAPI 后端：托管单页前端 + /api/stitch 拼接接口。

拼接核心在 app.stitcher 包内，与本层严格解耦（迁移移动端时本层整体替换即可）。
启动：uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .stitcher import StitchError, stitch_baseline, stitch_pro

app = FastAPI(title="Pano")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC = Path(__file__).parent / "static"
_stitch_lock = threading.Lock()  # CPU 密集，同一时间只跑一个拼接任务
_jobs_lock = threading.Lock()
_jobs: dict[str, "StitchJob"] = {}

MAX_SIDE = 3000  # 上传图片超过此边长先降采样，控制内存与耗时
JOB_TTL_SECONDS = 30 * 60


@dataclass
class StitchJob:
    events: queue.Queue[tuple[str, dict]]
    created: float = field(default_factory=time.time)
    result: dict | None = None
    done: bool = False


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


def _decode(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    scale = max_side / max(img.shape[:2])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _resize_output(
    image: np.ndarray,
    autocrop: tuple[int, int, int, int] | None,
    max_output_side: int,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if max_output_side <= 0:
        return image, autocrop
    scale = max_output_side / max(image.shape[:2])
    if scale >= 1.0:
        return image, autocrop
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if autocrop is None:
        return image, None
    x, y, w, h = autocrop
    rect = (
        max(0, int(round(x * scale))),
        max(0, int(round(y * scale))),
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return image, rect


def _run(
    engine: str,
    backend: str,
    projection: str,
    images: list[np.ndarray],
    max_output_side: int = 0,
    jpeg_quality: int = 92,
    progress=None,
) -> dict:
    progress = progress or (lambda _stage, _frac: None)
    max_output_side = _clamp(max_output_side, 0, 12000)
    jpeg_quality = _clamp(jpeg_quality, 60, 98)
    with _stitch_lock:
        t0 = time.time()
        if engine == "baseline":
            progress("基线拼接", 0.08)
            res = stitch_baseline(images, max_output_side=max_output_side)
            meta = {}
            progress("基线拼接完成", 0.92)
        elif engine == "pro":
            res = stitch_pro(images, backend_name=backend, projection=projection, progress=progress, max_output_side=max_output_side)
            meta = {"backend": res.backend, "focal": round(res.focal), "dropped": res.dropped, "diagnostics": res.diagnostics}
        else:
            raise StitchError(f"未知引擎：{engine}")

        progress("编码结果", 1.0)
        image, autocrop = _resize_output(res.image, res.autocrop, max_output_side)
        if autocrop:
            x, y, w, h = autocrop
            meta["autocrop"] = {"x": x, "y": y, "w": w, "h": h}
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise StitchError("结果编码失败")
        return {
            "ok": True,
            "engine": engine,
            "seconds": round(time.time() - t0, 1),
            "width": image.shape[1],
            "height": image.shape[0],
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
            "jpeg_quality": jpeg_quality,
            **meta,
        }


async def _read_images(files: list[UploadFile]) -> list[np.ndarray] | JSONResponse:
    images = []
    for f in files:
        img = _decode(await f.read())
        if img is None:
            return JSONResponse({"ok": False, "error": f"无法解码图片：{f.filename}"}, status_code=400)
        images.append(img)
    if len(images) < 2:
        return JSONResponse({"ok": False, "error": "至少上传 2 张图片"}, status_code=400)
    return images


@app.post("/api/stitch")
async def api_stitch(
    files: list[UploadFile] = File(...),
    engine: str = Form("pro"),
    backend: str = Form("auto"),
    projection: str = Form("spherical"),
    max_output_side: int = Form(0),
    jpeg_quality: int = Form(92),
):
    images = await _read_images(files)
    if isinstance(images, JSONResponse):
        return images
    try:
        return await run_in_threadpool(_run, engine, backend, projection, images, max_output_side, jpeg_quality)
    except StitchError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
    except Exception as e:  # 不让内部异常裸奔到前端
        return JSONResponse({"ok": False, "error": f"内部错误：{e}"}, status_code=500)


def _cleanup_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [job_id for job_id, job in _jobs.items() if now - job.created > JOB_TTL_SECONDS]
        for job_id in stale:
            _jobs.pop(job_id, None)


def _get_job(job_id: str) -> StitchJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_job(
    job_id: str,
    engine: str,
    backend: str,
    projection: str,
    images: list[np.ndarray],
    max_output_side: int,
    jpeg_quality: int,
) -> None:
    job = _get_job(job_id)
    if job is None:
        return

    def emit(stage: str, frac: float) -> None:
        job.events.put(("progress", {"stage": stage, "frac": round(float(frac), 3)}))

    try:
        result = _run(engine, backend, projection, images, max_output_side, jpeg_quality, emit)
    except StitchError as e:
        result = {"ok": False, "engine": engine, "error": str(e)}
    except Exception as e:
        result = {"ok": False, "engine": engine, "error": f"内部错误：{e}"}

    with _jobs_lock:
        job.result = result
        job.done = True
    job.events.put(("done", {"ok": bool(result.get("ok")), "engine": engine}))


@app.post("/api/stitch/jobs")
async def api_stitch_job(
    files: list[UploadFile] = File(...),
    engine: str = Form("pro"),
    backend: str = Form("auto"),
    projection: str = Form("spherical"),
    max_output_side: int = Form(0),
    jpeg_quality: int = Form(92),
):
    images = await _read_images(files)
    if isinstance(images, JSONResponse):
        return images
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    job = StitchJob(events=queue.Queue())
    with _jobs_lock:
        _jobs[job_id] = job
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, engine, backend, projection, images, max_output_side, jpeg_quality),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/stitch/jobs/{job_id}/events")
async def api_stitch_events(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"ok": False, "error": "任务不存在或已过期"}, status_code=404)

    def stream():
        if job.done and job.events.empty():
            yield _sse("done", {"ok": bool(job.result and job.result.get("ok"))})
            return
        while True:
            try:
                event, payload = job.events.get(timeout=15)
            except queue.Empty:
                yield _sse("ping", {})
                continue
            yield _sse(event, payload)
            if event == "done":
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/stitch/jobs/{job_id}/result")
async def api_stitch_result(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"ok": False, "error": "任务不存在或已过期"}, status_code=404)
    if not job.done:
        return JSONResponse({"ok": False, "error": "任务尚未完成"}, status_code=202)
    return JSONResponse(job.result)
