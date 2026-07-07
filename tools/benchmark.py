"""后端对比评测：SIFT vs DISK+LightGlue 在 5 个受控场景下的成功率。

判定标准（真实焦距 560px、5 张图全部应连接）：
  成功 = 全部图片连入全景 且 焦距误差 < 5%
  部分 = 有图片被丢弃但仍出图
  失败 = 抛出 StitchError

用法：python tools/benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402

from app.stitcher.pipeline import StitchError, stitch_pro  # noqa: E402
from tools.make_testset import FOCAL, SCENARIOS, generate  # noqa: E402

BENCH_DIR = Path("photos_bench")


def load(folder: Path):
    return [cv2.imread(str(p)) for p in sorted(folder.glob("*.jpg"))]


def run_one(images, backend: str) -> dict:
    t0 = time.time()
    try:
        res = stitch_pro(images, backend_name=backend)
    except StitchError as e:
        return {"status": "失败", "detail": str(e)[:24], "seconds": time.time() - t0}
    ferr = abs(res.focal - FOCAL) / FOCAL
    if res.dropped:
        status = f"部分（丢 {len(res.dropped)} 张）"
    elif ferr < 0.05:
        status = "成功"
    else:
        status = f"几何差（焦距误差 {ferr * 100:.0f}%）"
    return {"status": status, "focal": res.focal, "seconds": res.seconds,
            "size": f"{res.image.shape[1]}x{res.image.shape[0]}"}


def main():
    from app.stitcher.features import get_backend
    get_backend("deep")  # 预热：模型加载不计入单场景耗时

    rows = []
    for name in SCENARIOS:
        folder = BENCH_DIR / name
        if not folder.exists():
            generate(name, folder)
        images = load(folder)
        cell = {}
        for backend in ("sift", "deep"):
            r = run_one(images, backend)
            extra = f"，焦距 {r['focal']:.0f}" if "focal" in r else f"：{r.get('detail', '')}"
            cell[backend] = f"{r['status']}（{r['seconds']:.1f}s{extra}）"
            print(f"[{name}/{backend}] {cell[backend]}")
        rows.append((name, cell["sift"], cell["deep"]))

    print("\n| 场景 | SIFT | DISK+LightGlue |")
    print("|---|---|---|")
    for name, s, d in rows:
        print(f"| {name} | {s} | {d} |")


if __name__ == "__main__":
    main()
