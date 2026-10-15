"""后端对比评测：SIFT vs DISK+LightGlue 在 5 个受控场景下的成功率。

判定标准（真实焦距 560px、5 张图全部应连接）：
  成功 = 全部图片连入全景 且 焦距误差 < 5%
  部分 = 有图片被丢弃但仍出图
  失败 = 抛出 StitchError

用法：python tools/benchmark.py
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402

from app.stitcher.imageio import read_image, manifest_paths
from app.stitcher.pipeline import StitchError, stitch_pro  # noqa: E402
from tools.make_testset import FOCAL, SCENARIOS, generate  # noqa: E402

BENCH_DIR = Path("photos_bench")
FOCAL_TOLERANCE = 0.05
REGRESSION_CASES = (
    ("normal", "sift"),
    ("lowoverlap", "sift"),
    ("dark", "deep"),
)


def load(folder: Path):
    return [read_image(p) for p in manifest_paths(folder / "inputs.json")]


def run_one(images, backend: str, focal_tolerance: float = FOCAL_TOLERANCE) -> dict:
    t0 = time.time()
    try:
        res = stitch_pro(images, backend_name=backend)
    except (StitchError, RuntimeError, ValueError, cv2.error) as e:
        return {"ok": False, "status": "失败", "detail": str(e)[:80], "seconds": time.time() - t0}
    ferr = abs(res.focal - FOCAL) / FOCAL
    if res.dropped:
        status = f"部分（丢 {len(res.dropped)} 张）"
    elif ferr < focal_tolerance:
        status = "成功"
    else:
        status = f"几何差（焦距误差 {ferr * 100:.0f}%）"
    return {"ok": not res.dropped and ferr < focal_tolerance, "status": status,
            "focal": res.focal, "focal_error": ferr, "dropped": res.dropped, "seconds": res.seconds,
            "runtime":{"python":sys.version,"opencv":cv2.__version__,"threads":cv2.getNumThreads()}, "diagnostics": res.diagnostics, "actual_backend": res.backend, "used": res.used, "visual_quality": "未自动判定", "size": f"{res.image.shape[1]}x{res.image.shape[0]}"}


def ensure_dataset(name: str, bench_dir: Path) -> Path:
    folder = bench_dir / name
    if not (folder / "inputs.json").exists():
        generate(name, folder)
    return folder


def deep_weights_ready() -> bool:
    ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    disk = ckpt / "depth-save.pth"
    lightglue = [
        p for p in ckpt.glob("disk_lightglue*")
        if not p.name.endswith(".partial") and p.is_file() and p.stat().st_size > 1_000_000
    ]
    return disk.exists() and disk.stat().st_size > 1_000_000 and bool(lightglue)


def warmup(backends: set[str], allow_model_download: bool = False) -> bool:
    from app.stitcher.features import get_backend

    if "deep" not in backends and "auto" not in backends:
        return True
    if not allow_model_download and not deep_weights_ready():
        print("深度模型权重未完整缓存。首次下载请显式运行：", flush=True)
        print("  python tools/benchmark.py --regression --allow-model-download", flush=True)
        return False
    print("预热深度后端（首次运行可能下载 DISK/LightGlue 权重）...", flush=True)
    try:
        get_backend("deep")  # 预热：模型加载不计入单场景耗时
        return True
    except Exception as e:
        print(f"深度后端不可用：{e}")
        return False


def run_benchmark(scenarios: list[str], backends: list[str], bench_dir: Path, allow_model_download: bool) -> int:
    deep_ready = warmup(set(backends), allow_model_download)
    rows = []
    results = []
    for name in scenarios:
        folder = ensure_dataset(name, bench_dir)
        images = load(folder)
        cell = {}
        for backend in backends:
            r = run_one(images, backend) if backend == "sift" or deep_ready else {"ok": False, "status": "未验证", "seconds": 0, "detail": "深度后端未就绪"}
            results.append({"scenario": name, "requested_backend": backend, "inputs": json.loads((folder / "inputs.json").read_text(encoding="utf-8")), **r})
            extra = f"，焦距 {r['focal']:.0f}" if "focal" in r else f"：{r.get('detail', '')}"
            cell[backend] = f"{r['status']}（{r['seconds']:.1f}s{extra}）"
            print(f"[{name}/{backend}] {cell[backend]}")
        rows.append((name, cell))

    print("\n| 场景 | " + " | ".join(backends) + " |")
    print("|---|" + "|".join("---" for _ in backends) + "|")
    for name, cell in rows:
        print("| " + name + " | " + " | ".join(cell[b] for b in backends) + " |")
    (bench_dir / "benchmark-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(not r["ok"] for r in results))


def run_regression(bench_dir: Path, allow_model_download: bool) -> int:
    deep_ready = warmup({backend for _, backend in REGRESSION_CASES}, allow_model_download)
    failures = []
    results = []
    for scenario, backend in REGRESSION_CASES:
        folder = ensure_dataset(scenario, bench_dir)
        result = run_one(load(folder), backend) if backend == "sift" or deep_ready else {"ok": False, "status": "未验证", "seconds": 0, "detail": "深度后端未就绪"}
        results.append({"scenario":scenario,"requested_backend":backend,"inputs":json.loads((folder/"inputs.json").read_text(encoding="utf-8")),**result})
        extra = f"，焦距 {result['focal']:.0f}，误差 {result['focal_error'] * 100:.1f}%" if "focal" in result else f"：{result.get('detail', '')}"
        print(f"[回归 {scenario}/{backend}] {result['status']}（{result['seconds']:.1f}s{extra}）")
        if not result["ok"]:
            failures.append((scenario, backend, result["status"]))

    (bench_dir/"regression-results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    if failures:
        print("\n回归失败：")
        for scenario, backend, status in failures:
            print(f"- {scenario}/{backend}: {status}")
        return 1
    print("\n最小回归通过：normal/sift、lowoverlap/sift、dark/deep")
    return 0


def main():
    ap = argparse.ArgumentParser(description="SIFT vs DISK+LightGlue 拼接评测/回归")
    ap.add_argument("--regression", action="store_true",
                    help="运行最小回归：normal/sift、lowoverlap/sift、dark/deep，失败返回非零")
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    ap.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS),
                    help="普通评测要运行的场景")
    ap.add_argument("--backends", nargs="+", choices=["sift", "deep", "auto"], default=["sift", "deep"],
                    help="普通评测要运行的特征后端")
    ap.add_argument("--allow-model-download", action="store_true",
                    default=os.environ.get("PANO_ALLOW_MODEL_DOWNLOAD") == "1",
                    help="允许回归/评测在缺少深度权重时联网下载模型")
    ap.add_argument("--threads",type=int,default=2,help="OpenCV 线程数，便于复跑对照")
    args = ap.parse_args()
    if args.threads<1:ap.error("threads must be positive")
    cv2.setNumThreads(args.threads)

    if args.regression:
        raise SystemExit(run_regression(args.bench_dir, args.allow_model_download))
    raise SystemExit(run_benchmark(args.scenarios, args.backends, args.bench_dir, args.allow_model_download))


if __name__ == "__main__":
    main()
