"""命令行入口：python -m app.stitcher.cli photos/ -o output/pano.jpg --engine pro
用于快速验证与引擎对比，网页界面走 app.main。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .imageio import read_image, write_image, manifest_paths
from .baseline import stitch_baseline
from .pipeline import StitchError, stitch_pro

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(folder: Path, manifest: Path | None = None, exclude=()):
    paths = manifest_paths(manifest) if manifest else sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS and p.resolve() not in exclude)
    if not paths:
        sys.exit(f"目录 {folder} 里没有图片")
    imgs = []
    for p in paths:
        im = read_image(p)
        if im is None:
            print(f"跳过无法读取的文件：{p.name}")
            continue
        imgs.append(im)
        print(f"读取 {p.name}  {im.shape[1]}x{im.shape[0]}")
    return imgs


def main():
    ap = argparse.ArgumentParser(description="全景拼接")
    ap.add_argument("folder", type=Path, help="照片目录")
    ap.add_argument("-o", "--output", type=Path, default=Path("output/pano.jpg"))
    ap.add_argument("--engine", choices=["baseline", "pro", "both"], default="pro")
    ap.add_argument("--backend", choices=["auto", "sift", "deep"], default="auto",
                    help="pro 引擎的特征后端")
    ap.add_argument("--projection", choices=["spherical", "cylindrical", "mercator", "plane"],
                    default="spherical", help="pro 引擎的投影方式（plane=直线不弯）")
    ap.add_argument("--max-output-side", type=int, default=0,
                    help="输出图最大边长，0 表示不限制")
    ap.add_argument("--jpeg-quality", type=int, default=92,
                    help="JPEG 质量（60-98）")
    ap.add_argument("--report", type=Path, help="保存成功及失败的结构化诊断")
    ap.add_argument("--manifest", type=Path, help="固定输入清单（路径及 SHA256）")
    ap.add_argument("--memory-mb", type=float, default=768, help="合成峰值内存估算预算 MiB")
    ap.add_argument('--quality',choices=['full','fast'],default='full')
    ap.add_argument('--ba-mode',choices=['single','staged'],default='single')
    ap.add_argument('--pair-mode',choices=['all','ordered'],default='all',help='ordered 仅用于已知拍摄顺序；断连时尝试全对兜底')
    ap.add_argument('--no-local-warp',action='store_true')
    args = ap.parse_args()

    if args.manifest is None and (args.folder/"inputs.json").exists():
        args.manifest=args.folder/"inputs.json"
    excluded = {args.output.resolve(), args.output.with_stem(args.output.stem + "_baseline").resolve(), args.output.with_stem(args.output.stem + "_pro").resolve()}
    try:
        images = load_images(args.folder, args.manifest, excluded)
    except (OSError, ValueError, cv2.error) as e:
        print(f"读取失败：{e}", file=sys.stderr)
        return 1
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"输出目录创建失败：{e}", file=sys.stderr)
        return 1

    def report(stage, frac):
        print(f"  [{frac * 100:5.1f}%] {stage}")

    engines = ["baseline", "pro"] if args.engine == "both" else [args.engine]
    failures = 0
    reports = []
    for eng in engines:
        print(f"\n=== 引擎：{eng} ===")
        t0 = time.time()
        try:
            if eng == "baseline":
                res = stitch_baseline(images, max_output_side=args.max_output_side, memory_mb=args.memory_mb)
                out = args.output if args.engine != "both" else args.output.with_stem(args.output.stem + "_baseline")
                extra = ""
            else:
                res = stitch_pro(images, backend_name=args.backend, projection=args.projection, progress=report, max_output_side=args.max_output_side, memory_mb=args.memory_mb, quality=args.quality, ba_mode=args.ba_mode, pair_mode=args.pair_mode, local_warp=not args.no_local_warp)
                out = args.output if args.engine != "both" else args.output.with_stem(args.output.stem + "_pro")
                extra = f"，后端 {res.backend}，焦距 {res.focal:.0f}px"
                if res.dropped:
                    extra += f"，丢弃未连接图片 {res.dropped}"
        except (StitchError, RuntimeError, ValueError, cv2.error) as e:
            reports.append({"engine":eng,"ok":False,"error":str(e)})
            print(f"失败：{e}")
            failures += 1
            continue
        image = res.image
        if args.max_output_side > 0:
            scale = args.max_output_side / max(image.shape[:2])
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        quality = max(60, min(98, args.jpeg_quality))
        try:
            write_image(out, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        except (OSError, ValueError, cv2.error) as e:
            reports.append({"engine":eng,"ok":False,"error":str(e)})
            print(f"写入失败：{e}", file=sys.stderr)
            failures += 1
            continue
        reports.append({"engine":eng,"ok":True,"output":str(out),"backend":getattr(res,"backend",None),"used":getattr(res,"used",None),"dropped":getattr(res,"dropped",None),"diagnostics":getattr(res,"diagnostics",{}),"seconds":res.seconds})
        print(f"完成：{out}  {image.shape[1]}x{image.shape[0]}，{time.time() - t0:.1f} 秒{extra}")

    if args.report:
        try:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
        except OSError as e:
            print(f"诊断写入失败：{e}",file=sys.stderr)
            failures+=1
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
