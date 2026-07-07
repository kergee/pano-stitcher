"""命令行入口：python -m app.stitcher.cli photos/ -o output/pano.jpg --engine pro
用于快速验证与引擎对比，网页界面走 app.main。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .baseline import stitch_baseline
from .pipeline import StitchError, stitch_pro

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(folder: Path):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"目录 {folder} 里没有图片")
    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
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
    args = ap.parse_args()

    images = load_images(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def report(stage, frac):
        print(f"  [{frac * 100:5.1f}%] {stage}")

    engines = ["baseline", "pro"] if args.engine == "both" else [args.engine]
    for eng in engines:
        print(f"\n=== 引擎：{eng} ===")
        t0 = time.time()
        try:
            if eng == "baseline":
                res = stitch_baseline(images)
                out = args.output if args.engine != "both" else args.output.with_stem(args.output.stem + "_baseline")
                extra = ""
            else:
                res = stitch_pro(images, backend_name=args.backend, projection=args.projection, progress=report)
                out = args.output if args.engine != "both" else args.output.with_stem(args.output.stem + "_pro")
                extra = f"，后端 {res.backend}，焦距 {res.focal:.0f}px"
                if res.dropped:
                    extra += f"，丢弃未连接图片 {res.dropped}"
        except StitchError as e:
            print(f"失败：{e}")
            continue
        cv2.imwrite(str(out), res.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"完成：{out}  {res.image.shape[1]}x{res.image.shape[0]}，{time.time() - t0:.1f} 秒{extra}")


if __name__ == "__main__":
    main()
