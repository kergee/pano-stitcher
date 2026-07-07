"""引擎 1：基线（复刻 AutoStitch）。

直接使用 cv2.Stitcher —— 这是 AutoStitch 论文作者 M. Brown 的算法在 OpenCV
中的官方实现（特征匹配 + RANSAC + 光束法平差 + 球面投影 + 多频段融合），
输出质量与 AutoStitch.exe 同源同级，作为质量基准。

C++ 迁移对照：cv::Stitcher::create(cv::Stitcher::PANORAMA)，API 完全一致。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import autocrop
from .pipeline import StitchError

_STATUS_MSG = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "有效重叠图片不足，无法拼接（请确认相邻照片有 30% 以上重叠）",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "单应性估计失败（图片间可能缺少足够的匹配特征）",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相机参数优化失败",
}


@dataclass
class BaselineResult:
    image: np.ndarray
    seconds: float
    autocrop: tuple[int, int, int, int] | None = None  # 建议裁剪框 (x,y,w,h)


def stitch_baseline(images: list[np.ndarray]) -> BaselineResult:
    """用 OpenCV 内置 Stitcher 拼接一组 BGR 图片。"""
    if len(images) < 2:
        raise StitchError("至少需要 2 张图片")
    t0 = time.time()
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise StitchError(_STATUS_MSG.get(status, f"拼接失败（状态码 {status}）"))
    rect = autocrop.largest_inner_rect(autocrop.valid_mask_from_image(pano))
    return BaselineResult(image=pano, seconds=time.time() - t0, autocrop=rect)
