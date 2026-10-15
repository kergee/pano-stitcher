"""OpenCV cv::Stitcher 基线：用于经典旋转全景算法族的质量对照。
不宣称这是 AutoStitch 软件的官方实现，也不假定输出质量与原 EXE 等价。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import autocrop
from .pipeline import StitchError
from .budget import ComposeBudget, plan_projection

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


def stitch_baseline(images: list[np.ndarray], max_output_side: int = 0, memory_mb: float = 768) -> BaselineResult:
    """用 OpenCV 内置 Stitcher 拼接一组 BGR 图片。"""
    try:
        budget=ComposeBudget(max_output_side=max_output_side,memory_mb=memory_mb)
    except ValueError as e:
        raise StitchError(str(e)) from e
    if len(images) < 2:
        raise StitchError("至少需要 2 张图片")
    if any(im is None or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<8 for im in images):
        raise StitchError("输入必须为 uint8 BGR 图片")
    if sum(im.nbytes for im in images)>memory_mb*1024**2:
        raise StitchError("输入已经超过内存预算")
    t0 = time.time()
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    largest=max(im.shape[0]*im.shape[1] for im in images)
    first=images[0].shape[0]*images[0].shape[1]
    stitcher.setRegistrationResol(0.6*first/largest)
    stitcher.setSeamEstimationResol(0.12*first/largest)
    status = stitcher.estimateTransform(images)
    if status != cv2.Stitcher_OK:
        raise StitchError(_STATUS_MSG.get(status, f"拼接失败（状态码 {status}）"))
    used = list(stitcher.component())
    cameras = stitcher.cameras()
    factor = stitcher.workScale()
    # OpenCV owns per-camera intrinsics; plan each ROI in original coordinates.
    focal = float(np.median([c.focal for c in cameras])) / factor
    sizes = {i:(images[i].shape[1],images[i].shape[0]) for i in used}
    rotations = {i:np.ascontiguousarray(c.R,np.float32) for i,c in zip(used,cameras)}
    intrinsics={i:np.array([[c.focal/factor,0,c.ppx/factor],[0,c.focal*c.aspect/factor,c.ppy/factor],[0,0,1]],np.float32) for i,c in zip(used,cameras)}
    try:
        scale, _, _, _ = plan_projection(sizes, rotations, focal, 'spherical', budget,sum(im.nbytes for im in images),intrinsics=intrinsics)
    except ValueError as e:
        raise StitchError(str(e)) from e
    # Use minimum input area conservatively: OpenCV computes a shared scale from an input image.
    # No input (including the first/selected one) may receive a scale larger than the plan.
    stitcher.setCompositingResol(-1 if scale==1 else min(scale,0.89)**2*min(im.shape[0]*im.shape[1] for im in images)/1e6)
    status, pano = stitcher.composePanorama()
    if status != cv2.Stitcher_OK or pano is None:
        raise StitchError(_STATUS_MSG.get(status, f"拼接失败（状态码 {status}）"))
    if max_output_side and max(pano.shape[:2]) > max_output_side:
        pano = cv2.resize(pano,None,fx=max_output_side/max(pano.shape[:2]),fy=max_output_side/max(pano.shape[:2]),interpolation=cv2.INTER_AREA)
    if status != cv2.Stitcher_OK:
        raise StitchError(_STATUS_MSG.get(status, f"拼接失败（状态码 {status}）"))
    rect = autocrop.largest_inner_rect(autocrop.valid_mask_from_image(pano))
    return BaselineResult(image=pano, seconds=time.time() - t0, autocrop=rect)
