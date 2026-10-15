"""自动裁剪：求有效区域掩码内的最大内接轴对齐矩形（去除拼接黑边）。

算法：逐行累积高度直方图 + 单调栈求最大矩形，O(H×W)。
在缩小的掩码上计算再放大坐标，速度与精度平衡。
C++ 迁移对照：纯数组逻辑，逐行可译。
"""
from __future__ import annotations

import cv2
import numpy as np

_MAX_SIDE = 640  # 掩码缩小到此边长再计算


def largest_inner_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """mask: uint8（>0 为有效像素）。返回 (x, y, w, h)，原分辨率坐标。"""
    if mask.ndim!=2:
        raise ValueError('裁剪掩码必须为二维')
    H,W=mask.shape
    if not H or not W or not np.any(mask>0):
        return None
    # Exact conservative block pooling: a single invalid source pixel invalidates its cell.
    block=max(1,int(np.ceil(max(H,W)/_MAX_SIDE)))
    mh,mw=(H+block-1)//block,(W+block-1)//block
    valid=np.pad(mask>0,((0,mh*block-H),(0,mw*block-W)),constant_values=False)
    m=valid.reshape(mh,block,mw,block).all(axis=(1,3)).astype(np.int32)
    h, w = m.shape

    heights = np.zeros(w, np.int32)
    best = None
    best_area = 0
    for y in range(h):
        heights = (heights + 1) * m[y]
        stack: list[tuple[int, int]] = []  # (起始列, 高度)
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, y - sh + 1, x - sx, sh)
                start = sx
            if not stack or cur > stack[-1][1]:
                stack.append((start, cur))

    if best is None:
        return None
    x,y,rw,rh=(v*block for v in best)
    rect=(x,y,min(rw,W-x),min(rh,H-y))
    if not np.all(mask[y:y+rect[3],x:x+rect[2]]>0):
        raise ValueError('内部错误：裁剪矩形包含无效像素')
    return rect



def valid_mask_from_image(pano: np.ndarray) -> np.ndarray:
    """无掩码时（baseline 引擎）从成图估计有效区域：非纯黑像素 + 形态学闭运算补内部黑洞。"""
    mask = (pano.max(axis=2) > 0).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
