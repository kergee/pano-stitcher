"""自动裁剪：求有效区域掩码内的最大内接轴对齐矩形（去除拼接黑边）。

算法：逐行累积高度直方图 + 单调栈求最大矩形，O(H×W)。
在缩小的掩码上计算再放大坐标，速度与精度平衡。
C++ 迁移对照：纯数组逻辑，逐行可译。
"""
from __future__ import annotations

import cv2
import numpy as np

_MAX_SIDE = 640  # 掩码缩小到此边长再计算


def largest_inner_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """mask: uint8（>0 为有效像素）。返回 (x, y, w, h)，原分辨率坐标。"""
    H, W = mask.shape[:2]
    s = min(1.0, _MAX_SIDE / max(H, W))
    m = cv2.resize(mask, (max(1, int(W * s)), max(1, int(H * s))),
                   interpolation=cv2.INTER_NEAREST) if s < 1.0 else mask
    m = (m > 0).astype(np.int32)
    h, w = m.shape

    heights = np.zeros(w, np.int32)
    best = (0, 0, w, h)
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

    x0, y0, rw, rh = best
    # 还原到原分辨率并收缩 1px 防边缘半透明像素
    inv = 1.0 / s
    x0 = min(W - 1, int(np.ceil(x0 * inv)) + 1)
    y0 = min(H - 1, int(np.ceil(y0 * inv)) + 1)
    x1 = max(x0 + 1, int((best[0] + rw) * inv) - 1)
    y1 = max(y0 + 1, int((best[1] + rh) * inv) - 1)
    return (x0, y0, min(x1, W) - x0, min(y1, H) - y0)


def valid_mask_from_image(pano: np.ndarray) -> np.ndarray:
    """无掩码时（baseline 引擎）从成图估计有效区域：非纯黑像素 + 形态学闭运算补内部黑洞。"""
    mask = (pano.max(axis=2) > 0).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
