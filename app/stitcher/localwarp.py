"""★局部网格弹性校正（APAP 思想）：消除旋转相机模型无法解释的视差错位。

原理：全局旋转模型对齐后，把每对图片的全部匹配点投影到球面全景坐标，
残余错位向量 r = proj_j(p) - proj_i(p) 就是该处视差的直接度量。
给每张图叠一层稀疏网格，顶点位移取周围残差的高斯加权平均
（重叠区内容"各让半步"在中点汇合，远离匹配点处自动衰减为零、退回全局模型），
再用位移场对球面投影后的图像做 remap 弹性形变。
接缝搜索与全分辨率融合使用同一位移场（按分辨率线性缩放），几何保持一致。

C++ 迁移对照：纯 numpy 数学 + cv::remap / cv::resize，逐行可译。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GRID = 33            # 网格顶点数（每边）
SIGMA_FRAC = 0.12    # 高斯核带宽 = 对角线 × 此系数
SHRINK = 0.3         # 位移收缩正则（防止孤立匹配点过度拉扯）
MAX_RESIDUAL = 80.0  # 残差超过此值（全分辨率像素）视为误匹配，丢弃


@dataclass
class OffsetField:
    corner: tuple[float, float]   # 全分辨率全景坐标下该图 bbox 左上角
    size: tuple[float, float]     # bbox 宽高
    grid: np.ndarray              # (GRID, GRID, 2) 顶点位移（全分辨率全景像素）


def _project_uv(pts_centered: np.ndarray, warper, K: np.ndarray, R_cw: np.ndarray,
                size: tuple[int, int]) -> np.ndarray:
    """中心化图像坐标 → 全景坐标 (u,v)。
    直接调用 warper.warpPoint，自动与当前投影方式（球面/柱面/平面…）保持一致。"""
    w, h = size
    out = np.empty((len(pts_centered), 2), np.float64)
    for k, (x, y) in enumerate(pts_centered):
        out[k] = warper.warpPoint((float(x + w / 2), float(y + h / 2)), K, R_cw)
    return out


def build_fields(
    used: list[int],
    pairs,                       # list[_Pair]，all_i/all_j 为中心化工作分辨率坐标
    rot: dict[int, np.ndarray],  # 相机→世界旋转（waveCorrect 之后）
    warper,                      # 全分辨率 PyRotationWarper（任意投影）
    K_of,                        # i -> 全分辨率相机内参 K
    sizes: dict[int, tuple],     # i -> (w, h) 全分辨率图像尺寸
    work_scale: float,
    rois: dict[int, tuple],      # i -> (x, y, w, h) 全分辨率 warpRoi
) -> dict[int, OffsetField]:
    """为每张图构建位移网格。无匹配的图返回零场（跳过形变）。"""
    # 收集每张图的（全景坐标位置, 期望位移）样本
    corr: dict[int, list[np.ndarray]] = {i: [] for i in used}
    for p in pairs:
        if p.i not in rot or p.j not in rot or p.all_i is None or len(p.all_i) == 0:
            continue
        Ri = rot[p.i].astype(np.float32)
        Rj = rot[p.j].astype(np.float32)
        uv_i = _project_uv(p.all_i / work_scale, warper, K_of(p.i), Ri, sizes[p.i])
        uv_j = _project_uv(p.all_j / work_scale, warper, K_of(p.j), Rj, sizes[p.j])
        r = uv_j - uv_i
        keep = np.linalg.norm(r, axis=1) < MAX_RESIDUAL
        if not keep.any():
            continue
        # i 图内容移 +r/2、j 图移 -r/2，在中点汇合
        corr[p.i].append(np.hstack([uv_i[keep], r[keep] / 2]))
        corr[p.j].append(np.hstack([uv_j[keep], -r[keep] / 2]))

    fields: dict[int, OffsetField] = {}
    for i in used:
        x0, y0, w, h = rois[i]
        field = OffsetField((float(x0), float(y0)), (float(w), float(h)),
                            np.zeros((GRID, GRID, 2), np.float32))
        samples = np.vstack(corr[i]) if corr[i] else None
        if samples is not None and len(samples):
            gx = np.linspace(x0, x0 + w, GRID)
            gy = np.linspace(y0, y0 + h, GRID)
            vx, vy = np.meshgrid(gx, gy)
            verts = np.stack([vx.ravel(), vy.ravel()], axis=1)          # (G²,2)
            d2 = ((verts[:, None, :] - samples[None, :, :2]) ** 2).sum(-1)  # (G², M)
            sigma = SIGMA_FRAC * np.hypot(w, h)
            wgt = np.exp(-d2 / (2 * sigma * sigma))
            num = wgt @ samples[:, 2:]                                   # (G²,2)
            den = wgt.sum(axis=1, keepdims=True) + SHRINK
            field.grid = (num / den).reshape(GRID, GRID, 2).astype(np.float32)
        fields[i] = field
    return fields


def warp_with_field(
    img_wp: np.ndarray,
    mask_wp: np.ndarray,
    corner: tuple[int, int],
    field: OffsetField,
    scale: float,                # 当前阶段全景坐标 / 全分辨率全景坐标 的比值
) -> tuple[np.ndarray, np.ndarray]:
    """对球面投影后的图像应用位移场：输出像素 x 取自 x - D(x)，内容整体移动 +D。"""
    if not field.grid.any():
        return img_wp, mask_wp
    h, w = img_wp.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # 当前像素的全分辨率全景坐标 → 网格采样位置
    qx = (xs + corner[0]) / scale
    qy = (ys + corner[1]) / scale
    ux = ((qx - field.corner[0]) / max(field.size[0], 1e-6) * (GRID - 1)).astype(np.float32)
    uy = ((qy - field.corner[1]) / max(field.size[1], 1e-6) * (GRID - 1)).astype(np.float32)
    D = cv2.remap(field.grid, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_x = (xs - D[..., 0] * scale).astype(np.float32)
    map_y = (ys - D[..., 1] * scale).astype(np.float32)
    out = cv2.remap(img_wp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out_mask = cv2.remap(mask_wp, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return out, out_mask
