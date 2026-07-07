"""特征提取与匹配后端。

两个后端，统一接口（extract → ImageFeats，match → 对应点坐标对）：
- SiftBackend：经典 SIFT + 比值检验（AutoStitch 同款，作为回退）
- DeepBackend：DISK 深度特征 + LightGlue 匹配器（kornia 实现），
  在弱纹理/重复纹理/大视角变化场景下成功率远超 SIFT —— 引擎 2 的核心超越点。

C++ 迁移对照：SIFT → cv::SIFT + cv::BFMatcher；
深度后端 → 模型导出 ONNX 后用 onnxruntime (iOS/Android/鸿蒙均支持) 推理。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ImageFeats:
    kps: np.ndarray  # (N,2) float32，工作分辨率下的像素坐标
    payload: Any     # 后端自有的描述子载体
    size: tuple[int, int]  # (w, h) 工作分辨率


class SiftBackend:
    name = "sift"

    def __init__(self, nfeatures: int = 3000):
        self._sift = cv2.SIFT_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, img_bgr: np.ndarray) -> ImageFeats:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self._sift.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        return ImageFeats(pts, desc, (img_bgr.shape[1], img_bgr.shape[0]))

    def match(self, fi: ImageFeats, fj: ImageFeats) -> tuple[np.ndarray, np.ndarray]:
        empty = (np.zeros((0, 2), np.float32),) * 2
        if fi.payload is None or fj.payload is None or len(fi.kps) < 4 or len(fj.kps) < 4:
            return empty
        knn = self._matcher.knnMatch(fi.payload, fj.payload, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if not good:
            return empty
        pts_i = fi.kps[[m.queryIdx for m in good]]
        pts_j = fj.kps[[m.trainIdx for m in good]]
        return pts_i, pts_j


class DeepBackend:
    name = "disk+lightglue"

    def __init__(self, max_kps: int = 2048):
        import torch
        import kornia.feature as KF

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._disk = KF.DISK.from_pretrained("depth").to(self._device).eval()
        self._lg = KF.LightGlue("disk").to(self._device).eval()
        self._max_kps = max_kps

    def extract(self, img_bgr: np.ndarray) -> ImageFeats:
        torch = self._torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(self._device) / 255.0
        with torch.inference_mode():
            feats = self._disk(t, n=self._max_kps, pad_if_not_divisible=True)[0]
        kps = feats.keypoints.cpu().numpy().astype(np.float32)
        return ImageFeats(
            kps,
            (feats.keypoints.detach(), feats.descriptors.detach()),
            (img_bgr.shape[1], img_bgr.shape[0]),
        )

    def _lg_input(self, f: ImageFeats) -> dict:
        torch = self._torch
        kp, desc = f.payload
        return {
            "keypoints": kp[None].to(self._device),
            "descriptors": desc[None].to(self._device),
            "image_size": torch.tensor([list(f.size)], dtype=torch.float32, device=self._device),
        }

    def match(self, fi: ImageFeats, fj: ImageFeats) -> tuple[np.ndarray, np.ndarray]:
        empty = (np.zeros((0, 2), np.float32),) * 2
        if len(fi.kps) < 4 or len(fj.kps) < 4:
            return empty
        torch = self._torch
        with torch.inference_mode():
            out = self._lg({"image0": self._lg_input(fi), "image1": self._lg_input(fj)})
        if "matches" in out:  # kornia/lightglue 新版：每张图返回 (M,2) 索引对
            idx = out["matches"][0].cpu().numpy()
        else:  # 旧版接口：matches0 为 (N,) 指向图1的索引，-1 表示无匹配
            m0 = out["matches0"][0].cpu().numpy()
            valid = np.where(m0 >= 0)[0]
            idx = np.stack([valid, m0[valid]], axis=1)
        if len(idx) == 0:
            return empty
        return fi.kps[idx[:, 0]], fj.kps[idx[:, 1]]


_deep_backend = None
_deep_error: str | None = None


def get_backend(name: str = "auto"):
    """按名称获取后端；auto 优先深度后端，不可用时回退 SIFT。深度后端全局缓存（模型只加载一次）。"""
    global _deep_backend, _deep_error
    if name in ("auto", "deep"):
        if _deep_backend is None and _deep_error is None:
            try:
                _deep_backend = DeepBackend()
            except Exception as e:  # torch/kornia 缺失或权重下载失败
                _deep_error = str(e)
        if _deep_backend is not None:
            return _deep_backend
        if name == "deep":
            raise RuntimeError(f"深度后端不可用：{_deep_error}")
    return SiftBackend()
