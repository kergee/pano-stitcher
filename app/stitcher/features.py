"""统一特征与带评分匹配接口：SIFT 双向比值互检，DISK/LightGlue 可选后端。
明确指定 deep 时失败不会回退；auto 回退时结果报告实际后端。
C++ 对照：cv::SIFT、cv::BFMatcher；移动端深度模型仍需另行导出和验证。
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


@dataclass
class Matches:
    pts_i: np.ndarray
    pts_j: np.ndarray
    indices: np.ndarray
    scores: np.ndarray
    source: str

    def __iter__(self):  # compatibility with coordinate-pair callers
        return iter((self.pts_i, self.pts_j))


def match_record(fi, fj, indices, scores, source):
    idx = np.asarray(indices, dtype=np.int32).reshape(-1,2)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape!=(len(idx),):
        raise ValueError("匹配评分数量与索引数量不一致")
    chosen=[]; seen_i=set(); seen_j=set()
    for k in np.argsort(-scores, kind='stable'):
        a,b=idx[k]
        if min(a,b)<0 or a>=len(fi.kps) or b>=len(fj.kps):
            raise ValueError("匹配特征索引越界")
        if not np.isfinite(fi.kps[a]).all() or not np.isfinite(fj.kps[b]).all():
            continue
        # SIFT orientations can duplicate coordinates despite distinct feature indices.
        ki=tuple(np.round(fi.kps[a],1)); kj=tuple(np.round(fj.kps[b],1))
        if ki in seen_i or kj in seen_j or not np.isfinite(scores[k]):
            continue
        seen_i.add(ki);seen_j.add(kj);chosen.append(k)
    idx=idx[chosen]
    return Matches(fi.kps[idx[:,0]],fj.kps[idx[:,1]],idx,scores[chosen],source)


def empty_matches(source):
    return Matches(np.empty((0,2),np.float32),np.empty((0,2),np.float32),np.empty((0,2),np.int32),np.empty(0,np.float32),source)


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

    def match(self, fi: ImageFeats, fj: ImageFeats) -> Matches:
        if fi.payload is None or fj.payload is None or min(len(fi.kps),len(fj.kps)) < 4:
            return empty_matches(self.name)
        def accepted(a,b):
            return {row[0].queryIdx:(row[0].trainIdx,1-row[0].distance/max(row[1].distance,1e-9))
                    for row in self._matcher.knnMatch(a,b,k=2)
                    if len(row)==2 and row[0].distance < 0.8*row[1].distance}
        forward=accepted(fi.payload,fj.payload);reverse=accepted(fj.payload,fi.payload)
        good=[(a,b,min(score,reverse[b][1])) for a,(b,score) in forward.items() if b in reverse and reverse[b][0]==a]
        return match_record(fi,fj,[(a,b) for a,b,_ in good],[score for _,_,score in good],self.name)


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

    def match(self, fi: ImageFeats, fj: ImageFeats) -> Matches:
        empty = empty_matches(self.name)
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
        if 'scores' in out:
            scores=out['scores'][0].detach().cpu().numpy()
        elif 'matching_scores0' in out:
            scores=out['matching_scores0'][0].detach().cpu().numpy()[idx[:,0]]
        else:
            raise RuntimeError('LightGlue 接口未提供匹配置信度，拒绝无评分匹配')
        return match_record(fi,fj,idx,scores,self.name)


_deep_backend = None
_deep_error: str | None = None


def get_backend(name: str = "auto"):
    """按名称获取后端；auto 优先深度后端，不可用时回退 SIFT。深度后端全局缓存（模型只加载一次）。"""
    if name not in ("auto","deep","sift"):
        raise ValueError(f"未知特征后端：{name}")
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
