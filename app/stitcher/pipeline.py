"""引擎 2：超越版拼接流水线（旋转相机模型）。

流程（相对 AutoStitch 的超越点用 ★ 标注）：
 1. 缩图提特征 → ★深度特征匹配（DISK+LightGlue，弱纹理场景远超 SIFT）
 2. ★MAGSAC++ 估计成对单应性（比经典 RANSAC 更准更鲁棒）
 3. 由单应性闭式估计焦距（Brown 论文公式）→ 最大生成树串接相机旋转
 4. ★Scipy 光束法平差（旋转+焦距联合优化，soft_l1 抗外点）→ 波形校正拉平地平线
 5. 球面投影 → ★分块增益曝光补偿（AutoStitch 仅全局增益）
    → ★GraphCut 最优接缝（AutoStitch 无此步，接缝自动避开显眼区域）
    → 多频段融合，全分辨率合成

C++ 迁移对照：步骤 2-5 全部有同名 OpenCV C++ API（cv::findHomography(USAC_MAGSAC)、
cv::detail::waveCorrect、cv::detail::SphericalWarper、cv::detail::BlocksGainCompensator、
cv::detail::GraphCutSeamFinder、cv::detail::MultiBandBlender）；
光束法平差对应 cv::detail::BundleAdjusterRay。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import autocrop, localwarp
from .features import ImageFeats, get_backend

WORK_MEGAPIX = 0.6   # 特征/几何估计分辨率
SEAM_MEGAPIX = 0.12  # 接缝搜索分辨率
# 图片对有效阈值（Brown 置信度公式）。该公式为 SIFT+最近邻设计；
# LightGlue 输出已经过匹配器内部置信度过滤，误匹配率低得多，阈值相应放宽
CONF_THRESH = {"sift": 0.9, "deep": 0.45}
MIN_INLIERS = {"sift": 20, "deep": 12}


class StitchError(RuntimeError):
    pass


@dataclass
class StitchResult:
    image: np.ndarray
    seconds: float
    backend: str
    focal: float
    used: list[int]              # 参与拼接的原始图片下标
    dropped: list[int] = field(default_factory=list)  # 无法连接被丢弃的图片
    autocrop: tuple[int, int, int, int] | None = None  # 建议裁剪框 (x,y,w,h)


@dataclass
class _Pair:
    i: int
    j: int
    H: np.ndarray                # 中心化坐标下 i→j 的单应性
    pts_i: np.ndarray            # 内点（中心化坐标），几何估计/BA 用
    pts_j: np.ndarray
    confidence: float
    all_i: np.ndarray = None     # 全部匹配点（含视差点，MAGSAC 内点之外），局部网格校正用
    all_j: np.ndarray = None


def _noop(stage: str, frac: float) -> None:
    pass


# ---------------------------------------------------------------- 几何工具


def _center(pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """像素坐标 → 以主点（图像中心）为原点的坐标，便于焦距估计与 K=diag(f,f,1)。"""
    return pts - np.float32([size[0] / 2.0, size[1] / 2.0])


def _focals_from_homography(H: np.ndarray) -> list[float]:
    """Brown《Automatic Panoramic Image Stitching》附录的闭式焦距估计。
    对应 C++: cv::detail::focalsFromHomography。要求 H 基于中心化坐标。"""
    h = H.flatten()
    est: list[float] = []

    d1, d2 = h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if d1 != 0 else -1
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2 if d2 != 0 else -1
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        est.append(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
    elif v1 > 0:
        est.append(np.sqrt(v1))

    d1, d2 = h[0] * h[3] + h[1] * h[4], h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
    v1 = -h[2] * h[5] / d1 if d1 != 0 else -1
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2 if d2 != 0 else -1
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        est.append(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
    elif v1 > 0:
        est.append(np.sqrt(v1))
    return est


def _project_so3(M: np.ndarray) -> np.ndarray:
    """把近似旋转矩阵投影到 SO(3)（SVD 正交化）。"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def _max_spanning_tree(n: int, pairs: list[_Pair]) -> tuple[int, dict[int, list[_Pair]]]:
    """按置信度取最大生成树（Prim），返回（根节点=连接最强的图, 邻接表）。
    对应 AutoStitch 的全景识别：不在树上的图片自动被丢弃。"""
    adj: dict[int, list[_Pair]] = {i: [] for i in range(n)}
    for p in pairs:
        adj[p.i].append(p)
        adj[p.j].append(p)
    root = max(range(n), key=lambda i: sum(p.confidence for p in adj[i]))
    in_tree = {root}
    tree: dict[int, list[_Pair]] = {i: [] for i in range(n)}
    candidates = list(adj[root])
    while candidates:
        candidates.sort(key=lambda p: p.confidence)
        best = None
        while candidates:
            p = candidates.pop()
            new = ({p.i, p.j} - in_tree)
            if len(new) == 1:
                best = p
                break
        if best is None:
            break
        node = (({best.i, best.j}) - in_tree).pop()
        in_tree.add(node)
        tree[best.i].append(best)
        tree[best.j].append(best)
        candidates.extend(q for q in adj[node] if len({q.i, q.j} - in_tree) == 1)
    return root, tree


def _chain_rotations(n: int, root: int, tree: dict[int, list[_Pair]], f: float) -> dict[int, np.ndarray]:
    """从根节点沿生成树 BFS 串接旋转：H_ij = K R_j R_i^T K^-1（中心化坐标）。"""
    K = np.diag([f, f, 1.0])
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    rot: dict[int, np.ndarray] = {root: np.eye(3)}
    queue = [root]
    while queue:
        i = queue.pop(0)
        for p in tree[i]:
            j = p.j if p.i == i else p.i
            if j in rot:
                continue
            H_ij = p.H if p.i == i else np.linalg.inv(p.H)  # i→j
            rot[j] = _project_so3(Kinv @ H_ij @ K) @ rot[i]
            queue.append(j)
    return rot


def _bundle_adjust(
    pairs: list[_Pair], rot: dict[int, np.ndarray], root: int, f: float,
) -> tuple[dict[int, np.ndarray], float]:
    """光束法平差：联合优化焦距 + 各相机旋转（根相机固定），最小化匹配点重投影误差。
    soft_l1 损失抗残余外点。对应 C++: cv::detail::BundleAdjusterRay。"""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return rot, f  # 无 scipy 时跳过，串接结果直接可用

    nodes = sorted(rot.keys())
    free = [i for i in nodes if i != root]
    idx_of = {i: k for k, i in enumerate(free)}
    active = [p for p in pairs if p.i in rot and p.j in rot]
    if not active:
        return rot, f

    # 每对最多取 250 个内点参与优化，控制 CPU 耗时
    samples = []
    for p in active:
        m = len(p.pts_i)
        sel = np.random.default_rng(0).choice(m, min(m, 250), replace=False)
        pi = np.hstack([p.pts_i[sel], np.ones((len(sel), 1))])  # 齐次
        pj = p.pts_j[sel]
        samples.append((p.i, p.j, pi.astype(np.float64), pj.astype(np.float64)))

    def unpack(x):
        fx = np.exp(x[0])
        rs = {root: rot[root]}
        for i in free:
            rs[i] = cv2.Rodrigues(x[1 + 3 * idx_of[i]: 4 + 3 * idx_of[i]])[0]
        return fx, rs

    def residuals(x):
        fx, rs = unpack(x)
        K = np.diag([fx, fx, 1.0])
        Kinv = np.diag([1.0 / fx, 1.0 / fx, 1.0])
        out = []
        for i, j, pi, pj in samples:
            Hij = K @ rs[j] @ rs[i].T @ Kinv
            proj = pi @ Hij.T
            proj = proj[:, :2] / proj[:, 2:3]
            out.append((proj - pj).ravel())
        return np.concatenate(out)

    x0 = np.zeros(1 + 3 * len(free))
    x0[0] = np.log(f)
    for i in free:
        x0[1 + 3 * idx_of[i]: 4 + 3 * idx_of[i]] = cv2.Rodrigues(rot[i])[0].ravel()
    try:
        sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, max_nfev=200)
        f_new, rot_new = unpack(sol.x)
        if 0.2 * f < f_new < 5 * f:  # 发散保护
            return rot_new, float(f_new)
    except Exception:
        pass
    return rot, f


# ---------------------------------------------------------------- 主流程


PROJECTIONS = ("spherical", "cylindrical", "mercator", "plane")


def stitch_pro(
    images: list[np.ndarray],
    backend_name: str = "auto",
    projection: str = "spherical",
    progress=None,
) -> StitchResult:
    """超越版拼接。images 为 BGR 原始分辨率图片列表。
    projection：spherical（默认）/ cylindrical / mercator / plane（直线不弯，视角大时边缘拉伸）。"""
    progress = progress or _noop
    if len(images) < 2:
        raise StitchError("至少需要 2 张图片")
    if projection not in PROJECTIONS:
        raise StitchError(f"未知投影方式：{projection}")
    t0 = time.time()
    n = len(images)

    # --- 1. 缩放到工作分辨率，提特征
    area = images[0].shape[0] * images[0].shape[1]
    work_scale = min(1.0, np.sqrt(WORK_MEGAPIX * 1e6 / area))
    seam_scale = min(1.0, np.sqrt(SEAM_MEGAPIX * 1e6 / area))
    work_imgs = [cv2.resize(im, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA) for im in images]

    backend = get_backend(backend_name)
    progress(f"提取特征（{backend.name}）", 0.05)
    feats: list[ImageFeats] = []
    for k, im in enumerate(work_imgs):
        feats.append(backend.extract(im))
        progress(f"提取特征（{backend.name}）{k + 1}/{n}", 0.05 + 0.2 * (k + 1) / n)

    # --- 2. 成对匹配 + MAGSAC 单应性（★）
    progress("匹配图片对", 0.25)
    kind = "sift" if backend.name == "sift" else "deep"
    conf_thresh, min_inliers = CONF_THRESH[kind], MIN_INLIERS[kind]
    pairs: list[_Pair] = []
    pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for t, (i, j) in enumerate(pair_list):
        pts_i, pts_j = backend.match(feats[i], feats[j])
        if len(pts_i) < min_inliers:
            continue
        ci = _center(pts_i, feats[i].size)
        cj = _center(pts_j, feats[j].size)
        H, mask = cv2.findHomography(ci, cj, cv2.USAC_MAGSAC, 3.0, confidence=0.999)
        if H is None:
            continue
        inl = mask.ravel().astype(bool)
        num = int(inl.sum())
        conf = num / (8.0 + 0.3 * len(pts_i))  # Brown 置信度公式
        if num >= min_inliers and conf >= conf_thresh:
            pairs.append(_Pair(i, j, H, ci[inl], cj[inl], conf, all_i=ci, all_j=cj))
        progress("匹配图片对", 0.25 + 0.25 * (t + 1) / len(pair_list))

    if not pairs:
        raise StitchError("没有找到可靠的重叠图片对（请确认照片间有 30% 以上重叠）")

    # --- 3. 焦距闭式估计 + 生成树串接旋转
    progress("估计相机几何", 0.55)
    focal_est: list[float] = []
    for p in pairs:
        focal_est.extend(_focals_from_homography(p.H))
    if focal_est:
        f = float(np.median(focal_est))
    else:  # 纯平移等退化场景：按手机典型视场角兜底
        f = 0.9 * feats[0].size[0]
    root, tree = _max_spanning_tree(n, pairs)
    rot = _chain_rotations(n, root, tree, f)
    used = sorted(rot.keys())
    dropped = [i for i in range(n) if i not in rot]
    if len(used) < 2:
        raise StitchError("只有一张图片能连入全景，无法拼接")

    # --- 4. 光束法平差（★）+ 波形校正
    progress("光束法平差", 0.6)
    rot, f = _bundle_adjust([p for p in pairs if p.i in rot and p.j in rot], rot, root, f)
    # 内部 rot 为世界→相机；OpenCV 的 RotationWarper/waveCorrect 期望相机→世界（ray = R·K⁻¹·p），
    # 在此边界处统一转置
    rmats = [np.ascontiguousarray(rot[i].T, dtype=np.float32) for i in used]
    try:  # 对应 C++: cv::detail::waveCorrect
        rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
    except cv2.error:
        pass
    rot = {i: rmats[k] for k, i in enumerate(used)}

    # --- 5. 合成（接缝在低分辨率算，最终全分辨率融合）
    progress("搜索最优接缝", 0.68)
    pano, rect = _compose(images, used, rot, f, work_scale, seam_scale, pairs, projection, progress)
    progress("完成", 1.0)
    return StitchResult(
        image=pano, seconds=time.time() - t0, backend=backend.name,
        focal=f / work_scale, used=used, dropped=dropped, autocrop=rect,
    )


def _camera_K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float32)


def _compose(images, used, rot, f_work, work_scale, seam_scale, pairs, projection, progress):
    """局部弹性校正 + 曝光补偿 + GraphCut 接缝 + 多频段融合（cv2.detail，与 C++ 同 API）。"""
    sw_aspect = seam_scale / work_scale
    cf = f_work / work_scale  # 全分辨率焦距

    # ★局部网格弹性校正（APAP 思想）：先算全分辨率 ROI，再由匹配点残差构建位移场
    full_warper = cv2.PyRotationWarper(projection, cf)
    rois, sizes = {}, {}
    for i in used:
        h, w = images[i].shape[:2]
        sizes[i] = (w, h)
        rois[i] = full_warper.warpRoi((w, h), _camera_K(cf, w, h), rot[i].astype(np.float32))
    # 平面投影在大视角下会发散，先做输出面积保险
    total = sum(rw * rh for _, _, rw, rh in rois.values())
    if total > 6e8:
        raise StitchError("该投影方式下输出尺寸过大（视场角太大），请改用球面/柱面/墨卡托投影")
    fields = localwarp.build_fields(
        used, pairs, rot, full_warper,
        lambda i: _camera_K(cf, sizes[i][0], sizes[i][1]), sizes, work_scale, rois)

    # -- 接缝阶段（低分辨率）
    warper = cv2.PyRotationWarper(projection, f_work * sw_aspect)
    corners, sizes, warped, warped_masks = [], [], [], []
    for i in used:
        img = cv2.resize(images[i], None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        K = _camera_K(f_work * sw_aspect, w, h)
        R = rot[i].astype(np.float32)
        corner, img_wp = warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_wp = warper.warp(np.full((h, w), 255, np.uint8), K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        img_wp, mask_wp = localwarp.warp_with_field(img_wp, mask_wp, corner, fields[i], seam_scale)
        corners.append(corner)
        sizes.append((img_wp.shape[1], img_wp.shape[0]))
        warped.append(img_wp)
        warped_masks.append(mask_wp)

    # ★分块增益曝光补偿（对应 cv::detail::BlocksGainCompensator）
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners, images=warped, masks=warped_masks)

    # ★GraphCut 最优接缝（对应 cv::detail::GraphCutSeamFinder）；
    # COST_COLOR_GRAD 让接缝优先走颜色和梯度都低的区域，更能隐藏视差错位
    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    seam_masks = seam_finder.find([im.astype(np.float32) for im in warped], corners, warped_masks)
    seam_masks = [m.get() if hasattr(m, "get") else m for m in seam_masks]

    # -- 合成阶段（全分辨率）
    progress("全分辨率融合", 0.78)
    full_corners = [(rois[i][0], rois[i][1]) for i in used]
    full_sizes = [(rois[i][2], rois[i][3]) for i in used]

    dst_roi = cv2.detail.resultRoi(corners=full_corners, sizes=full_sizes)
    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * 5 / 100
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(max(1, int(np.log2(blend_width)) - 1))
    blender.prepare(dst_roi)

    for k, i in enumerate(used):
        img = images[i]
        h, w = img.shape[:2]
        K = _camera_K(cf, w, h)
        R = rot[i].astype(np.float32)
        corner, img_wp = full_warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_wp = full_warper.warp(np.full((h, w), 255, np.uint8), K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        img_wp, mask_wp = localwarp.warp_with_field(img_wp, mask_wp, corner, fields[i], 1.0)
        compensator.apply(k, corner, img_wp, mask_wp)
        # 接缝掩码放大回全分辨率，并与有效区域求交
        seam = cv2.dilate(seam_masks[k], np.ones((3, 3), np.uint8))
        seam = cv2.resize(seam, (mask_wp.shape[1], mask_wp.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
        mask = cv2.bitwise_and(seam, mask_wp)
        blender.feed(img_wp.astype(np.int16), mask, corner)
        progress("全分辨率融合", 0.78 + 0.2 * (k + 1) / len(used))

    result, result_mask = blender.blend(None, None)
    pano = np.clip(result, 0, 255).astype(np.uint8)
    mask = result_mask.reshape(result_mask.shape[:2])

    # 裁掉全黑边框，并求最大内接矩形供自动裁剪
    ys, xs = np.where(mask > 0)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pano, mask = pano[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    rect = autocrop.largest_inner_rect(mask)
    return pano, rect
