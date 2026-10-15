"""旋转相机拼接：特征匹配、MAGSAC、分量隔离、受检查 BA、预算内投影与融合。
核心没有 Web 依赖；局部形变为高斯残差位移场，并非完整 APAP Moving DLT。
C++ 对照：cv::findHomography、cv::detail::waveCorrect、RotationWarper、GraphCutSeamFinder、MultiBandBlender。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import autocrop, localwarp
from .budget import ComposeBudget, plan_projection, canvas_roi
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
    diagnostics: dict = field(default_factory=dict)


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
    scores: np.ndarray = None
    indices: np.ndarray = None
    coverage: float = 0.0


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


def _components(n, pairs):
    adj={i:set() for i in range(n)}
    for p in pairs:adj[p.i].add(p.j);adj[p.j].add(p.i)
    remaining=set(range(n));groups=[]
    while remaining:
        group={min(remaining)};queue=list(group)
        while queue:
            node=queue.pop()
            for other in adj[node]-group:group.add(other);queue.append(other)
        remaining-=group;groups.append(sorted(group))
    return sorted(groups,key=lambda group:(-len(group),-sum(p.confidence for p in pairs if p.i in group and p.j in group),group[0]))


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
    pairs: list[_Pair], rot: dict[int, np.ndarray], root: int, f: float, diagnostics=None, loss_scale=2.0,
) -> tuple[dict[int, np.ndarray], float]:
    """光束法平差：联合优化焦距 + 各相机旋转（根相机固定），最小化匹配点重投影误差。
    soft_l1 损失抗残余外点。对应 C++: cv::detail::BundleAdjusterRay。"""
    diag={"accepted":False,"unit":"work pixels","loss":"component soft_l1","f_scale":loss_scale}
    if diagnostics is not None:diagnostics.append(diag)
    try:
        from scipy.optimize import least_squares
    except ImportError:
        diag["reason"]="scipy unavailable"
        return rot, f  # 无 scipy 时跳过，串接结果直接可用

    nodes = sorted(rot.keys())
    free = [i for i in nodes if i != root]
    idx_of = {i: k for k, i in enumerate(free)}
    active = [p for p in pairs if p.i in rot and p.j in rot]
    if not active:
        diag["reason"]="no active pairs"
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

    def residuals(x, check=False):
        fx, rs = unpack(x)
        K = np.diag([fx, fx, 1.0])
        Kinv = np.diag([1.0 / fx, 1.0 / fx, 1.0])
        out = []
        for i, j, pi, pj in samples:
            Hij = K @ rs[j] @ rs[i].T @ Kinv
            proj = pi @ Hij.T
            z=proj[:,2:3]
            if check and (not np.isfinite(proj).all() or np.any(z<=1e-4)):
                raise ValueError('非有限投影或相机后方匹配点')
            proj = proj[:, :2] / np.maximum(z,1e-4)
            out.append((proj - pj).ravel())
        return np.concatenate(out)

    x0 = np.zeros(1 + 3 * len(free))
    x0[0] = np.log(f)
    for i in free:
        x0[1 + 3 * idx_of[i]: 4 + 3 * idx_of[i]] = cv2.Rodrigues(rot[i])[0].ravel()
    def cost(r):
        return float(np.sum(loss_scale**2*(np.sqrt(1+(r/loss_scale)**2)-1)))
    try:
        before=residuals(x0,check=True)
        diag.update(initial_rms=float(np.sqrt(np.mean(before**2))),initial_cost=cost(before))
        lower=np.full_like(x0,-np.inf);upper=np.full_like(x0,np.inf)
        lower[0]=np.log(f*0.2);upper[0]=np.log(f*5)
        sol = least_squares(residuals,x0,bounds=(lower,upper),loss='soft_l1',f_scale=loss_scale,max_nfev=200)
        if not np.isfinite(sol.x).all():raise ValueError('非有限 BA 参数')
        after=residuals(sol.x,check=True)
        f_new,rot_new=unpack(sol.x)
        diag.update(status=int(sol.status),success=bool(sol.success),nfev=int(sol.nfev),message=str(sol.message),
                    final_rms=float(np.sqrt(np.mean(after**2))),final_cost=cost(after),focal=float(f_new))
        valid=all(np.isfinite(R).all() and np.allclose(R.T@R,np.eye(3),atol=1e-5) and np.linalg.det(R)>0 for R in rot_new.values())
        if valid and 0.2*f<f_new<5*f and cost(after)<=cost(before)+1e-8 and diag['final_rms']<=diag['initial_rms']*1.01+1e-8:
            diag.update(accepted=True,reason='valid non-worsening solution (status alone is not acceptance)')
            return rot_new,float(f_new)
        diag['reason']='invalid cameras, focal bound or error increased'
    except Exception as e:
        diag['reason']=f'{type(e).__name__}: {e}'
    return rot,f



# ---------------------------------------------------------------- 主流程


PROJECTIONS = ("spherical", "cylindrical", "mercator", "plane")


def stitch_pro(
    images: list[np.ndarray],
    backend_name: str = "auto",
    projection: str = "spherical",
    progress=None,
    max_output_side: int = 0,
    memory_mb: float = 768,
    local_warp: bool = True,
    ba_mode: str = "single",
    pair_mode: str = "all",
    quality: str = "full",
) -> StitchResult:
    """自建旋转相机流水线拼接。images 为 BGR 原始分辨率图片列表。
    projection：spherical（默认）/ cylindrical / mercator / plane（直线不弯，视角大时边缘拉伸）。"""
    progress = progress or _noop
    try:
        ComposeBudget(max_output_side=max_output_side,memory_mb=memory_mb)
    except ValueError as e:
        raise StitchError(str(e)) from e
    if len(images) < 2:
        raise StitchError("至少需要 2 张图片")
    if projection not in PROJECTIONS:
        raise StitchError(f"未知投影方式：{projection}")
    if ba_mode not in ('single','staged') or pair_mode not in ('all','ordered') or quality not in ('full','fast'):
        raise StitchError('未知的优化、匹配、质量或边缘模式')
    if quality=='fast':
        local_warp=False
        if not max_output_side:max_output_side=1600
    t0 = time.time()
    n = len(images)

    # --- 1. 缩放到工作分辨率，提特征
    if any(im is None or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 8 for im in images):
        raise StitchError("输入必须为至少 8×8 的 uint8 BGR 图片")
    area = max(im.shape[0] * im.shape[1] for im in images)
    work_scale = min(1.0, np.sqrt(WORK_MEGAPIX * 1e6 / area))
    seam_scale = min(1.0, np.sqrt(SEAM_MEGAPIX * 1e6 / area))
    if sum(im.nbytes for im in images)*(1+work_scale**2) > memory_mb*1024**2:
        raise StitchError("输入与工作数组超过内存预算，请先缩小输入")
    work_imgs = [cv2.resize(im, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA) for im in images]

    try:
        backend = get_backend(backend_name)
    except (ValueError, RuntimeError) as e:
        raise StitchError(str(e)) from e
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
    all_candidates=[(i,j) for i in range(n) for j in range(i+1,n)]
    pair_list=[(i,j) for i,j in all_candidates if pair_mode=='all' or j-i<=2 or (i==0 and j==n-1)]
    candidates_initial=len(pair_list)
    attempted=[]
    # If ordered candidates do not connect the batch, append exhaustive rescue pairs.
    # Connected ordered batches remain a deliberate recall/latency tradeoff.

    for t,(i,j) in enumerate(pair_list):
        attempted.append((i,j))
        matches=backend.match(feats[i],feats[j])
        pts_i,pts_j=matches
        if len(pts_i)>=min_inliers:
            ci=_center(pts_i,feats[i].size);cj=_center(pts_j,feats[j].size)
            H,mask=cv2.findHomography(ci,cj,cv2.USAC_MAGSAC,3.,confidence=0.999)
            if H is not None and mask is not None and np.isfinite(H).all():
                inl=mask.ravel().astype(bool);num=int(inl.sum())
                conf=num/(8.+0.3*len(pts_i))
                coverage=min(cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))/(feats[k].size[0]*feats[k].size[1])
                             for pts,k in [(pts_i[inl],i),(pts_j[inl],j)]) if num>=3 else 0
                if num>=min_inliers and conf>=conf_thresh and coverage>=0.003:
                    pairs.append(_Pair(i,j,H,ci[inl],cj[inl],conf,all_i=ci,all_j=cj,scores=matches.scores,indices=matches.indices,coverage=coverage))
        if pair_mode=='ordered' and t==candidates_initial-1 and len(_components(n,pairs)[0])<n:
            pair_list.extend(p for p in all_candidates if p not in pair_list)
        progress('匹配图片对',0.25+0.25*(t+1)/len(pair_list))

    if not pairs:
        raise StitchError("没有找到可靠的重叠图片对（请确认照片间有 30% 以上重叠）")

    # --- 3. 焦距闭式估计 + 生成树串接旋转
    progress("估计相机几何", 0.55)
    groups=_components(n,pairs)
    selected=set(groups[0])
    pairs=[p for p in pairs if p.i in selected and p.j in selected]
    focal_est: list[float] = []
    for p in pairs:
        focal_est.extend(_focals_from_homography(p.H))
    focal_est = [v for v in focal_est if np.isfinite(v) and v>0]
    if focal_est:
        f = float(np.median(focal_est))
    else:  # 纯平移等退化场景：按手机典型视场角兜底
        f = 0.9 * np.median([feats[i].size[0] for i in selected])
    root, tree = _max_spanning_tree(n, pairs)
    rot = _chain_rotations(n, root, tree, f)
    used = sorted(rot.keys())
    dropped = [i for i in range(n) if i not in rot]
    if len(used) < 2:
        raise StitchError("只有一张图片能连入全景，无法拼接")

    # --- 4. 光束法平差（★）+ 波形校正
    progress("光束法平差", 0.6)
    ba_diagnostics=[]
    if ba_mode=='staged':
        tree_pairs=[]
        for edges in tree.values():
            for p in edges:
                if not any(q is p for q in tree_pairs):tree_pairs.append(p)
        rot,f=_bundle_adjust(tree_pairs,rot,root,f,ba_diagnostics)
        ba_diagnostics[-1]['stage']='tree initialization'
    rot, f = _bundle_adjust(pairs, rot, root, f, ba_diagnostics)
    ba_diagnostics[-1]['stage']='global'
    if ba_mode=='staged':
        rot,f=_bundle_adjust(pairs,rot,root,f,ba_diagnostics,loss_scale=1.0)
        ba_diagnostics[-1]['stage']='robust refinement'
    # 内部 rot 为世界→相机；OpenCV 的 RotationWarper/waveCorrect 期望相机→世界（ray = R·K⁻¹·p），
    # 在此边界处统一转置
    rmats = [np.ascontiguousarray(rot[i].T, dtype=np.float32) for i in used]
    try:  # 对应 C++: cv::detail::waveCorrect
        rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
    except cv2.error:
        pass
    rot = {i: rmats[k] for k, i in enumerate(used)}

    # --- 5. 合成（接缝在低分辨率算，最终合成与融合）
    progress("搜索最优接缝", 0.68)
    diagnostics = {"work_scale":work_scale, "seam_scale":seam_scale, "candidate_pairs":len(attempted), "possible_pairs":len(all_candidates), "pair_mode":pair_mode, "quality":quality, "components":groups, "selected_component":sorted(selected), "ba":ba_diagnostics, "pairs":[{"i":p.i,"j":p.j,"matches":len(p.all_i),"inliers":len(p.pts_i),"coverage":p.coverage} for p in pairs]}
    try:
        budget = ComposeBudget(max_output_side=max_output_side, memory_mb=memory_mb)
        try:
            pano, rect = _compose(images, used, rot, f, work_scale, seam_scale, pairs, projection, progress, budget, local_warp, diagnostics, quality)
        except localwarp.UnsafeFieldError as e:
            diagnostics['local_warp_fallback']=str(e)
            diagnostics['local_warp_rejected']=diagnostics.get('local_warp',{}).copy()
            pano, rect = _compose(images, used, rot, f, work_scale, seam_scale, pairs, projection, progress, budget, False, diagnostics, quality)
    except (ValueError, cv2.error) as e:
        raise StitchError(f"合成失败：{e}") from e
    progress("完成", 1.0)
    return StitchResult(
        image=pano, seconds=time.time() - t0, backend=backend.name,
        focal=f / work_scale, used=used, dropped=dropped, autocrop=rect, diagnostics=diagnostics,
    )


def _camera_K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float32)


def _compose(images, used, rot, f_work, work_scale, seam_scale, pairs, projection, progress, budget, local_warp, diagnostics, quality):
    cf = f_work / work_scale
    original_sizes = {i:(images[i].shape[1], images[i].shape[0]) for i in used}
    resident = sum(im.nbytes for im in images) + int(sum(im.nbytes for im in images)*work_scale**2)
    compose_scale, full_warper, rois, plan = plan_projection(original_sizes, rot, cf, projection, budget, resident, padding_pixels=26 if local_warp else 0)
    diagnostics.update(plan)
    diagnostics["effective_seam_scale"]=min(seam_scale,compose_scale)
    # Fields use compose coordinates, preserving the same field across both passes.
    seam_scale = min(seam_scale, compose_scale)
    sizes = {i:(max(1,round(w*compose_scale)),max(1,round(h*compose_scale))) for i,(w,h) in original_sizes.items()}
    cf *= compose_scale
    fields = localwarp.build_fields(used, pairs if local_warp else [], rot, full_warper,
        lambda i:_camera_K(cf,*sizes[i]), sizes, work_scale/compose_scale, rois, coordinate_scale=compose_scale, diagnostics=diagnostics)
    sw_aspect = seam_scale / work_scale

    sizes_of_compose = sizes
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
        if local_warp:
            pad=int(np.ceil(26*seam_scale))
            img_wp,mask_wp,corner=localwarp.pad_support(img_wp,mask_wp,corner,pad)
        img_wp, mask_wp = localwarp.warp_with_field(img_wp, mask_wp, corner, fields[i], seam_scale/compose_scale)
        corners.append(corner)
        sizes.append((img_wp.shape[1], img_wp.shape[0]))
        warped.append(img_wp)
        warped_masks.append(mask_wp)

    # ★分块增益曝光补偿（对应 cv::detail::BlocksGainCompensator）
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners, images=warped, masks=warped_masks)

    # ★GraphCut 最优接缝（对应 cv::detail::GraphCutSeamFinder）；
    # COST_COLOR_GRAD 让接缝优先走颜色和梯度都低的区域，更能隐藏视差错位
    seam_images=[im.copy() for im in warped]
    for k in range(len(used)):
        compensator.apply(k,corners[k],seam_images[k],warped_masks[k])
    def overlap_mae(imgs):
        values=[]
        for a in range(len(imgs)):
            for b in range(a+1,len(imgs)):
                ax,ay=corners[a];bx,by=corners[b]
                x=max(ax,bx);y=max(ay,by)
                x1=min(ax+imgs[a].shape[1],bx+imgs[b].shape[1]);y1=min(ay+imgs[a].shape[0],by+imgs[b].shape[0])
                if x1<=x or y1<=y:continue
                sa=(slice(y-ay,y1-ay,3),slice(x-ax,x1-ax,3));sb=(slice(y-by,y1-by,3),slice(x-bx,x1-bx,3))
                valid=(warped_masks[a][sa]>0)&(warped_masks[b][sb]>0)
                if valid.any():values.append(float(np.mean(np.abs(imgs[a][sa].astype(np.float32)-imgs[b][sb].astype(np.float32))[valid])))
        return float(np.mean(values)) if values else None
    diagnostics['exposure_overlap_mae']={'before':overlap_mae(warped),'after':overlap_mae(seam_images),'unit':'BGR intensity; includes parallax'}
    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    seam_masks = seam_finder.find([im.astype(np.float32) for im in seam_images], corners, [m.copy() for m in warped_masks])
    seam_masks = [m.get() if hasattr(m, "get") else m for m in seam_masks]

    # -- 合成阶段（全分辨率）
    progress("合成与融合", 0.78)
    full_corners = [(rois[i][0], rois[i][1]) for i in used]
    full_sizes = [(rois[i][2], rois[i][3]) for i in used]

    dst_roi = canvas_roi(list(rois.values()))
    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * 5 / 100
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(2 if quality=="fast" else max(1, min(8,int(np.log2(max(blend_width,2))) - 1)))
    blender.prepare(dst_roi)

    for k, i in enumerate(used):
        img = cv2.resize(images[i], sizes_of_compose[i], interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        K = _camera_K(cf, w, h)
        R = rot[i].astype(np.float32)
        corner, img_wp = full_warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_wp = full_warper.warp(np.full((h, w), 255, np.uint8), K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        if local_warp:
            pad=int(np.ceil(26*compose_scale))
            img_wp,mask_wp,corner=localwarp.pad_support(img_wp,mask_wp,corner,pad)
        img_wp, mask_wp = localwarp.warp_with_field(img_wp, mask_wp, corner, fields[i], 1.0)
        compensator.apply(k, corner, img_wp, mask_wp)
        # 接缝掩码放大回全分辨率，并与有效区域求交
        seam = cv2.dilate(seam_masks[k], np.ones((3, 3), np.uint8))
        seam = cv2.resize(seam, (mask_wp.shape[1], mask_wp.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
        mask = cv2.bitwise_and(seam, mask_wp)
        blender.feed(img_wp.astype(np.int16), mask, corner)
        progress("合成与融合", 0.78 + 0.2 * (k + 1) / len(used))

    result, result_mask = blender.blend(None, None)
    pano = np.clip(result, 0, 255).astype(np.uint8)
    mask = result_mask.reshape(result_mask.shape[:2])

    # 裁掉全黑边框，并求最大内接矩形供自动裁剪
    ys, xs = np.where(mask > 0)
    if not len(ys):
        raise StitchError("融合结果没有有效支持像素")
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pano, mask = pano[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    rect = autocrop.largest_inner_rect(mask)
    diagnostics["valid_coverage"]=float(np.count_nonzero(mask)/mask.size)
    diagnostics["crop_valid_coverage"]=1.0 if rect else None
    diagnostics["crop_area_fraction"]=rect[2]*rect[3]/mask.size if rect else 0
    return pano, rect
