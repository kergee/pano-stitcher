"""局部高斯残差位移场：近似减小旋转模型无法解释的视差；不是完整 APAP。

原理：全局旋转模型对齐后，把每对图片的通过可靠性与邻域一致性筛选的匹配点投影到球面全景坐标，
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


class UnsafeFieldError(RuntimeError):
    pass


@dataclass
class OffsetField:
    corner: tuple[float, float]   # 合成分辨率全景坐标下该图 bbox 左上角
    size: tuple[float, float]     # 合成分辨率 bbox 宽高
    grid: np.ndarray  # (GRID,GRID,2)，合成分辨率全景位移
    stats: dict | None = None
    image_index: int = -1


def _project_uv(pts_centered: np.ndarray, warper, K: np.ndarray, R_cw: np.ndarray,
                size: tuple[int, int]) -> np.ndarray:
    """中心化图像坐标 → 全景坐标 (u,v)。
    直接调用 warper.warpPoint，自动与当前投影方式（球面/柱面/平面…）保持一致。"""
    w, h = size
    out = np.empty((len(pts_centered), 2), np.float64)
    for k, (x, y) in enumerate(pts_centered):
        out[k] = warper.warpPoint((float(x + w / 2), float(y + h / 2)), K, R_cw)
    return out


def coherent_samples(uv, residual, limit, radius):
    """Keep coherent parallax outside global-H inliers; reject isolated bad correspondences."""
    from scipy.spatial import cKDTree
    keep=np.isfinite(uv).all(1)&np.isfinite(residual).all(1)&(np.linalg.norm(residual,axis=1)<limit)
    candidates=np.flatnonzero(keep)
    result=np.zeros(len(uv),bool)
    if len(candidates)<4:
        return result
    dist,idx=cKDTree(uv[candidates]).query(uv[candidates],k=min(9,len(candidates)))
    for k,original in enumerate(candidates):
        neighbors=idx[k,1:][dist[k,1:]<radius]
        if len(neighbors)<3:
            continue
        vectors=residual[candidates[neighbors]]
        med=np.median(vectors,axis=0)
        mad=np.median(np.linalg.norm(vectors-med,axis=1))
        result[original]=np.linalg.norm(residual[original]-med)<=max(limit*0.04,3*mad)
    return result


def build_fields(
    used: list[int],
    pairs,                       # list[_Pair]，all_i/all_j 为中心化工作分辨率坐标
    rot: dict[int, np.ndarray],  # 相机→世界旋转（waveCorrect 之后）
    warper,                      # 合成分辨率 PyRotationWarper（任意投影）
    K_of,                        # i -> 合成分辨率相机内参 K
    sizes: dict[int, tuple],     # i -> (w, h) 合成分辨率图像尺寸
    work_scale: float,
    rois: dict[int, tuple],      # i -> (x, y, w, h) 合成分辨率 warpRoi
    coordinate_scale: float = 1.0,
    diagnostics: dict | None = None,
) -> dict[int, OffsetField]:
    """为每张图构建位移网格。无匹配的图返回零场（跳过形变）。"""
    # 收集每张图的（全景坐标位置, 期望位移）样本
    corr: dict[int, list[np.ndarray]] = {i: [] for i in used}
    stats = {"candidates":0,"retained":0,"limited_fields":0,"coverage_checks":[]}
    paired_samples=[]
    limit=MAX_RESIDUAL*coordinate_scale
    for p in pairs:
        if p.i not in rot or p.j not in rot or p.all_i is None or len(p.all_i) == 0:
            continue
        Ri = rot[p.i].astype(np.float32)
        Rj = rot[p.j].astype(np.float32)
        uv_i = _project_uv(p.all_i / work_scale, warper, K_of(p.i), Ri, sizes[p.i])
        uv_j = _project_uv(p.all_j / work_scale, warper, K_of(p.j), Rj, sizes[p.j])
        r = uv_j - uv_i
        radius=max(8*coordinate_scale,0.15*min(np.hypot(*sizes[p.i]),np.hypot(*sizes[p.j])))
        keep=coherent_samples(uv_i,r,limit,radius)&coherent_samples(uv_j,r,limit,radius)
        stats['candidates']+=len(r);stats['retained']+=int(keep.sum())
        if not keep.any():
            continue
        # i 图内容移 +r/2、j 图移 -r/2，在中点汇合
        paired_samples.append((p.i,p.j,uv_i[keep],uv_j[keep]))
        weights = np.clip(p.scores[keep],0.05,1) if p.scores is not None else np.ones(keep.sum())
        corr[p.i].append(np.column_stack([uv_i[keep], r[keep] / 2,weights]))
        corr[p.j].append(np.column_stack([uv_j[keep], -r[keep] / 2,weights]))

    fields: dict[int, OffsetField] = {}
    for i in used:
        x0, y0, w, h = rois[i]
        field = OffsetField((float(x0), float(y0)), (float(w), float(h)),
                            np.zeros((GRID, GRID, 2), np.float32), stats=stats, image_index=i)
        samples = np.vstack(corr[i]) if corr[i] else None
        if samples is not None and len(samples):
            gx = np.linspace(x0, x0 + w, GRID)
            gy = np.linspace(y0, y0 + h, GRID)
            vx, vy = np.meshgrid(gx, gy)
            verts = np.stack([vx.ravel(), vy.ravel()], axis=1)          # (G²,2)
            # Space-balanced cap and blocked accumulation avoid G²×M broadcast allocations.
            cell=np.floor((samples[:,:2]-[x0,y0])/[max(w,1),max(h,1)]*32).astype(int)
            _,selection=np.unique(cell,axis=0,return_index=True)
            samples=samples[np.sort(selection)]
            sigma=SIGMA_FRAC*np.hypot(w,h)
            num=np.zeros((len(verts),2));den=np.full((len(verts),1),SHRINK)
            for start in range(0,len(samples),128):
                block=samples[start:start+128]
                dx=verts[:,None,0]-block[None,:,0];dy=verts[:,None,1]-block[None,:,1]
                weights=np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*block[None,:,4]
                num+=weights@block[:,2:4];den+=weights.sum(1,keepdims=True)
            grid=(num/den).reshape(GRID,GRID,2)
            amplitude=np.linalg.norm(grid,axis=2).max()
            # Bound displacement and Jacobian norm: prevent folding/tearing.
            gx=np.linalg.norm(np.diff(grid,axis=1),axis=2).max()/max(w/(GRID-1),1e-6)
            gy=np.linalg.norm(np.diff(grid,axis=0),axis=2).max()/max(h/(GRID-1),1e-6)
            factor=min(1.,min(24*coordinate_scale,0.03*np.hypot(w,h))/max(amplitude,1e-6),0.25/max(gx+gy,1e-6))
            if factor<1:stats['limited_fields']+=1
            field.grid=(grid*factor).astype(np.float32)
        fields[i] = field
    before=[];after=[]
    def displacement(field,uv):
        ux=((uv[:,0]-field.corner[0])/field.size[0]*(GRID-1)).astype(np.float32)[:,None]
        uy=((uv[:,1]-field.corner[1])/field.size[1]*(GRID-1)).astype(np.float32)[:,None]
        return cv2.remap(field.grid,ux,uy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]
    for i,j,a,b in paired_samples:
        before.extend(np.linalg.norm(b-a,axis=1))
        after.extend(np.linalg.norm(b+displacement(fields[j],b)-a-displacement(fields[i],a),axis=1))
    if before:
        stats['expected_sample_residual']={'before_mean':float(np.mean(before)),'after_mean':float(np.mean(after)),
                                         'unit':'compose pixels; forward-field prediction, not visual error'}
    if diagnostics is not None:diagnostics["local_warp"]=stats
    return fields


def warp_with_field(
    img_wp: np.ndarray,
    mask_wp: np.ndarray,
    corner: tuple[int, int],
    field: OffsetField,
    scale: float,                # 当前阶段全景坐标 / 合成分辨率全景坐标 的比值
) -> tuple[np.ndarray, np.ndarray]:
    """对球面投影后的图像应用位移场：输出像素 x 取自 x - D(x)，内容整体移动 +D。"""
    if not field.grid.any():
        return img_wp, mask_wp
    h, w = img_wp.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # 当前像素的合成分辨率全景坐标 → 网格采样位置
    qx = (xs + corner[0]) / scale
    qy = (ys + corner[1]) / scale
    ux = ((qx - field.corner[0]) / max(field.size[0], 1e-6) * (GRID - 1)).astype(np.float32)
    uy = ((qy - field.corner[1]) / max(field.size[1], 1e-6) * (GRID - 1)).astype(np.float32)
    D = cv2.remap(field.grid, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_x = (xs - D[..., 0] * scale).astype(np.float32)
    map_y = (ys - D[..., 1] * scale).astype(np.float32)
    out = cv2.remap(img_wp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Require all bilinear support pixels valid, avoiding dark invalid-edge interpolation.
    support=cv2.remap(mask_wp,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    out_mask=np.where(support>=254,255,0).astype(np.uint8)
    baseline=cv2.erode(mask_wp,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    ratio=np.count_nonzero(out_mask)/max(1,np.count_nonzero(baseline))
    if field.stats is not None:field.stats['coverage_checks'].append({'image':field.image_index,'scale':scale,'relative_to_conservative_support':float(ratio)})
    if ratio<0.98:
        raise UnsafeFieldError('局部形变有效覆盖比保守插值基准下降超过 2%，全流程回退到全局模型')
    return out,out_mask


def pad_support(image,mask,corner,pad):
    """Reserve bounded displacement around ROI; moved edge pixels remain in the canvas."""
    return (cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REFLECT),
            cv2.copyMakeBorder(mask,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0),
            (corner[0]-pad,corner[1]-pad))
