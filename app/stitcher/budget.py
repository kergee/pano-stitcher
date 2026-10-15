"""Pre-allocation projection planning (cv::RotationWarper::warpRoi).

Byte estimates include pyramids, masks, float maps and temporary copies.
They are conservative planning estimates, not an OS hard RSS limit.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ComposeBudget:
    max_output_side: int = 0
    max_canvas_pixels: int = 24_000_000
    max_image_pixels: int = 16_000_000
    memory_mb: float = 768

    def __post_init__(self):
        if not np.isfinite([self.max_output_side,self.max_canvas_pixels,self.max_image_pixels,self.memory_mb]).all():
            raise ValueError("预算必须为有限数值")
        if self.max_output_side < 0 or min(self.max_canvas_pixels, self.max_image_pixels, self.memory_mb) <= 0:
            raise ValueError('输出预算必须为正（边长 0 表示仅受内存预算限制）')


def canvas_roi(rois):
    if not rois:
        raise ValueError('投影 ROI 为空')
    for r in rois:
        if len(r) != 4 or not np.isfinite(r).all() or min(r[2:]) <= 0 or max(abs(v) for v in r) > 2**29:
            raise ValueError('投影坐标或尺寸无效')
    x = min(r[0] for r in rois); y = min(r[1] for r in rois)
    return (x, y, max(r[0]+r[2] for r in rois)-x, max(r[1]+r[3] for r in rois)-y)


def estimate_bytes(rois, resident_bytes=0):
    _, _, w, h = canvas_roi(rois)
    # canvas pyramids/accumulators/result/crop temporaries + one remap + seams
    areas = [r[2]*r[3] for r in rois]
    return int(resident_bytes + 64*w*h + 96*max(areas) + 24*sum(areas))


def plan_projection(sizes, rotations, focal, projection, budget, resident_bytes=0, padding_pixels=0, intrinsics=None):
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError('焦距无效')
    # A plane crossing its horizon has an unbounded interior even if corners look finite.
    if projection == 'plane':
        for i, (w,h) in sizes.items():
            rays = np.array([[x/focal,y/focal,1] for x in (-w/2,w/2) for y in (-h/2,h/2)])
            z = (rays @ rotations[i].T)[:,2]
            if z.min() <= 1e-3:
                raise ValueError('平面投影跨越投影地平线，请改用球面或柱面')
    if any(np.asarray(R).shape!=(3,3) or not np.isfinite(R).all() for R in rotations.values()):
        raise ValueError("旋转矩阵必须是有限 3×3 数组")
    scale = 1.0
    for _ in range(24):
        warper = cv2.PyRotationWarper(projection, float(focal*scale))
        rois = {}
        for i,(w,h) in sizes.items():
            sw,sh = max(1,round(w*scale)),max(1,round(h*scale))
            K = np.array([[focal*scale,0,sw/2],[0,focal*scale,sh/2],[0,0,1]],np.float32)
            if intrinsics is not None:
                K=np.ascontiguousarray(intrinsics[i],np.float32).copy();K[:2,:]*=scale
            raw=warper.warpRoi((sw,sh),K,np.ascontiguousarray(rotations[i],np.float32))
            pad=int(np.ceil(padding_pixels*scale))
            rois[i]=(raw[0]-pad,raw[1]-pad,raw[2]+2*pad,raw[3]+2*pad)
        dst = canvas_roi(list(rois.values()))
        peak = estimate_bytes(list(rois.values()),resident_bytes)
        ratios = [1.0, np.sqrt(budget.max_canvas_pixels/(dst[2]*dst[3])),
                  np.sqrt(budget.max_image_pixels/max(r[2]*r[3] for r in rois.values()))]
        available = budget.memory_mb*1024**2-resident_bytes
        if available <= 0:
            raise ValueError('输入和工作数组已经超过内存预算')
        ratios.append(np.sqrt(available/max(1,peak-resident_bytes)))
        if budget.max_output_side:
            ratios.append(budget.max_output_side/max(dst[2:]))
        factor=min(ratios)
        if factor >= 1:
            return scale,warper,rois,{'compose_scale':scale,'canvas':dst,'estimated_peak_bytes':peak}
        scale *= factor*0.97
        if scale < 1e-4 or min(min(s) for s in sizes.values())*scale < 8:
            raise ValueError('预算内投影过小或画布跨度异常，拒绝分配')
    raise ValueError('无法在预算内规划投影')
