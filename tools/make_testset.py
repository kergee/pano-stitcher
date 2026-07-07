"""生成合成测试照片：从程序生成的环境全景中，用旋转相机模型渲染若干张
互相重叠的针孔视图（已知真实焦距/旋转），用于端到端验证与后端对比评测。

场景：
  normal      纹理丰富、~72% 重叠（基准场景）
  lowtex      弱纹理：低对比度的天空/墙面，SIFT 的对比度阈值会滤掉大部分特征
  lowoverlap  低重叠：纹理丰富但相邻仅 ~24% 重叠
  dark        暗光：亮度压到 22% + 传感器噪声
  extreme     弱纹理 + 低重叠（~37%）叠加

用法：python tools/make_testset.py [场景=normal] [输出目录=photos_test]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PANO_W, PANO_H = 4096, 1600   # 环境图（经度 ±90°，纬度 ±35°）
LON_RANGE = np.pi
LAT_RANGE = 0.7
VIEW_W, VIEW_H, FOCAL = 720, 540, 560.0  # hfov ≈ 65.5°

SCENARIOS = {
    "normal": dict(rich=True, yaw_step=18, dark=False),
    "lowtex": dict(rich=False, yaw_step=18, dark=False),
    "lowoverlap": dict(rich=True, yaw_step=50, dark=False),
    "dark": dict(rich=True, yaw_step=18, dark=True),
    "extreme": dict(rich=False, yaw_step=42, dark=False),
}
N_VIEWS = 5


def make_environment(rich: bool) -> np.ndarray:
    rng = np.random.default_rng(42)
    if rich:
        base = rng.integers(0, 255, (PANO_H // 16, PANO_W // 16, 3), np.uint8)
        env = cv2.resize(base, (PANO_W, PANO_H), interpolation=cv2.INTER_CUBIC)
        env = cv2.GaussianBlur(env, (0, 0), 9)
    else:
        # 弱纹理：平滑渐变 + 振幅仅 ±6 灰阶的低对比度云絮
        gy = np.linspace(0, 1, PANO_H)[:, None, None]
        col = (np.float32([225, 200, 150]) * (1 - gy * 0.35) + np.float32([180, 185, 190]) * gy * 0.35)
        env = np.repeat(col.astype(np.float32), PANO_W, axis=1)
        cloud = rng.normal(0, 1, (PANO_H // 8, PANO_W // 8)).astype(np.float32)
        cloud = cv2.resize(cv2.GaussianBlur(cloud, (0, 0), 5), (PANO_W, PANO_H))
        env = np.clip(env + cloud[..., None] * 6.0, 0, 255).astype(np.uint8)

    sky = np.linspace(1.0, 0.25, PANO_H // 3)[:, None, None]
    env[: PANO_H // 3] = (np.float32([235, 206, 135]) * (1 - sky * 0.3)
                          + env[: PANO_H // 3] * sky * 0.3).astype(np.uint8)

    n_obj = 260 if rich else 10  # 弱纹理场景只留极少量锚点物体
    for _ in range(n_obj):
        p1 = (int(rng.integers(0, PANO_W)), int(rng.integers(PANO_H // 4, PANO_H)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            p2 = (p1[0] + int(rng.integers(20, 240)), p1[1] + int(rng.integers(20, 160)))
            cv2.rectangle(env, p1, p2, color, -1 if rng.random() < 0.7 else 3)
        elif kind == 1:
            cv2.circle(env, p1, int(rng.integers(8, 80)), color, -1)
        else:
            cv2.putText(env, f"P{int(rng.integers(0, 999)):03d}", p1,
                        cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.8, 2.2), color, 2)
    return env


def render_view(env: np.ndarray, yaw_deg: float, dark: bool, rng) -> np.ndarray:
    """针孔相机渲染：像素 → 射线 → 经纬度 → 环境图采样。"""
    yaw = np.radians(yaw_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    xs, ys = np.meshgrid(np.arange(VIEW_W), np.arange(VIEW_H))
    d = np.stack([(xs - VIEW_W / 2) / FOCAL, (ys - VIEW_H / 2) / FOCAL, np.ones_like(xs, np.float64)], -1)
    d = d @ Ry.T
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    mx = ((lon + LON_RANGE / 2) / LON_RANGE * PANO_W).astype(np.float32)
    my = ((lat + LAT_RANGE / 2) / LAT_RANGE * PANO_H).astype(np.float32)
    view = cv2.remap(env, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    gain = 1.0 + 0.12 * np.sin(yaw)  # 模拟不同曝光
    view *= gain
    if dark:  # 暗光 + 传感器噪声
        view = view * 0.22 + rng.normal(0, 10, view.shape)
    return np.clip(view, 0, 255).astype(np.uint8)


def generate(scenario: str, out: Path) -> None:
    cfg = SCENARIOS[scenario]
    out.mkdir(parents=True, exist_ok=True)
    env = make_environment(cfg["rich"])
    rng = np.random.default_rng(7)
    yaws = [(k - (N_VIEWS - 1) / 2) * cfg["yaw_step"] for k in range(N_VIEWS)]
    for k, yaw in enumerate(yaws):
        p = out / f"view_{k:02d}.jpg"
        cv2.imwrite(str(p), render_view(env, yaw, cfg["dark"], rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"生成 {p}（yaw={yaw:.0f}°，真实焦距 {FOCAL}px）")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "photos_test")
    generate(scenario, out)
