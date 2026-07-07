# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

全景图像拼接项目：复刻并超越 AutoStitch。Python 原型（未来计划迁移 iOS/鸿蒙/Android，因此**拼接核心 `app/stitcher/` 与 Web 层严格解耦**——核心只依赖 OpenCV/numpy/torch，禁止 import FastAPI；关键步骤注释了对应的 C++ API 名以便日后翻译）。

- `README.md` — 项目现状、实测数据、**Roadmap**（短期打磨 / 中期移动端迁移 / 长期生成式与 3D）
- `pano.md` — 拼接技术 20 年演进笔记（中文），本项目的技术路线依据
- `autostitch64/` — AutoStitch.exe 官方二进制（对比基准），配置在其 `ini.ini`

## 常用命令

```powershell
pip install -r requirements.txt   # 另需: pip install torch --index-url https://download.pytorch.org/whl/cpu; pip install kornia
python tools/make_testset.py normal photos_test   # 生成合成测试图（场景：normal/lowtex/lowoverlap/dark/extreme，真实焦距 560px）
python tools/benchmark.py                         # SIFT vs 深度后端 5 场景对比评测（photos_bench/）
python -m app.stitcher.cli photos_test -o output/pano.jpg --engine both --backend sift   # 命令行拼接（--projection plane 可拉直直线）
python -m uvicorn app.main:app --port 8000        # 网页界面（或 preview_start "pano-web"）
```

无正式测试套件；回归验证方式 = 跑合成测试集，检查 pro 引擎输出的焦距 ≈560px 且成图与 baseline 布局一致。

## 架构

两个拼接引擎，统一由 `app/main.py`（FastAPI + `app/static/index.html` 单页）暴露：

- **引擎 1 基线** `app/stitcher/baseline.py`：`cv2.Stitcher`（即 AutoStitch 作者 Brown 的算法），质量基准。
- **引擎 2 超越版** `app/stitcher/pipeline.py`：自建旋转相机模型流水线——深度特征匹配（`features.py`：DISK+LightGlue via kornia，SIFT 回退）→ MAGSAC 单应性 → 闭式焦距估计 + 最大生成树串接旋转 → scipy 光束法平差 → `cv2.detail` 波形校正/投影（球面/平面/柱面/墨卡托，`projection` 参数）→ 局部网格弹性校正（`localwarp.py`，APAP 思想消视差：匹配点残差的高斯加权位移场 + remap，接缝与合成两阶段共用一场）→ 分块曝光补偿/GraphCut 接缝/多频段融合 → 自动裁剪（`autocrop.py`，最大内接矩形）。注意视差校正用的是 `_Pair.all_i/all_j`（全部匹配点）——MAGSAC 内点只含符合全局单应性的匹配，视差点恰好在内点之外。

### 关键约定（易错点）

- **旋转矩阵方向**：pipeline 内部 `rot[i]` 是世界→相机（`p = K·R·X`）；OpenCV 的 `PyRotationWarper`/`waveCorrect` 期望相机→世界。仅在 stitch_pro 第 4 步边界处转置一次，且必须 `np.ascontiguousarray`（`.T` 的视图会被 cv2 绑定以 TypeError 拒收）。
- **坐标中心化**：所有单应性/焦距估计在以图像中心为原点的坐标系下进行（`_center`），K = diag(f,f,1)；只有喂给 warper 的 K 才带主点偏移。
- **投影无关性**：`localwarp` 的全景坐标一律通过 `warper.warpPoint` 计算，不手写投影公式——换投影方式（`PROJECTIONS`）时视差校正自动跟随。平面投影视角过大时输出会发散，`_compose` 有面积保险（>6e8 报错）。
- **后端相关阈值**：`CONF_THRESH`/`MIN_INLIERS` 是按后端区分的 dict——Brown 置信度公式为 SIFT+最近邻设计，LightGlue 已内部过滤误匹配，故 deep 阈值放宽（0.9→0.45、20→12）。
- 深度后端全局缓存（`features.get_backend`），模型只加载一次；首次运行会从网络下载 DISK/LightGlue 权重到 `~/.cache/torch`。
- CLI 输出到 GBK 控制台会乱码，跑命令时加 `PYTHONIOENCODING=utf-8`；调试静默崩溃用 `python -u -X faulthandler`（stdout 管道下是块缓冲，崩溃点会显示得比实际早）。
