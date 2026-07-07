# Pano — 复刻并超越 AutoStitch 的全景拼接

从零实现的全景图像拼接项目（2026-07-07 启动）。目标：先复刻 AutoStitch（Brown 2007）级别的经典流水线，再用现代技术超越它。技术路线依据见 [pano.md](pano.md)（拼接技术 20 年演进笔记）。

## 当前能力

**双引擎**（网页与命令行均可选）：

- **基线引擎**：`cv2.Stitcher`——AutoStitch 作者 Brown 的算法在 OpenCV 中的官方实现，作为质量基准。
- **超越版引擎**：自建流水线，相对 AutoStitch 有六个升级点：
  1. **深度特征匹配**（DISK + LightGlue，kornia 实现；SIFT 自动回退）——弱纹理/暗光场景从"完全失败"变为"可拼"
  2. **MAGSAC++** 几何估计（替代经典 RANSAC）
  3. **局部网格视差校正**（APAP 思想）——手持拍摄的近景错位/鬼影消除，AutoStitch 与 OpenCV Stitcher 均无此能力
  4. **GraphCut 最优接缝**（颜色+梯度代价），接缝自动避开显眼区域
  5. **分块增益曝光补偿**（AutoStitch 仅全局增益）
  6. **可选投影方式**：球面/平面（直线不弯）/柱面/墨卡托
- **自动裁剪**：有效区域最大内接矩形去黑边，前端裁剪框可四边/四角拖动微调，canvas 本地导出。

**网页界面**：拖拽上传 → 选引擎/后端/投影 → 双引擎对比 → 裁剪 → 下载。

## 实测数据

合成场景评测（真实焦距 560px，5 张图应全部连接，`python tools/benchmark.py` 可复跑）：

| 场景 | SIFT | DISK+LightGlue |
|---|---|---|
| 纹理丰富、72% 重叠 | ✅ 焦距 560 | ✅ 焦距 560 |
| 弱纹理（±6 灰阶低对比度） | ❌ 完全失败 | ✅ 4/5 张，焦距误差 4% |
| 低重叠 24% | ✅ 焦距 562 | ✅ 焦距 561 |
| 暗光（亮度 22% + 噪声） | ❌ 完全失败 | ✅ 全部成功，焦距 557 |
| 弱纹理 + 低重叠 | ❌ 完全失败 | ⚠️ 4/5 张 |

真实照片（7 张竖拍室内，近景视差）：两引擎均成功；超越版视差校正后消除了门板/椅子处的重影，质量与基线同级或更优。

## 使用

```powershell
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kornia   # 深度后端；缺失时自动回退 SIFT

# 网页（http://127.0.0.1:8000）
python -m uvicorn app.main:app --port 8000

# 命令行
python -m app.stitcher.cli 照片目录 -o output/pano.jpg --engine both --projection plane

# 合成测试集与评测
python tools/make_testset.py normal photos_test
python tools/benchmark.py
```

深度模型权重首次运行自动下载到 `~/.cache/torch/hub/checkpoints/`（`depth-save.pth`、`disk_lightglue_v0-1_arxiv-pth`），网络不畅可手动放置。

## 架构

```
app/
  stitcher/            # 拼接核心：零 Web 依赖，为 C++ 迁移做准备（关键步骤注释 C++ API 对照）
    baseline.py        # 引擎1：cv2.Stitcher
    pipeline.py        # 引擎2：特征匹配→MAGSAC→焦距估计→生成树→光束法平差→波形校正→合成
    features.py        # 特征后端：SIFT / DISK+LightGlue（统一接口）
    localwarp.py       # 局部网格视差校正（warpPoint 驱动，自动适配任意投影）
    autocrop.py        # 最大内接矩形自动裁剪
    cli.py             # 命令行入口
  main.py              # FastAPI（薄壳）
  static/index.html    # 单页前端（含裁剪编辑器）
tools/
  make_testset.py      # 合成测试场景生成（5 种难度）
  benchmark.py         # SIFT vs 深度后端评测
```

## Roadmap

### 短期（原型打磨）
- [ ] 运动物体/鬼影处理：接缝代价加入差异惩罚，或多帧择优保留清晰帧
- [ ] 拼接进度实时推送到前端（SSE），替代现在的计时器
- [ ] 参数面板：输出分辨率上限、JPEG 质量、投影中心微调
- [ ] 深度后端提速：DISK/LightGlue 导出 ONNX + onnxruntime（CPU 推理预计 2-3×，同时为移动端铺路）
- [ ] 真实弱纹理/夜景照片集实测（合成评测已验证，待真实场景确认）

### 中期（移动端迁移，平台未定：iOS / 鸿蒙 / Android）
- [ ] 拼接核心 C++ 化：`pipeline.py` 逐模块翻译（OpenCV C++ API 一一对应，光束法平差改用 `cv::detail::BundleAdjusterRay` 或 Ceres）
- [ ] 深度模型移动端推理：ONNX → CoreML（iOS）/ NNAPI（Android）/ NNRt（鸿蒙）
- [ ] 移动端交互：取景时陀螺仪引导拍摄（重叠率提示）、拍完即拼
- [ ] 核心库单元测试 + 合成集回归自动化（现为手动跑 benchmark）

### 长期（对齐 pano.md 第三阶段）
- [ ] 生成式补边：扩散模型填充不规则黑边（替代裁剪，"多拍少裁"）
- [ ] Transformer 稠密匹配（LoFTR / RoMa）作为第三特征后端，攻极端低重叠
- [ ] 从 2D 全景到 3D：3D Gaussian Splatting 场景重建，360° 可漫游

## 已知边界

- 视差校正是弹性近似，极端近景（<1m 障碍物 + 大幅平移）仍可能残留错位——根治需相机平移建模（超出旋转全景范畴）
- 平面投影在视场角 >~110° 时边缘拉伸明显，>150° 输出发散（已有面积保险与提示）
- 弱纹理的物理极限：重叠区若连深度特征都提不出 12 个内点（如纯渐变天空），该图会被丢弃而非强行错拼
