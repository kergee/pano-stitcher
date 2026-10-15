# Pano — 参考 AutoStitch 算法族的全景拼接

从零实现的全景图像拼接项目（2026-07-07 启动）。目标：建立经典旋转全景流水线，再以受控实验评估现代技术。当前没有与原 AutoStitch EXE 同输入同参数的受控对照，不宣称已经超越。技术路线依据见 [pano.md](pano.md)（拼接技术 20 年演进笔记）。

## 当前能力

**双引擎**（网页与命令行均可选）：

- **基线引擎**：`cv2.Stitcher`——经典旋转全景算法族的 OpenCV 实现，作为质量基准；不是 AutoStitch 软件的官方实现。
- **超越版引擎**：自建流水线，提供以下可独立验证的步骤：
  1. **深度特征匹配**（DISK + LightGlue，kornia 实现；SIFT 自动回退）；明确指定 deep 不回退，auto 报告实际后端。本轮深度权重不完整，未验证深度质量
  2. **MAGSAC++** 几何估计（替代经典 RANSAC）
  3. **局部高斯残差位移场**——近似减小视差，加入互检去重、邻域一致性、幅度/梯度限制及覆盖回退；不是完整 APAP Moving DLT
  4. **GraphCut 最优接缝**（颜色+梯度代价），接缝自动避开显眼区域
  5. **分块增益曝光补偿**；曝光补偿后搜索接缝
  6. **可选投影方式**：球面/平面（直线不弯）/柱面/墨卡托
- **自动裁剪**：有效区域最大内接矩形去黑边，前端裁剪框可四边/四角拖动微调，canvas 本地导出。

**网页界面**：拖拽上传 → 选引擎/后端/投影/输出边长/JPEG 质量 → 实时进度 → 双引擎对比 → 裁剪 → 下载。

## 2026-10-15 验证结果

Python 3.12.10 / OpenCV 5.0.0 / numpy 2.5.3 / scipy 1.18.1，OpenCV 固定 2 线程。使用确定性输入清单与 SHA256，不扫描旧成图；七张真实样例为原始**山景**照片。

| 场景 | pro/SIFT 成图 | 有限几何检查 | 视觉检查 |
|---|---|---|---|
| 普通 | 5/5 | 焦距 559.989px，误差 0.0019% | 代表图连续，未见明显大范围拉裂 |
| 低重叠 | 5/5 | 焦距 561.745px，误差 0.312% | 代表图连续；需裁剪外部无效区域 |
| 混合尺寸（同焦距、中心裁切） | 5/5 | 焦距 560.469px，误差 0.084% | 代表图连续，保守裁剪框有效 |
| 弱纹理 / 暗光 / 极端 / 重复纹理 | 失败，明确报告 | 未通过 | 没有结果图，不能称质量合格 |
| 真实山景 | 7/7，1517×1003 | BA 工作像素 RMS 5.001→0.875；相机真值未知 | 已查看成图和 1329×894 裁剪图；不是完整视觉认证 |

20 项反例测试通过，覆盖 Unicode/长路径、失败退出、画布跨度、混合尺寸、局部坏匹配/真实一致视差、位移限制、连通分量、BA 回退、空掩码及单像素洞。四种投影的小尺寸输出通过，跨投影地平线的平面输出在分配前拒绝。Web ASGI multipart / 400 / 422 / 异步任务 / SSE / 结果接口通过；没有浏览器或桌面包验证。

普通、低重叠、真实照片各 3 次独立进程测量，中位数默认耗时分别从 2.25→2.68s、1.58→1.83s、4.98→6.87s；新增可靠性检查有成本。480px 输出下，低重叠和真实照片的实际峰值工作集从约 259→205MiB、291→236MiB；这些包含 Python/库开销，不等于仅合成数组预算。快速模式是另一次单次对照，不作统计性速度承诺。

DISK 完整但 LightGlue 仅有不完整缓存；本轮没有下载深度模型。旧 README 的 weak/dark 深度成功数字属于历史结果，本轮未复核，不能用于当前后端验收。`--regression` 仍跑 SIFT 两项，再把 dark/deep 标为未验证并返回 1。

详细修改、实验取舍及剩余问题见 [中文变更说明](docs/2026-10-15-改进与验证.md)。原始指标、输入清单和成图保存在本机 `output/verification-20261015/`（忽略入 Git）。

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
python tools/benchmark.py --scenarios normal lowoverlap mixed --backends sift

# 最小回归（normal/sift、lowoverlap/sift、dark/deep；失败或未验证返回非零）
python tools/benchmark.py --regression
# 首次需要联网下载深度权重时显式允许下载
python tools/benchmark.py --regression --allow-model-download

# 桌面壳开发/打包（Tauri v2 + Python sidecar）
npm install
npm run tauri:dev
npm run tauri:build
# 若发布包必须内置 DISK+LightGlue（体积/构建时间显著增加）
powershell -ExecutionPolicy Bypass -File tools/package_tauri.ps1 -BundleDeep
```

显式运行 deep/auto 时，深度模型权重首次运行可能自动下载到 `~/.cache/torch/hub/checkpoints/`（`depth-save.pth`、`disk_lightglue_v0-1_arxiv-pth`），网络不畅可手动放置。

## 架构

```
app/
  stitcher/            # 拼接核心：零 Web 依赖，为 C++ 迁移做准备（关键步骤注释 C++ API 对照）
    baseline.py        # 引擎1：cv2.Stitcher
    pipeline.py        # 引擎2：特征匹配→MAGSAC→焦距估计→生成树→光束法平差→波形校正→合成
    features.py        # 特征后端：SIFT / DISK+LightGlue（统一接口）
    localwarp.py       # 局部网格视差校正（warpPoint 驱动，自动适配任意投影）
    autocrop.py        # 保守内接矩形，原分辨率支持校验
    budget.py          # 合成前 ROI、画布、临时数组预算
    imageio.py         # Unicode 文件读写、原子替换、哈希输入清单
    cli.py             # 命令行入口
  main.py              # FastAPI（薄壳）
  static/index.html    # 单页前端（含裁剪编辑器）
tools/
  make_testset.py      # 合成场景（普通/低重叠/弱纹理/暗光/极端/重复/混合尺寸）
  benchmark.py         # SIFT vs 深度后端评测 + 最小回归
  verify_changes.py    # 独立进程前后/选项对照，测实际峰值工作集
  verify_web.py        # ASGI 路由与 SSE 验证
tests/                 # 可靠性反例测试
src-tauri/             # Tauri 桌面壳；发布版通过 PyInstaller sidecar 启动 FastAPI
```

## Roadmap

### 短期（原型打磨）
- [ ] 运动物体/鬼影处理：接缝代价加入差异惩罚，或多帧择优保留清晰帧
- [x] 拼接进度实时推送到前端（SSE），替代现在的计时器
- [x] 参数面板：输出分辨率上限、JPEG 质量、投影/后端选择
- [x] Tauri 桌面打包骨架：开发模式拉起 FastAPI，发布模式使用 PyInstaller sidecar
- [ ] 参数面板：投影中心微调
- [ ] 深度后端提速：DISK/LightGlue 导出 ONNX + onnxruntime（CPU 推理预计 2-3×，同时为移动端铺路）
- [ ] 真实弱纹理/夜景照片集实测（本轮 SIFT 未通过，深度未验证）

### 中期（移动端迁移，平台未定：iOS / 鸿蒙 / Android）
- [ ] 拼接核心 C++ 化：`pipeline.py` 逐模块翻译（OpenCV C++ API 一一对应，光束法平差改用 `cv::detail::BundleAdjusterRay` 或 Ceres）
- [ ] 深度模型移动端推理：ONNX → CoreML（iOS）/ NNAPI（Android）/ NNRt（鸿蒙）
- [ ] 移动端交互：取景时陀螺仪引导拍摄（重叠率提示）、拍完即拼
- [x] 核心可靠性反例测试 + 固定输入合成回归
- [ ] CI 集成、真实夜景/弱纹理/近景视差与移动端设备验收

### 长期（对齐 pano.md 第三阶段）
- [ ] 生成式补边：扩散模型填充不规则黑边（替代裁剪，"多拍少裁"）
- [ ] Transformer 稠密匹配（LoFTR / RoMa）作为第三特征后端，攻极端低重叠
- [ ] 从 2D 全景到 3D：3D Gaussian Splatting 场景重建，360° 可漫游

## 已知边界

- 视差校正是弹性近似，极端近景（<1m 障碍物 + 大幅平移）仍可能残留错位——根治需相机平移建模（超出旋转全景范畴）
- 平面投影在视场角 >~110° 时边缘拉伸明显，>150° 输出发散（合成前检查投影地平线、单图 ROI、最终画布及估算内存；超预算缩小或拒绝）
- 弱纹理的物理极限：重叠区若连深度特征都提不出 12 个内点（如纯渐变天空），该图会被丢弃而非强行错拼

## 核心接口和实验模式

`stitch_pro(images, backend_name="sift", max_output_side=1600, memory_mb=768, local_warp=True, quality="full", pair_mode="all", ba_mode="single")` 不依赖 FastAPI。新增参数均有默认值；`StitchResult.diagnostics` 报告分量、BA、实际候选数、合成比例/预算、曝光重叠误差、局部场和有效覆盖。

`max_output_side=0` 仍受默认 24MP 画布 / 16MP 单图 / 768MiB 数组估算约束；限制在投影、remap、blender.prepare 前生效。预算估算包含输入、金字塔、掩码、映射和临时副本，不是 OS 硬 RSS 上限；不包括模型缓存、解释器和全部第三方分配。混合尺寸按批次最大面积确定共同工作/接缝比例。中心裁切保持共享焦距；独立缩放、异焦段或偏心裁切需要逐图内参，本轮未支持。

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s tests -v
python -m app.stitcher.cli photos_bench/normal --manifest photos_bench/normal/inputs.json --backend sift --max-output-side 1600 --memory-mb 512 --report output/诊断.json -o output/全景.jpg
python -m app.stitcher.cli photos_bench/normal --backend sift --quality fast -o output/快速.jpg
python tools/verify_changes.py --real-manifest output/verification-20261015/real-inputs.json
```

CLI 自动使用目录内 `inputs.json`；任一引擎失败、解码失败或写入失败返回非零。没有清单的普通照片目录会枚举照片，排除本次输出路径；用户应将其他旧成图移到另一个目录。生成器覆盖自己的五张 `view_XX.jpg` 并重建清单，不把其他文件纳入评测。

- `--quality fast`：关闭局部场、两层 OpenCV 金字塔频带、默认最长边 1600，仍保留曝光补偿和 GraphCut；不是 AutoStitch Demo 的同尺寸双频带实现。
- `--pair-mode ordered`：仅用于已知拍摄顺序，±2 邻居和首尾候选；断连时尝试全对兜底。可能漏掉仍连通组中的约束；真实七图仅召回全对有效边的 8/15，默认仍是 `all`。
- `--ba-mode staged`：实验性的树初始化→全局→较小尺度 soft_l1 精修。没有证明稳定改善，默认仍是 `single`；与 AutoStitch 二维径向 Huber 不等价。
- `--no-local-warp`：用于全局模型对照。局部形变回退会重做两阶段，避免接缝和融合使用不同几何。

基线通过 OpenCV 的估计/合成分离接口预先限制合成；其无效掩码仍由颜色近似估计，真实黑色物体不能与支持外区域严格区分。自动裁剪返回建议框，原全景仍保留完整有效内容。

## GitHub Actions

推送 main / codex 分支或提交 PR 时运行 `.github/workflows/build.yml`，也支持手动 Run workflow。先在 Ubuntu、Windows 的 Python 3.12 上运行可靠性测试、普通/低重叠/混合尺寸 SIFT 回归及 ASGI/SSE 检查，通过后构建 Windows x64 Tauri 安装包。

在 Actions 运行详情的 Artifacts 下载 `Pano-Windows-x64-SIFT`（NSIS EXE / MSI）；核验记录为 `verification-ubuntu-latest` / `verification-windows-latest`，保留 14 天。默认不打包 torch/kornia 或下载深度权重，不自动发布 Release。安装包未做代码签名；CI 后端启动检查不等于 Windows WebView 完整交互验收。
