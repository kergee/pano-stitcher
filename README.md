# Pano — 参考与借鉴 AutoStitch 的全景拼接实验项目

Pano 是一个全景图像拼接实验项目（2026-07-07 启动），参考 AutoStitch 相关论文与已有调研，借鉴特征匹配、旋转相机估计和多频段融合等思路，结合 OpenCV 与可选深度特征构建独立流水线。项目能力以本仓库记录的实测为依据。技术路线笔记见 [pano.md](pano.md)。

## 当前能力

**双引擎**（网页与命令行均可选）：

- **基线引擎**：`cv2.Stitcher`——OpenCV 提供的经典旋转全景流水线，作为本项目的对照基线。
- **自建引擎（pro）**：提供以下可独立验证的步骤：
  1. **深度特征匹配**（DISK + LightGlue，kornia 实现；SIFT 自动回退）；明确指定 deep 不回退，auto 报告实际后端。本轮深度权重不完整，未验证深度质量
  2. **MAGSAC++** 几何估计（替代经典 RANSAC）
  3. **局部高斯残差位移场**——近似减小视差，加入互检去重、邻域一致性、幅度/梯度限制及覆盖回退；不是完整 APAP Moving DLT
  4. **GraphCut 接缝搜索**（颜色+梯度代价）；不保证识别或避开所有显眼物体
  5. **分块增益曝光补偿**；曝光补偿后搜索接缝
  6. **可选投影方式**：球面/平面（直线不弯）/柱面/墨卡托
- **自动裁剪**：返回保守内接矩形建议框，并在原分辨率掩码验证支持；不保证严格最大面积。前端可拖动四边/四角微调并本地导出。

**网页界面**：拖拽上传 → 选引擎/后端/投影/输出边长/JPEG 质量 → 实时进度 → 双引擎对比 → 裁剪 → 下载。

## 项目状态（2026-10-15）

当前是**已有跨平台核心回归和 Windows 构建产物的 Python 原型**，正在补齐真实场景与桌面使用验收。算法、安装包、完整产品体验分别记录状态。

| 层面 | 已有证据 | 尚需验收 |
|---|---|---|
| 核心可靠性 | 20 项反例测试；普通/低重叠/混合尺寸 SIFT 回归；合成前预算、匹配与 BA 回退 | 真实暗光/弱纹理/近景视差、移动物体、异焦段输入 |
| Web 接口 | Ubuntu、Windows 的 ASGI 上传、错误码、任务/SSE/结果检查通过 | 浏览器拖拽、进度、裁剪、导出和异常恢复的完整交互 |
| Windows 分发 | GitHub Actions 成功生成 NSIS EXE、MSI，打包后端启动检查通过 | 干净 Windows 安装/卸载、WebView 交互、端口冲突与进程退出 |
| 深度后端 | 已接入 DISK + LightGlue，明确 deep 不静默回退 | 本机 LightGlue 权重不完整，本轮未验证深度质量；默认 CI/安装包仅覆盖 SIFT |
| 移动端 | 核心与 FastAPI 分离，保留 OpenCV/C++ 对照注释 | 平台选择、原生接口、设备内存/速度/质量验证均未完成 |

[已通过的 Actions 运行（提交 0e4a0b3）](https://github.com/kergee/pano-stitcher/actions/runs/38666481283) 中，`Pano-Windows-x64-SIFT` 包含安装包；下载方法见文末。完整桌面交互尚未由这次 CI 证明。

## 2026-10-15 验证结果

Python 3.12.10 / OpenCV 5.0.0 / numpy 2.5.3 / scipy 1.18.1，OpenCV 固定 2 线程。使用确定性输入清单与 SHA256，不扫描旧成图；七张真实样例为原始**山景**照片。

| 场景 | pro/SIFT 成图 | 有限几何检查 | 视觉检查 |
|---|---|---|---|
| 普通 | 5/5 | 焦距 559.989px，误差 0.0019% | 代表图连续，未见明显大范围拉裂 |
| 低重叠 | 5/5 | 焦距 561.745px，误差 0.312% | 代表图连续；需裁剪外部无效区域 |
| 混合尺寸（同焦距、中心裁切） | 5/5 | 焦距 560.469px，误差 0.084% | 代表图连续，保守裁剪框有效 |
| 弱纹理 / 暗光 / 极端 / 重复纹理 | 失败，明确报告 | 未通过 | 没有结果图，不能称质量合格 |
| 真实山景 | 7/7，1517×1003 | BA 工作像素 RMS 5.001→0.875；相机真值未知 | 已查看成图和 1329×894 裁剪图；不是完整视觉认证 |

20 项反例测试通过，覆盖 Unicode/长路径、失败退出、画布跨度、混合尺寸、局部坏匹配/真实一致视差、位移限制、连通分量、BA 回退、空掩码及单像素洞。四种投影的小尺寸输出通过，跨投影地平线的平面输出在分配前拒绝。Web ASGI multipart / 400 / 422 / 异步任务 / SSE / 结果接口通过。后续 Actions 已补充 Windows 安装包构建与打包后端启动检查；浏览器和安装后完整交互仍未验证。

普通、低重叠、真实照片各 3 次独立进程测量，中位数默认耗时分别从 2.25→2.68s、1.58→1.83s、4.98→6.87s；新增可靠性检查有成本。480px 输出下，低重叠和真实照片的实际峰值工作集从约 259→205MiB、291→236MiB；这些包含 Python/库开销，不等于仅合成数组预算。快速模式是另一次单次对照，不作统计性速度承诺。

DISK 完整但 LightGlue 仅有不完整缓存；本轮没有下载深度模型。旧 README 的 weak/dark 深度成功数字属于历史结果，本轮未复核，不能用于当前后端验收。`--regression` 仍跑 SIFT 两项，再把 dark/deep 标为未验证并返回 1。

详细修改、实验取舍及剩余问题见 [中文变更说明](docs/2026-10-15-改进与验证.md)。原始指标、输入清单和成图保存在本机 `output/verification-20261015/`（忽略入 Git）。

## 使用

### SIFT 基础环境

建议使用独立 Python 3.12 环境；本机原全局环境曾出现 Pydantic 与 pydantic-core 不匹配，不能把全局安装视为已验收环境。

```powershell
python -m venv .venv
# 不依赖 PowerShell 激活脚本的执行策略
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONIOENCODING='utf-8'

# 生成固定输入并拼接；显式 sift，不尝试下载深度权重
.\.venv\Scripts\python.exe tools/make_testset.py normal photos_test
.\.venv\Scripts\python.exe -m app.stitcher.cli photos_test --backend sift -o output/pano.jpg --max-output-side 1600

# 网页（http://127.0.0.1:8000）；页面后端先选择 SIFT
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

后续命令中的 `python` 指向已安装依赖的环境；使用上面的独立环境时，可替换为 `.\.venv\Scripts\python.exe`。CLI 和核心默认 `auto`，可能尝试加载深度模型；需要明确可复跑的 SIFT 路径时传 `--backend sift`。

### 测试与可选深度后端

```powershell
python -m unittest discover -s tests -v
python tools/benchmark.py --scenarios normal lowoverlap mixed --backends sift
# 加上 dark/deep 的最小回归：失败或未验证返回非零
python tools/benchmark.py --regression

# 可选：安装深度依赖，并显式允许首次联网下载模型
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install kornia
python tools/benchmark.py --regression --allow-model-download
```

深度权重缓存位于 `~/.cache/torch/hub/checkpoints/`，包括 DISK `depth-save.pth` 与 LightGlue 权重。`.partial` 文件不算完整模型；验收须使用明确的 deep 后端及实际输出，auto/SIFT 回退不能算深度成功。

### 桌面开发与本地打包

在同一终端准备好包含项目依赖的 Python 环境、Node.js、Rust 和 Windows 构建工具后运行：

```powershell
npm ci
npm run tauri:dev
npm run tauri:build
# 实验性的深度依赖收集开关
powershell -ExecutionPolicy Bypass -File tools/package_tauri.ps1 -BundleDeep
```

构建脚本调用 PATH 中的 `python`；使用独立环境时，先将 `.venv/Scripts` 加入当前终端 PATH。`-BundleDeep` 只尝试收集环境中已安装的 torch/kornia，不保证依赖或模型权重完整，也未验证离线深度能力。默认 Actions 安装包为 SIFT 版本。

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

## Roadmap 评估（2026-10-15）

下一阶段应先完成**可安装、可复跑、真实场景质量可判断**的桌面版本。当前主要缺口是深度后端未复核、复杂真实样例不足、完整交互未验收，以及默认模式增加的耗时。移动端全面迁移和更多模型应以后续证据为前提。

### 已完成的基础

- [x] Unicode 文件读写、写入/失败退出检查、固定哈希输入清单。
- [x] 合成前输出与数组预算、混合尺寸工作比例、异常投影拒绝。
- [x] 匹配互检去重、可信视差点筛选、局部位移保护、分量隔离及 BA 诊断/回退。
- [x] 曝光补偿后搜索接缝、空掩码处理、原分辨率裁剪支持检查。
- [x] SSE、输出边长/JPEG 质量/后端/投影参数面板（代码和接口已有；完整 UI 验收另列）。
- [x] 核心反例与合成回归接入 Ubuntu/Windows CI。
- [x] Tauri/PyInstaller 打包、Windows 安装包产物和打包后端启动检查。

### 近期 P0：补齐发布与质量证据

| 顺序 | 工作 | 完成条件 |
|---|---|---|
| 1 | Windows 桌面与浏览器完整验收 | 在干净 Windows 环境安装/启动，验证拖拽、编辑参数、SSE、裁剪、导出、错误恢复、卸载；检查端口占用、慢启动、重复打开和退出后端回收，并保存版本与结果 |
| 2 | 深度后端恢复与严格复核 | 完整且可校验的权重，明确 deep 跑普通/暗光/弱纹理/低重叠；记录实际后端、使用/丢弃图、误差和失败原因。失败如实留档，回退不计通过；覆盖首次下载失败、离线重启与接口兼容 |
| 3 | 建立真实场景质量基准 | 增加可授权留存的弱纹理、夜景、重复纹理、近景视差、曝光差及移动物体样例；固定清单，检查关键结构/接缝/鬼影/色偏，比较 baseline 与 pro、局部场开/关，区分出图、几何检查与人工视觉结论 |
| 4 | 定位默认性能回退 | 先 profile 匹配、局部场、BA、合成和冷启动；同机同输入至少 3 次记录中位数及实际峰值工作集。优化须保留既有几何、裁剪与可信视差收益；暂不承诺固定加速倍数 |

默认耗时目前比改进前增加约 16%–38%，所以性能和质量基准应先于全量 C++ 重写。CI 依赖已记录解析版本，但尚未形成完整锁定与跨版本兼容策略；发布时还应补齐依赖/模型版本、许可证、校验信息及用户可理解的后端状态。

### 后续 P1：有对照收益再扩大能力

| 方向 | 前置条件与取舍 | 验收重点 |
|---|---|---|
| 非全局旋转场景与鬼影 | 先用真实样例定位失败；分别评估差异接缝、择帧、更准确的局部模型 | 关键结构改善且不新增拉裂/色差；不能仅凭匹配残差下降判成功 |
| 逐图内参与多组全景 | 当前中心裁切混合尺寸已测，任意 resize/异焦段不在契约内；先明确输入元数据和输出接口 | 独立缩放/裁切案例可解释；分组不污染焦距/BA，返回可追踪的 used/dropped |
| 大批量候选检索与几何提速 | ordered 仅为顺序启发式，七图有效边仅召回 8/15；默认继续 all | 以全对为参照同时报告边召回、连通率、误差和耗时；评估无序输入、闭环与断连兜底 |
| BA/局部场/深度推理优化 | 依据 P0 profile 选择采样/向量化、稀疏求解或 ONNX；模型导出依赖已通过的深度基准 | 参数与精度差异可复跑，速度和内存实测；staged/fast 的默认切换需独立质量证据 |
| 用户控制与分发完善 | 投影中心微调、更多核心选项进入 UI、进度/取消、签名与版本发布逐项推进 | 入口、编辑、保存、重显和异常恢复可用；核心参数/诊断保持无 Web 依赖 |
| AutoStitch 受控对照 | 使用同一原图清单，固定 EXE 版本、配置及输出范围 | 对比连接、几何/视觉质量、时间和内存，以实测评估参考方案的适用场景 |

### 移动端 P2：先选择平台并验证最小流程

- [ ] 确定首个平台、目标设备、输入规模、最长边、内存与等待时间预算。
- [ ] 固定核心输入/输出、参数、进度与诊断契约；以 SIFT 和同一套基准样例先验证“导入→拼接→预览→导出”。
- [ ] 在目标设备测量冷启动、峰值内存、耗时、质量和失败恢复，再决定哪些 Python 热点迁移 C++，以及 BA 保留何种目标函数/求解器。
- [ ] 深度推理路径在桌面 deep 通过后评估；不预设 ONNX 或某个平台推理框架必然更快。
- [ ] 之后再评估陀螺仪拍摄引导、重叠提示、拍完即拼及多平台复用。

FastAPI 解耦和 C++ API 注释降低迁移成本，算法数值、依赖、内存策略和设备表现仍需逐项验证。避免在首个平台验收前同时铺开 iOS、鸿蒙与 Android。

### 远期探索：独立立项，不进入近期承诺

- [ ] LoFTR / RoMa 等稠密匹配：仅在已有后端的具体失败样例上评估，比较质量收益、资源成本和模型使用条件。
- [ ] 生成式补边：另列生成内容的真实性、原图保留、标识与验收标准，不能替代有效区域和几何正确性。
- [ ] 3D Gaussian Splatting / 可漫游重建：需要不同的采集与验证流程，作为独立 3D 方向，不计入当前旋转全景的完成度。

## 已知边界

- 视差校正是弹性近似，极端近景（<1m 障碍物 + 大幅平移）仍可能残留错位——根治需相机平移建模（超出旋转全景范畴）
- 平面投影视角增大时会明显拉伸；是否发散取决于相机姿态和投影地平线，不能用统一角度阈值保证安全。合成前检查投影地平线、单图 ROI、最终画布及估算内存，超预算缩小或拒绝
- 图对必须同时满足匹配数量、置信度和空间覆盖等条件；SIFT/深度的最低内点数为 20/12，但仅达到数量不等于图对可靠。缺少可靠连接时会丢弃图片或明确失败，弱纹理深度能力本轮尚未验证

## 核心接口和实验模式

示例：`stitch_pro(images, backend_name="sift", max_output_side=1600, memory_mb=768, local_warp=True, quality="full", pair_mode="all", ba_mode="single")`。核心不依赖 FastAPI；真实默认值为 `backend_name="auto"`、`max_output_side=0`，示例显式选择 SIFT 和输出上限。`StitchResult.diagnostics` 报告分量、BA、实际候选数、合成比例/预算、曝光重叠误差、局部场和有效覆盖。

`max_output_side=0` 仍受默认 24MP 画布 / 16MP 单图 / 768MiB 数组估算约束；限制在投影、remap、blender.prepare 前生效。预算估算包含输入、金字塔、掩码、映射和临时副本，不是 OS 硬 RSS 上限；不包括模型缓存、解释器和全部第三方分配。混合尺寸按批次最大面积确定共同工作/接缝比例。中心裁切保持共享焦距；独立缩放、异焦段或偏心裁切需要逐图内参，本轮未支持。

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s tests -v
python -m app.stitcher.cli photos_bench/normal --manifest photos_bench/normal/inputs.json --backend sift --max-output-side 1600 --memory-mb 512 --report output/诊断.json -o output/全景.jpg
python -m app.stitcher.cli photos_bench/normal --backend sift --quality fast -o output/快速.jpg
python tools/verify_changes.py  # 仅合成矩阵，不需要本机私人样例
# 另有固定真实原图清单时：
python tools/verify_changes.py --real-manifest output/verification-20261015/real-inputs.json
```

CLI 自动使用目录内 `inputs.json`；任一引擎失败、解码失败或写入失败返回非零。没有清单的普通照片目录会枚举照片，排除本次输出路径；用户应将其他旧成图移到另一个目录。生成器覆盖自己的五张 `view_XX.jpg` 并重建清单，不把其他文件纳入评测。

- `--quality fast`：关闭局部场、两层 OpenCV 金字塔频带、默认最长边 1600，仍保留曝光补偿和 GraphCut；不是 AutoStitch Demo 的同尺寸双频带实现。
- `--pair-mode ordered`：仅用于已知拍摄顺序，±2 邻居和首尾候选；断连时尝试全对兜底。可能漏掉仍连通组中的约束；真实七图仅召回全对有效边的 8/15，默认仍是 `all`。
- `--ba-mode staged`：实验性的树初始化→全局→较小尺度 soft_l1 精修。没有证明稳定改善，默认仍是 `single`；与 AutoStitch 二维径向 Huber 不等价。
- `--no-local-warp`：用于全局模型对照。局部形变回退会重做两阶段，避免接缝和融合使用不同几何。

基线通过 OpenCV 的估计/合成分离接口预先限制合成；其无效掩码仍由颜色近似估计，真实黑色物体不能与支持外区域严格区分。自动裁剪返回建议框，原全景仍保留完整有效内容。

## GitHub Actions 与安装包

推送 `main` / `codex/**` 分支、提交 PR 或手动 Run workflow 时运行 [构建工作流](.github/workflows/build.yml)。先在 Ubuntu、Windows 的 Python 3.12 上运行反例测试、普通/低重叠/混合尺寸 SIFT 回归及 ASGI/SSE 检查，通过后构建 Windows x64 安装包。

截至 2026-10-15，[提交 0e4a0b3 的运行](https://github.com/kergee/pano-stitcher/actions/runs/38666481283) 全部成功：两个核心/API 任务通过，Windows NSIS EXE / MSI 构建、打包后端启动及产物上传通过。

| Artifacts 名称 | 内容 | 保留时间 |
|---|---|---|
| `Pano-Windows-x64-SIFT` | Windows 安装包；本次归档约 196MiB，默认不含 torch/kornia 和深度权重 | 14 天 |
| `verification-ubuntu-latest` | Ubuntu 合成清单、几何/接口检查和解析依赖版本 | 14 天 |
| `verification-windows-latest` | Windows 对应核验记录 | 14 天 |

进入运行详情的 **Artifacts** 下载，GitHub 可能要求登录；过期后需要重新运行构建。当前不自动发布 GitHub Release，安装包未做代码签名。打包后端启动检查只验证进程可启动和首页 HTTP 200，安装、完整 WebView 交互、退出生命周期和真实照片质量仍按 Roadmap 分别验收。
