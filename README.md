# GlyphStack

本地文档识别与质量复核平台。终端后台调度 PaddleOCR-VL、OXR、OvisOCR3 和 Jev，浏览器中可对照原图和识别结果、审核难例，并导出训练数据。支持 PDF、PNG、JPEG 和递归目录。

[安装](#安装) · [单卡部署](#单卡部署) · [日常使用](#日常使用) · [故障排查](#故障排查) · [复核与导出](#复核与导出)

## 安装

以下命令以 Ubuntu 24.04（Linux 或 WSL2）、NVIDIA GPU 和 Conda 为例。先用 `nvidia-smi` 确认驱动能识别 GPU。所有 GlyphStack 命令从仓库根目录执行。

### 1. 创建环境

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3-dev nodejs fontconfig fonts-noto-cjk
git clone https://github.com/yogurtss/rsi_ocr.git
cd rsi_ocr
conda create -p "$PWD/runs/conda" -c conda-forge python=3.12 pip
conda activate "$PWD/runs/conda"
python -m pip install -e '.[quality,jev]' \
  'vllm==0.25.1' 'transformers==5.13.1' \
  'paddlepaddle==3.3.1' 'paddleocr[doc-parser]==3.7.0' huggingface_hub
PLAYWRIGHT_BROWSERS_PATH="$PWD/runs/playwright" python -m playwright install --with-deps chromium
```

**Jev SDK 随 GlyphStack 源码和 Python 包一起分发。** 默认配置直接使用内置 SDK，安装时只需克隆上面的 GlyphStack 仓库。模型权重单独下载，Chromium 和字体由本节命令安装。[内置 SDK 来源和许可证](rsi_ocr/_vendor/jev/README.md)

### 2. GPU 版面检测

上一步安装的 `paddlepaddle` 用于默认的 CPU 版面检测。如果要在「模型部署 → 版面检测」选择 GPU，请在运行 Console 的同一个 Conda 环境中把 CPU 版替换成支持 CUDA 的 `paddlepaddle-gpu`。下面的安装命令以 CUDA 12.6 为例。[PaddlePaddle GPU 安装说明](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)

```bash
conda activate "$PWD/runs/conda"
python -m pip uninstall -y paddlepaddle
python -m pip install 'paddlepaddle-gpu==3.3.1' \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

**Python 3.12 + CUDA 13.0** 的 PaddlePaddle GPU 3.3.1 官方 wheel：

- Linux / WSL2（x86_64）：[paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl](https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl)
- 原生 Windows（x64）：[paddlepaddle_gpu-3.3.1-cp312-cp312-win_amd64.whl](https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-win_amd64.whl)

使用直链时，卸载 CPU 版后，在同一环境运行 `python -m pip install <下载的 .whl 路径>`。安装完成后重启已运行的 Console。继续使用 CPU 版面检测时可跳过本节。

### 3. 下载模型

已有权重时，可在配置里指向现有目录，并安装已有 OXR 目录中的插件。首次下载使用下列命令。OXR 和 OvisOCR3 需要完整模型目录，分别包含 vLLM 插件和 OCR 提示文件。

```bash
hf download PaddlePaddle/PaddleOCR-VL-1.6 --local-dir runs/model-cache/paddleocr-vl-16
hf download PaddlePaddle/PP-DocLayoutV3 --local-dir runs/model-cache/PP-DocLayoutV3
hf download SoMarkAI/OXR-1.0 --local-dir runs/model-cache/OXR-1.0
hf download ATH-MaaS/OvisOCR3 --local-dir runs/model-cache/OvisOCR3
hf download openbmb/MiniCPM-OCR-Jev --local-dir runs/model-cache/MiniCPM-OCR-Jev
python -m pip install ./runs/model-cache/OXR-1.0/plugins/vllm
python -m pip check
```

## 单卡部署

首次部署时复制配置，并创建默认输入目录。

```bash
mkdir -p runs data/inbox
cp configs/import.yaml runs/server.yaml
```

编辑 `runs/server.yaml` 中现有的 `deployment.model_root`。使用上述下载命令时将其设为 `runs/model-cache`，已有权重时填写实际根目录。各模型的 `path` 可使用相对于根目录的目录名或绝对路径。保持 `deployment.mode: on_demand` 即可在单卡上依次加载模型。已有显存上限时，在 `deployment.gpu_memory` 填写预算，例如 `0.60`。

**终端 1** 启动后台。模型在处理任务时按阶段加载，空闲时可以显示离线。

```bash
glyphstack console --config runs/server.yaml
```

**终端 2** 激活相同的环境并跑样例。可先执行第一行，只验证 PDF 和图片导入。第二行会提交完整推理并在完成后打印批次目录。第三行启动浏览器复核工作台。

```bash
rsi-ocr run examples/inputs --settings runs/server.yaml --import-only
glyphstack run examples/inputs --port 8520
glyphstack studio --port 8520
```

导入自检应在打印出的 `input/pages.jsonl` 中生成两条记录。完整推理完成后，打开 [复核工作台](http://127.0.0.1:8517)，点 **导入结果** 选择该批次，在批次概览中查看处理结果。样例包含一张[文字图片](examples/inputs/inspection.png)和一份[单页表格 PDF](examples/inputs/measurements.pdf)，可核对图片上的 `12.50 mm` 和表格首行的 `GS-001 / 12.20`。

后台和工作台默认监听服务器的本地回环地址。若在本机浏览器访问远程服务器，从本机建立 SSH 转发，再打开同一工作台地址。

```bash
ssh -N -L 8517:127.0.0.1:8517 user@server
```

## 日常使用

```bash
glyphstack run --port 8520                    # 处理配置中的 input_path，默认为 data/inbox
glyphstack run /data/invoice.pdf --port 8520  # 处理指定文件或整个目录
glyphstack status --port 8520                 # 查看模型和任务状态
```

同一个未完成批次可用相同输入路径再次提交。Console 的 **R** 会提交配置中的默认输入。复核时保留批次内的原页图片、裁图和结果文件。

## 故障排查

| 现象 | 检查方法 |
|---|---|
| 无法连接后台 | 确认终端 1 仍在运行，`--port` 与配置中的 `manager_port` 一致。 |
| 提示权重缺失或模型加载失败 | 核对 `deployment.model_root`、各模型的 `path` 和 GPU 编号。用 `glyphstack status --port 8520` 查找任务的输出目录，再看其中对应模型的日志。手动启动模型的日志在 `<output_root>/.manager/`。 |
| OXR 无法注册或 Ovis 缺提示文件 | 确认已安装 `OXR-1.0/plugins/vllm`，并确认 Ovis 权重目录中存在 `prompts/plain_ocr.txt`。 |
| 提示「GPU 版面检测需要支持 CUDA 的 paddlepaddle-gpu」 | 按 [GPU 版面检测](#2-gpu-版面检测) 在 Console 使用的 Conda 环境里替换 CPU 版，然后重启 Console。 |
| Jev 找不到 `sdk/app.py` | 新版默认从安装包加载 SDK。确认 Console 使用的是新版，再检查自用 YAML 中的 `deployment.minicpm_repo`。配置中删掉该覆盖项即可使用内置 SDK。 |
| Jev 引擎启动后渲染仍失败 | 看 `jev-render.log`，检查 Node.js、Chromium 的安装路径、系统库和 CJK 字体。 |
| Jev 报 `No available memory for the cache blocks` | 看 `jev.log`。可先在 `deployment.models.jev` 中设置 `max_sequences: 4` 以降低启动时的显存需求，然后重启 Console。 |
| CUDA 显存不足 | 保持串行按需模式，核对卡上其他进程、`gpu_memory` 和模型的 `max_sequences`。 |

## 模型控制台

顶部模型卡片显示健康状态、GPU、显存上限和服务端口。版面检测卡片显示权重路径和设备，例如 `CPU` 或 `GPU 0`。较高的终端还会显示最近 60 秒的请求活动。操作栏下方显示任务队列和实时 POST。按 **1–4** 选择模型，**S** 启动、**X** 停止，**R** 提交配置中的输入。在前端 **流程与设置 → 模型部署** 选择模式并保存，也可编辑 YAML 的 `deployment.mode`。

底部的运行日志自动滚动，显示带时间的模型加载结果、文件夹批次进度、PDF 渲染与 OCR 页码，以及任务完成或失败信息。常规视图会给日志留出更多空间，按 **P** 可以展开请求列表。`glyphstack monitor` 连接后台后也会显示这些日志。

按 **A** 可提交卸载全部模型的操作。后台会先处理已提交的任务，再卸载模型，控制台保持运行。模型卡片显示离线后即可按 **Ctrl+C** 退出。也可以直接按 **Ctrl+C** 中止批次并开始释放模型，日志窗口会显示关闭进度。

模型卡片的「运行 3/8　等 1」表示 vLLM 正在处理 3 个请求，上限为 8，还有 1 个请求排队。终端较高时也会显示 KV 缓存占用。服务没有返回调度指标时，卡片显示进行中的 POST 数。

每个模型的默认并发上限为 6。按 **1–4** 选中模型，再按 **C**、输入新的并发上限并按 **Enter** 提交，按 **Esc** 取消输入。设置会排在已有任务之后，保存到当前 YAML，并重启已加载的选中模型。教师转录和 Jev 评分会按新上限发出请求。Paddle 的 PDF 版面识别仍逐页执行。

| 模式 | 服务生命周期 |
|---|---|
| `on_demand` 串行按需（默认） | 每个批次按阶段逐个启动模型，切换时释放上一模型，批次结束释放显存。 |
| `managed` 后台并行常驻 | 后台启动或保存常驻设置时自动并行加载已启用的模型。批次复用常驻服务，OXR 与 Ovis 可并行转录。 |

两种模式都用 `glyphstack run <输入路径> --port 8520` 提交任务。常驻模型正在加载时，任务会排队等待。工作台顶部「模型后台」显示部署模式、就绪数量、各模型的 GPU 和启动错误。常驻模式需按可用显存分配各模型的 `gpu` 和 `gpu_memory`。

任务中断后，用相同输入路径再次提交即可接着跑。按 **R** 也可以续跑配置中的默认输入。后台沿用未完成批次的时间戳目录，跳过已写入的页图，并复用有效的 OCR 和评分缓存。修改输入文件集合、文件内容或渲染 DPI，或者上次已生成 `production/summary.json`，都会新建批次。没有续跑记录的旧批次可以用 `rsi-ocr run <原输入路径> --out <旧批次目录>` 手动复用已有 OCR 缓存。

```bash
glyphstack console --headless              # 无交互终端时运行后台
glyphstack monitor --port 8520             # 连接运行中的后台，打开彩色交互面板
glyphstack status --port 8520              # 输出模型和任务状态 JSON
glyphstack start all --port 8520           # 手动并行启动常驻模式下尚未就绪的模型
glyphstack start ovis --port 8520          # 命令行加载模型
glyphstack restart ovis --port 8520        # 重读配置，重启指定模型
glyphstack restart all --port 8520         # 重启当前在线的模型
glyphstack stop ovis --port 8520           # 命令行卸载模型
glyphstack stop all --port 8520            # 命令行卸载全部模型
```

按 **T** 重启选中卡片的模型，按 **P** 展开或收起实时 POST 列表。进行中的请求显示 `LIVE`，成功结束时显示 `DONE` 和 HTTP 状态码，失败时显示 `ERR` 和状态码，断开时显示「中断」。列表显示耗时，并保留当前后台会话最近 128 条请求的元数据，包括直接发往托管模型端口的请求。Jev 的评分入口和底层引擎分别标为 `jev`、`jev/engine`。

## 配置

日常部署只需编辑 [configs/import.yaml](configs/import.yaml)。也可以用 `glyphstack console --config configs/team.yaml` 指定另一份配置，客户端只需知道该后台端口。

| 参数 | 用途 |
|---|---|
| `manager_port`、`port` | 终端后台端口与前端端口，默认分别为 8520、8517。 |
| `deployment` | 运行方式、权重根目录 `model_root`、可选的 Jev 外部源码目录 `minicpm_repo`、默认 GPU 设置。 |
| `deployment.models` | 各模型的启用状态、权重路径和部署参数。`layout` 配置版面检测的权重路径和设备。 |
| `input_path`、`output_root` | 默认输入文件或目录、结果批次根目录。复核时保留批次内的图片和裁图。 |
| `extensions`、`recursive`、`dpi` | 输入类型、递归开关、PDF 渲染分辨率。 |
| `mining` | 难例触发阈值、排序权重、抽检比例和入队配额。 |

在 **流程与设置 → 模型部署** 可分别修改权重路径、GPU 和启用状态，保存后写回后台使用的 YAML。PaddleOCR-VL 必须启用，OXR、OvisOCR3 和 Jev 可独立关闭。权重路径可以是绝对路径，也可以相对于 `deployment.model_root`，GPU 留空则继承部署默认值。关闭 Jev 后批次仍会筛选难例并生成复核结果。

在「版面检测」模型行可以设置权重路径和设备，默认使用 `PP-DocLayoutV3` 和 CPU。选择 GPU 后填写设备编号，例如 `0` 或 `1`，保存后用于后续批次。YAML 中对应 `deployment.models.layout: {path: PP-DocLayoutV3, device: gpu:0}`。编号从 0 开始，按 Console 环境可见的 GPU 顺序计算，可与 PaddleOCR-VL 识别服务的 GPU 分开指定。单卡部署时，GPU 版面检测和 PaddleOCR-VL 识别共用这张卡的显存。

保留原来的 Conda 环境，并按 [GPU 版面检测](#2-gpu-版面检测) 安装 CUDA 版 PaddlePaddle。

例如将 `deployment.models.ovis` 设为 `{enabled: true, path: OvisOCR3, port: 8102, context: 16384, gpu: "1", gpu_memory: 0.75, max_sequences: 4}`，Ovis 就会部署到 GPU 1。跨卡模型可同时设置 `gpu: "2,3"` 和 `tensor_parallel: 2`。在页面保存时，受变更影响的服务会卸载，常驻模式会按新配置启动。切换到串行按需模式会卸载常驻模型。直接编辑 YAML 后，可选中对应卡片按 **T** 重读该模型的配置，启用的模型会重启。

## 后台数据怎么产生

| 步骤 | 批次内的产物 |
|---|---|
| PDF 逐页渲染，图片直接复制；记录相对文件名和页号 | `input/pages.jsonl`、`input/pages/` |
| Paddle 做版面检测并识别，保存实际送入识别器的裁图 | `capture/regions.jsonl`、`capture/crops/` |
| OXR、OvisOCR3 各自识别同一裁图，Jev 对每份有效转录评分。按需模式轮换占用 GPU，常驻模式并行运行两个教师模型 | `production/evidence.jsonl` 保存三模型结果、分数和判定信号 |
| 规则标记难例，按排序分、配额和随机抽检形成审核队列 | `production/review.jsonl`、`production/summary.json` |

只有页图和清单需求时可用 `rsi-ocr run /data/inbox --import-only`。如需排查推理失败，查看批次根目录中对应模型的 `.log`。

## 调整难例阈值

打开顶部 **流程与设置**，点选流程图里的质量分、候选增益或模型分歧节点。拖动滑块即可看到当前批次的难例数、文字与表格分布，以及包含随机抽检的预计入队数。同一条难例可能命中多个规则。

数字冲突、空转录、格式无效、Jev 预检拒绝各有独立开关和解释，切换后会同步试算。命令行新批次读取 `mining.fixed_rules`。

点 **应用到本批次** 更新质控台的筛选与复核队列。设置保存在该批次的 `review.sqlite`，直接复用已保存的 OCR 和 Jev 结果。**恢复运行时阈值** 可预览初始策略，再点应用即可恢复。后续命令行批次的默认值由 `configs/import.yaml` 管理。

## 复核与导出

- **看原图**。点击裁图或原页，滚轮缩放、拖动平移。原页框标出当前区域，可手动补框。
- **选结果**。三个模型的转录同时展示，可切换差异基准，无需人工参考标注。在「显示模型」中选择要对照的模型，表格会同步滚动。可按行列坐标跳转，勾选「只看差异格」查看缺失、内容和结构差异。展开「判定明细与计算过程」可查看实测值、阈值和排序分。桌面端可拖动左侧队列边界或点击箭头收起队列。
- **做决定**。选择模型后点 **Pass** 保存其转录，**Not pass** 驳回。筛选「非难例」可批量通过当前筛选下尚未审核的有效 Paddle 结果，支持撤销。
- **导出**。审核决定写入 `production/review.sqlite`。顶部「导出」生成带 UTC 时间戳的 SFT JSONL，保存在 `production/sft_exports/`。图片路径指向批次裁图，表格答案采用 OTSL。[PaddleOCR‑VL 官方数据格式](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft_zh.md)

已有 OmniDocBench 批次、区域清单或独立模型服务，可参阅[基准评测和分步命令](docs/advanced.md)。
