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

导入自检应在打印出的 `input/pages.jsonl` 中生成两条记录。完整推理完成后，打开 [复核工作台](http://127.0.0.1:8517)，在导入页通过首页缩略图辨认批次，也可搜索首份文件名或批次名。点批次卡片即可导入并复核。也可以点右上角的 **导入结果**，粘贴服务器上已完成的批次路径（如 `/data/runs/批次名` 或 `/data/runs/批次名/production`），按回车或点 **加载** 进入复核。相对路径以后台配置的 `output_root` 为起点。只有 `--import-only` 生成的页图时，需先用 `glyphstack run` 完成推理，批次才会出现在列表中。样例包含一张[文字图片](examples/inputs/inspection.png)和一份[单页表格 PDF](examples/inputs/measurements.pdf)，可核对图片上的 `12.50 mm` 和表格首行的 `GS-001 / 12.20`。

后台和工作台默认监听 `127.0.0.1`。要从可信局域网直接访问，在服务器上用下面的命令启动 Studio，再从浏览器打开 `http://服务器IP:8517`。

```bash
glyphstack studio --host 0.0.0.0 --port 8520
```

`0.0.0.0` 监听所有 IPv4 网卡，也可以指定服务器某个网卡的 IP。`--port 8520` 指定 Studio 要连接的模型后台，浏览器使用配置文件中的 `port`（默认 `8517`）。Studio 没有登录鉴权，能访问该端口的用户也能修改复核结果和模型部署设置。

也可以使用默认监听地址，在本机建立 SSH 转发后打开 `http://127.0.0.1:8517`。

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

顶部模型卡片显示健康状态、GPU、显存上限和服务端口。版面检测卡片显示配置设备和 Paddle 检测器实际选择的设备。配置 GPU 却使用 CPU 时会标出「回退」；整批命中页缓存时显示「整批复用 · 已跳过」，无法读取检测器设备时显示「设备未核验」。较高的终端还会显示最近 60 秒的请求活动。操作栏下方显示任务队列和实时 POST。按 **1–4** 选择模型，**S** 启动、**X** 停止，**R** 提交配置中的输入。在前端 **设置 → 模型部署** 选择模式并保存，也可编辑 YAML 的 `deployment.mode`。

Paddle 进度行显示复用页数、本次新完成页数、剩余页数和当前文件页码。版面与 OCR 同时运行时分别显示正在处理的页。上一完成页的「版面」是检测器耗时，「vLLM 请求」是至少一个识别请求在途的时间，重叠请求只计一次。单页耗时还包含裁图和结果处理。

运行日志顶部固定显示批次和当前文件的进度条。按 **G** 查看每个 PDF 的总进度，以及页图、Paddle、OXR、Ovis、Jev 各自的已处理数和总数。用 **[** 和 **]** 翻页，再按 **G** 返回。关闭的模型会从进度表中移除，关闭 Jev 时末阶段显示「筛选」。页图和 Paddle 按页计数，教师和评分按区域计数，批次与文件总进度按启用的阶段等权计算。缓存命中计入已处理，失败的区域也计入已处理并标记错误。最终结果写入期间批次总进度最多显示 99%。

底部的运行日志自动滚动，显示带时间的模型加载结果、文件夹批次进度、PDF 渲染与 OCR 页码，以及任务完成或失败信息。常规视图会给日志留出更多空间，按 **P** 可以展开请求列表。`glyphstack monitor` 连接后台后也会显示这些日志。

`glyphstack console` 和 `glyphstack monitor` 的 vLLM 日志窗口默认自动跟随模型。按需加载时，窗口随任务切换到 PaddleOCR-VL、OXR、OvisOCR3 或 Jev；多个模型常驻时，窗口按模型名分组显示各引擎的最新输出。全部卸载后保留最近启动的模型日志。按 **1–4** 选中模型卡片并固定查看该模型，按 **0** 恢复自动跟随。Jev 显示其 vLLM 引擎日志。按 **L** 开关窗口，宽终端默认打开。

按 **A** 可提交卸载全部模型的操作。后台会先处理已提交的任务，再卸载模型，控制台保持运行。模型卡片显示离线后即可按 **Ctrl+C** 退出。也可以直接按 **Ctrl+C** 中止批次并开始释放模型，日志窗口会显示关闭进度。

模型卡片的「运行 3/8　等 1」表示 vLLM 正在处理 3 个请求，上限为 8，还有 1 个请求排队。终端较高时也会显示 KV 缓存占用。服务没有返回调度指标时，卡片显示进行中的 POST 数。

每个模型的默认并发上限为 6。按 **1–4** 选中模型，再按 **C**、输入新的并发上限并按 **Enter** 提交，按 **Esc** 取消输入。设置会排在已有任务之后，保存到当前 YAML，并重启已加载的选中模型。Paddle 的客户端参数 `vl_rec_max_concurrency` 也使用该并发上限。默认的 PP-DocLayoutV3 路径采用双页流水线。vLLM 识别当前页的区域时，同一个检测器可以处理下一页。每页独立绑定裁图与原始响应，区域清单按输入页序输出。开启文档方向校正、图像去畸变，或使用不支持版面预取的后端时，Console 会显示「逐页」。[PaddleOCR-VL 官方用法与性能调优](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html#33-performance-tuning)

| 模式 | 服务生命周期 |
|---|---|
| `on_demand` 串行按需（默认） | 每个批次按阶段逐个启动模型，切换时释放上一模型，批次结束释放显存。 |
| `managed` 后台并行常驻 | 后台启动或保存常驻设置时自动并行加载已启用的模型。批次复用常驻服务，OXR 与 Ovis 可并行转录。 |

两种模式都用 `glyphstack run <输入路径> --port 8520` 提交任务。常驻模型正在加载时，任务会排队等待。工作台顶部「模型后台」显示部署模式、就绪数量、各模型的 GPU 和启动错误。常驻模式需按可用显存分配各模型的 `gpu` 和 `gpu_memory`。

任务中断后，用相同输入路径再次提交即可接着跑。按 **R** 也可以续跑配置中的默认输入。保持输入、DPI 和输出根目录相同，后台和不带 `--out` 的 `rsi-ocr run` 都会找到未完成的原批次。启动模型前日志会给出可复用页数与首个待处理页。所有页均已完成时，按需模式直接跳过 Paddle 加载。新输入先读取文档页数，首张页图就绪即可推理；单个后台线程按需预取后续页图，按原 DPI 无损保存 PNG。

OXR、Ovis 的区域转录和 Jev 的区域评分会逐请求保存到 `production/cache.sqlite`。按需模式续跑时，遇到首个缺少结果的区域才加载对应模型；日志显示文件、页码、区域编号，以及本阶段复用、本次完成和错误的请求数。同输入的新批次也会从上个完整批次复用匹配结果。缓存按模型、revision、服务地址和请求内容匹配；原地替换权重时应更新 revision。最终复核队列按本次筛选规则重新汇总。

整页断点包含版面与基线 OCR 结果。使用随安装说明提供的 PP-DocLayoutV3 检测器时，版面框会在该页发起区域识别前单独落盘。中断在 OCR 期间，下次可复用该页版面并重新识别该页区域。旧版本保留的整页缓存仍可读取。输入文件集合、内容或 DPI 改变时新建批次。完整批次再次提交也会产生新目录，并从同输入的上个完整批次复用匹配的整页结果与裁图。没有续跑记录的旧批次可以用 `rsi-ocr run <原输入路径> --out <旧批次目录>` 手动复用已有 OCR 缓存。

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

在 **设置 → 模型部署** 可分别修改权重路径、GPU 和启用状态，保存后写回后台使用的 YAML。PaddleOCR-VL 必须启用，OXR、OvisOCR3 和 Jev 可独立关闭。权重路径可以是绝对路径，也可以相对于 `deployment.model_root`，GPU 留空则继承部署默认值。关闭 Jev 后批次仍会筛选难例并生成复核结果。

在「版面检测」模型行可以设置权重路径和设备，默认使用 `PP-DocLayoutV3` 和 CPU。选择 GPU 后填写设备编号，例如 `0` 或 `1`，保存后用于后续批次。YAML 中对应 `deployment.models.layout: {path: PP-DocLayoutV3, device: gpu:0}`。编号从 0 开始，按 Console 环境可见的 GPU 顺序计算，可与 PaddleOCR-VL 识别服务的 GPU 分开指定。单卡部署时，GPU 版面检测和 PaddleOCR-VL 识别共用这张卡的显存。

保留原来的 Conda 环境，并按 [GPU 版面检测](#2-gpu-版面检测) 安装 CUDA 版 PaddlePaddle。

例如将 `deployment.models.ovis` 设为 `{enabled: true, path: OvisOCR3, port: 8102, context: 16384, gpu: "1", gpu_memory: 0.75, max_sequences: 4}`，Ovis 就会部署到 GPU 1。跨卡模型可同时设置 `gpu: "2,3"` 和 `tensor_parallel: 2`。在页面保存时，受变更影响的服务会卸载，常驻模式会按新配置启动。切换到串行按需模式会卸载常驻模型。直接编辑 YAML 后，可选中对应卡片按 **T** 重读该模型的配置，启用的模型会重启。

## 后台数据怎么产生

| 步骤 | 批次内的产物 |
|---|---|
| PDF 逐页渲染，图片直接复制；记录相对文件名和页号 | `input/pages.jsonl`、`input/pages/` |
| Paddle 做版面检测并识别，保存实际送入识别器的裁图 | `capture/regions.jsonl`、`capture/crops/` |
| OXR、OvisOCR3 各自识别同一裁图，Jev 对每份有效转录评分。按需模式轮换占用 GPU，常驻模式并行运行两个教师模型 | `production/evidence.jsonl` 保存三模型结果、分数和判定信号 |
| 每个区域取 Jev 成功评分的最高分转录，覆盖全部已处理区域 | `production/selected.jsonl` 每行记录文档、页、区域、采用模型、转录、得分、与下一份不同转录的分差及状态 |
| 规则标记难例，按排序分、配额和随机抽检形成审核队列 | `production/review.jsonl`、`production/summary.json` |

只有页图和清单需求时可用 `rsi-ocr run /data/inbox --import-only`。如需排查推理失败，查看批次根目录中对应模型的 `.log`。

## 调整难例阈值

打开顶部 **设置 → 筛选规则与阈值**，点选流程图里的质量分、候选增益或模型分歧节点。拖动滑块即可看到当前批次的难例数、文字与表格分布，以及包含随机抽检的预计入队数。同一条难例可能命中多个规则。

数字冲突、空转录、格式无效、Jev 预检拒绝各有独立开关和解释，切换后会同步试算。命令行新批次读取 `mining.fixed_rules`。

点 **应用到本批次** 更新质控台的筛选与复核队列。设置保存在该批次的 `review.sqlite`，直接复用已保存的 OCR 和 Jev 结果。**恢复运行时阈值** 可预览初始策略，再点应用即可恢复。后续命令行批次的默认值由 `configs/import.yaml` 管理。

## 复核与导出

日常批次按无 GT（人工参考标注）运行。区域择优使用 OCR 候选和 Jev 评分，难例入队使用模型分歧、风险规则、配额和抽检。提供 GT 时，Studio 在结果生成后单独对照参考标注，自动选择、队列和导出结果均沿用原批次。

有部分或全部区域的 GT 时，可在批次的 `input/references.jsonl` 中按区域写入 `doc_id`、`page_id`、`region_id`、`task`、`image_sha256` 和 `reference`（文字转录或表格 HTML），然后重新导入批次。前五项可从 `production/selected.jsonl` 复制。复核页显示自动所选模型和最接近 GT 的模型，文字按字符错误率 CER 取最低，表格按 TEDS 取最高，同分并列。总览显示有 GT 的区域数，以及至少两个有效候选且有自动选择的区域中有多少命中最近候选。

- **看结果**。文字和表格默认显示整页原图、区域裁图和选中的转录，点「下一条」即可连续浏览。点击图片可放大、拖动，原页框标出当前区域。打开有人工记录的区域时会恢复上次复核所选模型。
- **看择优依据**。相同转录只占一个名次。同分优先保留 Paddle，再按教师配置顺序选择。最高分低于 `mining.proposal_judge`（默认 0.90）、不同转录分差低于 `mining.selection_margin`（默认 0.05）、并列最高或部分候选评分失败时，保留最高分转录并标记待核对。全部评分失败或预检拒绝时，转录留空。关闭 Jev 时按顺序保留首份有效转录并标记待核对。可用「样本范围 → 择优待核对」集中查看。
- **按需复核**。展开「各模型转录与判定依据」可对照差异或改选模型。点 **确认通过** 保存人工确认，点 **驳回** 标记错误。展开「筛选条件」并选择「非难例」可批量确认其中通过择优质量筛查的结果，支持撤销。
- **取自动结果**。批次结束即生成 `production/selected.jsonl`，也可从 Studio 顶部「导出 → 区域自动择优结果 JSONL」下载。旧批次直接用已保存的评分生成下载内容。
- **导出**。审核决定写入 `production/review.sqlite`。顶部「导出」生成带 UTC 时间戳的 SFT JSONL，保存在 `production/sft_exports/`。图片路径指向批次裁图，表格答案采用 OTSL。[PaddleOCR‑VL 官方数据格式](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft_zh.md)

已有 OmniDocBench 批次、区域清单或独立模型服务，可参阅[基准评测和分步命令](docs/advanced.md)。
