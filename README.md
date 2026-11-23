# GlyphStack

本地文档识别与质量复核平台。终端后台调度三个 OCR 模型与 Jev 评分器，浏览器呈现原图、模型差异和判定依据。支持 PDF、PNG、JPEG 和递归目录。

## 快速开始

从仓库根目录运行。首次准备 GPU 环境、模型和 Jev 渲染依赖，见[单卡环境安装](docs/advanced.md#环境与模型)。

```bash
conda activate "$PWD/runs/conda"
python -m pip install -e '.[quality]'
glyphstack console
```

把待处理的 PDF 或图片放进 `data/inbox/`，也可以在[统一配置](configs/import.yaml)中修改 `input_path` 后重启后台。在其他已激活同一环境的终端中运行：

```bash
glyphstack run --port 8520                 # 提交配置中的输入，等待并打印批次目录
glyphstack run /data/invoice.pdf --port 8520  # 临时指定文件或目录
glyphstack studio --port 8520              # 启动前端，自动读取后台的结果目录和前端端口
```

在[GlyphStack 复核平台](http://127.0.0.1:8517)点 **导入结果**，选择已完成的批次。上方「模型后台」可查看服务状态和当前任务。

## 部署自检

仓库附带两个合成输入：[文字图片](examples/inputs/inspection.png)含批号和小数，[单页 PDF](examples/inputs/measurements.pdf)含一张 13 行 × 10 列的表格。

```bash
rsi-ocr run examples/inputs --import-only  # 无需 GPU，检查图片导入与 PDF 渲染
glyphstack run examples/inputs --port 8520  # 后台启动后，提交同一组文件做完整推理
```

导入后，批次的 `input/pages.jsonl` 应有两条记录。完整推理结束后，在复核平台导入该批次，核对图片中的 `12.50 mm` 和 PDF 首行数据 `GS-001 / 12.20`。在难例复核首页的「显示模型」中切换单模型和多模型，检查表格宽度。可直接横向和纵向滚动表格，多模型对照时会同步滚动。桌面端可拖动左侧队列边界或点击箭头收起队列。勾选「只看差异格」，即可按行列坐标并排核对各模型的值、缺失和合并情况。

## 模型控制台

顶部模型卡片显示健康状态、GPU、显存上限和服务端口。较高的终端还会显示最近 60 秒的请求活动。操作栏下方显示任务队列和实时 POST。按 **1–4** 选择模型，**S** 启动、**X** 停止，**R** 提交配置中的输入。在前端 **流程与设置 → 模型部署** 选择模式并保存，也可编辑 YAML 的 `deployment.mode`。

| 模式 | 服务生命周期 |
|---|---|
| `on_demand` 串行按需（默认） | 每个批次按阶段逐个启动模型，切换时释放上一模型，批次结束释放显存。 |
| `managed` 后台并行常驻 | 后台启动或保存常驻设置时自动并行加载已启用的模型。批次复用常驻服务，OXR 与 Ovis 可并行转录。 |

两种模式都用 `glyphstack run <输入路径> --port 8520` 提交任务。常驻模型正在加载时，任务会排队等待。工作台顶部「模型后台」显示部署模式、就绪数量、各模型的 GPU 和启动错误。常驻模式需按可用显存分配各模型的 `gpu` 和 `gpu_memory`。

```bash
glyphstack console --headless              # 无交互终端时运行后台
glyphstack monitor --port 8520             # 连接运行中的后台，打开彩色交互面板
glyphstack status --port 8520              # 输出模型和任务状态 JSON
glyphstack start all --port 8520           # 手动并行启动常驻模式下尚未就绪的模型
glyphstack start ovis --port 8520          # 命令行加载模型
glyphstack restart ovis --port 8520        # 重读配置，重启指定模型
glyphstack restart all --port 8520         # 重启当前在线的模型
glyphstack stop ovis --port 8520           # 命令行卸载模型
```

按 **T** 重启选中卡片的模型，按 **P** 展开或收起实时 POST 列表。请求到达时显示 LIVE，完成后显示 HTTP 状态码和耗时；直接调用托管模型端口的请求也可见。列表保留当前后台会话最近 128 条请求的元数据。Jev 的评分入口和底层引擎分别标为 `jev`、`jev/engine`。

## 配置

日常部署只需编辑 [configs/import.yaml](configs/import.yaml)。也可以用 `glyphstack console --config configs/team.yaml` 指定另一份配置，客户端只需知道该后台端口。

| 参数 | 用途 |
|---|---|
| `manager_port`、`port` | 终端后台端口与前端端口，默认分别为 8520、8517。 |
| `deployment` | 运行方式、权重根目录 `model_root`、Jev 源码目录 `minicpm_repo`、版面检测权重和默认 GPU 设置。 |
| `deployment.models` | 各模型的启用状态 `enabled`、权重目录 `path`、GPU、服务端口和上下文长度，以及 Jev 渲染端口。 |
| `input_path`、`output_root` | 默认输入文件或目录、结果批次根目录。复核时保留批次内的图片和裁图。 |
| `extensions`、`recursive`、`dpi` | 输入类型、递归开关、PDF 渲染分辨率。 |
| `mining` | 难例触发阈值、排序权重、抽检比例和入队配额。 |

在 **流程与设置 → 模型部署** 可分别修改权重路径、GPU 和启用状态，保存后写回后台使用的 YAML。PaddleOCR-VL 必须启用，OXR、OvisOCR3 和 Jev 可独立关闭。权重路径可以是绝对路径，也可以相对于 `deployment.model_root`，GPU 留空则继承部署默认值。关闭 Jev 后批次仍会筛选难例并生成复核结果。

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
- **选结果**。三个模型的转录同时展示，可切换差异基准，无需人工参考标注。表格支持按行列坐标跳转，分块查看合并单元格，并逐处对照缺失、内容和结构差异。展开「判定明细与计算过程」可查看实测值、阈值和排序分。
- **做决定**。选择模型后点 **Pass** 保存其转录，**Not pass** 驳回。筛选「非难例」可批量通过当前筛选下尚未审核的有效 Paddle 结果，支持撤销。
- **导出**。审核决定写入 `production/review.sqlite`。顶部「导出」生成带 UTC 时间戳的 SFT JSONL，保存在 `production/sft_exports/`。图片路径指向批次裁图，表格答案采用 OTSL。[PaddleOCR‑VL 官方数据格式](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft_zh.md)

已有 OmniDocBench 批次、区域清单或独立模型服务，可参阅[基准评测和分步命令](docs/advanced.md)。
