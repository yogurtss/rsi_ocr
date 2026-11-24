# 模型服务与进阶用法

以下命令从仓库根目录执行。

## 1. 安装、接口和启动

```bash
python -m pip install -e .
cp configs/default.yaml configs/local.yaml
```

`configs/local.yaml` 集中管理模型地址、服务模型名、权重版本、任务 prompt、并发、请求参数、启动命令、挖掘阈值和训练参数。示例采用 YAML 的流式映射语法，也可改成普通块状 YAML。模型的新版本、prompt 或服务实际权重发生变化时，更新 `revision` 和对应配置。API 密钥通过 `api_key_env` 引用环境变量。`prompt_file` 可指向与配置相对的文件，例如 OvisOCR3 权重目录里的 `prompts/plain_ocr.txt`，并优先于内联 `prompt`。默认 Ovis 表格指令面向已裁好的单个表格。

| 模型 | 默认调用地址 | 部署方式 |
|---|---|---|
| PaddleOCR‑VL | `127.0.0.1:8100/v1` | vLLM，OpenAI Chat Completions |
| OXR | `127.0.0.1:8080/v1` | 官方脚本加载 OXR vLLM 插件，仅启动识别服务 |
| OvisOCR3 | `127.0.0.1:8102/v1` | vLLM，关闭 thinking |
| MiniCPM‑OCR‑Jev | `127.0.0.1:18500` | vLLM pooling/classify 后端 `18499` 和官方 render+score 服务 `18500` |
| 内部强 VLM（可选） | `RSI_INTERNAL_VLM_URL` | 已部署的 OpenAI 兼容多模态 API，`strong_budget > 0` 才启用 |

已有接口时，填好 YAML 就可直接使用。在本机启动时，各模型可分别使用官方 Linux 环境，按机器分配 `CUDA_VISIBLE_DEVICES`。Jev 的 render 进程使用其官方渲染环境，包含 Chromium 及表格渲染依赖。

```bash
# 在各自环境和终端运行，命令后加 --show 可查看 startup argv
rsi-ocr --config configs/local.yaml serve paddle
OXR_REPO=/srv/OXR OXR_WEIGHTS=/models/OXR-1.0 rsi-ocr --config configs/local.yaml serve oxr
rsi-ocr --config configs/local.yaml serve ovis
JEV_WEIGHTS=/models/MiniCPM-OCR-Jev rsi-ocr --config configs/local.yaml serve jev
rsi-ocr --config configs/local.yaml serve jev --sidecar render
rsi-ocr --config configs/local.yaml probe
```

默认以原版 `PaddlePaddle/PaddleOCR-VL` 为示例，它对应百度 ERNIEKit 公开 SFT 教程和 Swift 的 `paddle_ocr` 模板。已有 PaddleOCR‑VL‑1.6 产线时，将识别服务权重、`revision`、`capture.pipeline.pipeline_version` 一起改为实际版本。Swift 当前对 1.6 使用 `--model_type paddleocr_vl --template paddle_ocr_1_5`，需要 Transformers 5 及支持该注册项的 Swift 版本。百度公开 SFT 示例使用原版 0.9B，1.6 的 SFT 请将 `training.sft.argv` 换成手头与该 checkpoint 匹配的 ERNIEKit 配方。GRPO 默认原生 Transformers rollout，可在 Swift 和所用 checkpoint 的 rollout 组合验证后，通过 YAML 切换 vLLM。

## 2. 生成区域清单并挖掘

**已有生产 crop** 时可直接准备 `regions.jsonl`。每行对应一张 PNG 或 JPEG crop，路径可相对该清单。`split` 以文档为单位，默认 `mine`。如已有该图的 Paddle 原始输出，可传 `target`，保留 `status`、`raw`、`prompt`、`format`、`model`、`revision` 和生成参数。只提供生成后的 Markdown 或转换过的 HTML 时，建议让工具在原始 crop 上重跑 Paddle。

```json
{"doc_id":"invoice-001","page_id":0,"region_id":"r02","task":"text","image":"crops/r02.png","split":"mine"}
{"doc_id":"invoice-001","page_id":0,"region_id":"r07","task":"table","image":"crops/r07.png","split":"mine"}
```

**只有页面图像** 时，先在 PaddleOCR/PaddleX 环境安装本项目。把 PDF 按生产分辨率转成逐页 PNG/JPEG，以及准备 `pages.jsonl`。

```json
{"doc_id":"invoice-001","page_id":0,"image":"pages/invoice-001-0.png","split":"mine"}
```

```bash
rsi-ocr --config configs/local.yaml capture pages.jsonl --out runs/round01/capture
rsi-ocr --config configs/local.yaml mine runs/round01/capture/regions.jsonl --out runs/round01/mine
```

`capture` 运行现有 PaddleOCR‑VL layout→识别流水线，通过仅监听回环地址的转发器保留真实 vLLM 请求中的图像、任务 prompt、参数和原始响应。采集到的 `region_id` 是页内请求标识。已有 bbox 可在自备清单带入。采集器会记录与原页版面框像素唯一吻合的裁图坐标。采集按页恢复，挖掘结果按图像、请求和模型版本缓存。更换同名线上权重时更新 `revision`，以触发重新采集和评分。

同一 crop 上，教师各自盲读。OCR 按字符编辑距离和数字字段比较。表格将 OTSL 与 HTML 转为单元格及跨行跨列坐标，再比较结构和单元格文本。候选 Jev 打分按区块批量调用；只有 `verdict=scored` 获得有效概率。目标低分、教师分歧、候选得分改善和数字冲突触发审核，模型超时或输出触顶单独记录。默认保留 10% 随机审计，并对难例按任务和文档设配额。建议先用人工核验结果校准默认阈值。

| 文件 | 用途 |
|---|---|
| `evidence.jsonl` | 原始候选、模型与版本、Jev verdict、距离、排序信号和裁图 SHA-256 |
| `review.jsonl` | 优先难例与随机审计队列 |
| `candidate_labels.jsonl` | 机器候选以及支持家族和 Jev 证据，状态为 `unverified` |
| `summary.json` | 挖掘量、服务状态、配置快照和任务分布 |

## 3. 审核后导出并训练

审核后写 `decisions.jsonl`，例如下面的 OCR 样本。表格填写一张经过核验的完整 HTML 表格，或同等 OTSL。训练导出检查裁图 SHA‑256、标签和数据划分。需要导出验证集时，将核验用文档设为 `val`，把 `val` 加到这一轮的 `mining.splits`。相同 crop 不会跨训练、验证集出口。

```json
{"case_id":"<取自 review.jsonl>","image_sha256":"<取自 review.jsonl>","status":"approved","source":"human_verified","label":"合计：-10.50 元"}
```

```bash
rsi-ocr --config configs/local.yaml export runs/round01/mine/evidence.jsonl decisions.jsonl --out runs/round01/data

# 在百度 ERNIEKit 环境使用官方脚本，传 --dry-run 查看命令
ERNIEKIT_REPO=/srv/ERNIE rsi-ocr --config configs/local.yaml train sft \
  --data runs/round01/data --out runs/round01/sft

# 在 ms-swift 4.x 环境安装本项目，--checkpoint 指向 SFT 导出的 HF 权重目录
rsi-ocr --config configs/local.yaml train grpo --data runs/round01/data \
  --checkpoint runs/round01/sft/checkpoint-100 --out runs/round01/grpo
```

- **SFT** 使用 ERNIEKit 原生 `image_info` 和 `text_info`。Paddle 的 `OCR:` 或 `Table Recognition:` 为 `mask`，训练答案为 `no_mask`。表格按目标模型配置导出为 OTSL。通用 replay 可以在训练输入侧按 token 预算与 `sft_train.jsonl` 合并，比例和样本来源应记录在该轮数据清单。
- **GRPO** 使用 ms-swift 图文 `messages`、`images`、`solution` 和 `ocr_task`，通过 `--external_plugins` 装载 `rsi_ocr_region`。目标表格原始格式由 `reward.table_format` 控制，并与 Paddle 的任务协议一致。`--checkpoint` 可传现有 HF 权重目录或模型 ID。
- 接受哪些样本、调用哪些老师、如何调学习率和 rollout 预算，均可在 YAML 中配置。重新开一轮时给 Paddle 新权重、更新 `revision` 并使用新输出目录。评估用按文档或模板隔离的真实样本，分别看 OCR 的 CER/数字准确率，以及表格的 TEDS、TEDS-S 和单元格准确率。对照同一通用集追踪 SFT 前后退化。

## 4. GRPO 奖励

| 任务 | 默认奖励 | 细节 |
|---|---|---|
| OCR | `0.85 × 字符相似度 + 0.15 × 数字序列精确一致` | 字符相似度为 `1 − 编辑距离 / max(预测长度, 参考长度, 1)`。统一 NFC 及空白保留大小写、符号、前导零、小数点、负号和百分号。数字分关心顺序、遗漏和添加。 |
| Table | `0.45 × 结构 F1 + 0.45 × 单元格内容相似度 + 0.10 × 数字单元格一致率` | 结构以起始行列和两个 span 匹配。只对匹配位置计内容相似度，漏/多出的单元格进入分母。数字按单元格位置核对。格式不符合目标协议或合并占位非法时奖励为 0。 |

无数字的样本去掉数字项并重新归一化。可用 `rsi-ocr reward table prediction.otsl reference.html`（OCR 用 `text`）输出分项。建议首轮抽查正确答案、近似错字、错数字、错合并、缺行和多输出，确认奖励顺序与业务要求一致。表格奖励使用网格结构而独立评估使用 PubTabNet 定义的 TEDS，两者可以同时观察。

## 参考

- OXR 的公开 RSI 经验提供了从真实错误找数据的思路。工程沿用这个起点，并保留每条真实候选的模型与像素证据。[OXR 仓库和 FAQ](https://github.com/SoMarkAI/OXR/blob/main/docs/FAQ.md)
- PaddleOCR‑VL‑1.6 报告提供了欠优化区域、多专家标注、表格 canonical form 和按任务计算可验证奖励的参考。olmOCR 2 用可验证的文档解析单元测试训练，PubTabNet 的 TEDS 提供了训练外表格评估依据。[PaddleOCR‑VL‑1.6](https://arxiv.org/html/2606.03264v1) · [olmOCR 2](https://arxiv.org/abs/2510.19817) · [PubTabNet / TEDS](https://arxiv.org/abs/1911.10683)
- 启动与数据协议以官方工程为准。[ERNIEKit SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft.md) · [ms-swift Reward Plugins](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/reward_function.html) · [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3) · [Jev vLLM](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/vllm_deploy_jev_en.md) · [Jev 请求合同](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/minicpm_ocr_jev_docker_api_en.md)

## 5. 单卡本地评测与复核

以下示例使用仓库内的 Conda 环境、Linux/WSL 及 `/mnt/g/models`。单卡依次加载 Paddle、OXR、Ovis 和 Jev，OCR 响应写入 SQLite 缓存，后续评分和重试可复用。Paddle 页面版面分析使用 CPU。

### 环境与模型

先完成主 README 的环境安装和模型下载。以下命令为分阶段评测生成配置，`--models` 指向模型根目录。[安装步骤](../README.md#安装)

```bash
conda activate "$PWD/runs/conda"
python scripts/configure_local.py --models runs/model-cache
```

Jev 的文本预检需要 PATH 上的 Node.js，Chromium 需要 Linux 共享库和 CJK 字体。可使用系统已安装依赖，也可将 Noto CJK 字体放入 `runs/fonts/` 并创建 `runs/font-cache/`。Ubuntu 24.04 上缺少 `libnspr4.so`、`libnss3.so` 或 `libasound.so.2` 时，可将 `libnspr4`、`libnss3`、`libasound2t64` 的系统包解包到 `runs/chromium-libs/` 后生成配置，供渲染服务加载。生成器默认使用内置 SDK，可用 `--minicpm /srv/MiniCPM-OCR` 指定外部源码仓库。生成器会保留已有 `configs/local.yaml`，需要重新生成时可以用 `--out` 指定另一个文件。

### 基准评测

```bash
rsi-ocr prepare-omni runs/omnidocbench --out runs/omni-pilot/input \
  --pages 40 --table-pages 20 --regions-per-task 48 --download
python scripts/run_local.py --regions runs/omni-pilot/input/regions.jsonl \
  --out runs/omni-pilot
rsi-ocr evaluate runs/omni-pilot/benchmark/evidence.jsonl \
  runs/omni-pilot/input/references.jsonl --out runs/omni-pilot/eval \
  --partition holdout --budget 25
```

抽样按文档类型分层，`--table-pages` 为表格页预留名额。省略 `--regions-per-task` 可跑所选页面的全部区域。标注和页面来自固定版本的 OmniDocBench。按可识别的文件名前缀分组，匿名页沿用各自的页面标识。标注裁图的文字 CER、数字一致性、表格规范化 TEDS/TEDS-S 和数字单元格准确率用于区域识别对比。表格先用本项目的解析器统一成单元格树，分数适用于同批区域比较。参考标注单独写入 `references.jsonl`，评测样本标记为 `benchmark` 并排除在训练导出之外。先在 `calibration` 分区核对阈值，再看 `holdout`。评测报告在同一审核预算下比较均匀抽样、教师最大分歧和完整挖掘排序。[OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)

需要验证 Paddle 原页采集时，给 `run_local.py` 追加 `--pages runs/omni-pilot/input/pages.jsonl`，结果写入 `capture/` 与 `production/`。真实裁图可在质控页面回放。工作台优先使用已记录的区域框。对缺少坐标的历史裁图，只有与原页高相似度且唯一匹配时才显示定位框。日志按服务写入输出目录，在同一目录用 `--stage paddle|oxr|ovis|jev` 可重跑指定服务。
