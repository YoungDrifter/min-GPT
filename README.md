# min-GPT

一个面向学习与研究的英文对话模型项目：使用基础 PyTorch 手写 GPT-2 兼容的 Decoder-only Transformer，加载 `microsoft/DialoGPT-small` 预训练权重，并在 DailyDialog 数据集上进行全参数微调。

项目没有依赖 `transformers` 实现 tokenizer、模型前向传播或训练循环。代码从 byte-level BPE、对话样本组织和 causal self-attention 开始，覆盖训练、完整测试集评估、文本生成、多轮终端对话及 W&B 指标记录。

## 项目特点

- 手写 GPT-2 byte-level BPE tokenizer；
- 手写 causal self-attention、MLP、Pre-LN Block 和 Decoder-only GPT；
- 兼容 DialoGPT-small 的 tokenizer、模型结构与预训练权重；
- 使用 reply-only loss，只监督目标回复，保留历史对话作为上下文；
- 支持单轮生成、多轮聊天以及微调前后的固定参数对比；
- 在完整 test split 上分别评估 baseline 与最佳 checkpoint；
- 按 MPS、CUDA、CPU 的顺序自动选择运行设备；
- 可选接入 W&B 记录训练与评估指标。

为了让核心实现保持清晰，当前版本没有使用 PyTorch 的 Transformer/Attention 封装，也没有实现 KV cache、beam search 或 top-p sampling。

## 项目结构

```text
min-GPT/
├── README.md
├── requirements.txt
├── config.py                  # 路径、模型、训练与生成配置
├── tokenizer.py               # GPT-2 byte-level BPE tokenizer
├── data.py                    # DailyDialog 数据集与 batch 构造
├── model.py                   # Decoder-only GPT 与预训练权重导入
├── train.py                   # 全参数微调、验证与 checkpoint 保存
├── evaluate.py                # 完整 test split 评估
├── generate.py                # generate / chat / compare 命令行入口
├── data/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
└── results/
    ├── ACCEPTANCE.md           # 最终训练、测试与生成验收记录
    ├── evaluation.json         # 训练和完整测试集指标
    ├── baseline_samples.json   # 预训练模型的固定 prompt 输出
    └── comparison_samples.json # 微调前后的固定 prompt 对比
```

以下本地资产由程序读取或生成，但不纳入 Git：

```text
.cache/DialoGPT-small/         # tokenizer 文件与预训练权重
checkpoints/                   # latest.pt 与 best.pt
wandb/                         # 本地 W&B 运行记录
```

## 环境准备

建议在独立的 Python 环境中安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

项目当前固定的主要依赖为：

```text
torch==2.13.0
numpy==2.4.6
safetensors==0.8.0
wandb==0.29.0
```

### DialoGPT-small 本地资产

运行前需要在 `.cache/DialoGPT-small/` 中准备以下文件：

```text
model.safetensors
vocab.json
merges.txt
```

- `model.safetensors`：DialoGPT-small 预训练权重；
- `vocab.json`：包含 50,257 个 token 的 GPT-2 词表；
- `merges.txt`：GPT-2 BPE 合并规则。

模型结构、特殊 token 和默认超参数统一定义在 `config.py` 中。

### DailyDialog 数据

`data/` 包含英文对话数据的三个 split，每行是一个 JSON 对象：

```json
{"history": ["Hello, how are you?"], "reply": "I'm fine, thank you."}
```

- `history`：当前回复之前的对话历史；
- `reply`：模型需要学习生成的目标回复。

当前数据规模：

| Split | 样本数 |
| --- | ---: |
| train | 76,052 |
| validation | 7,069 |
| test | 6,740 |

## 快速开始

所有命令均在项目根目录运行。

### 使用预训练模型生成回复

```bash
python3 generate.py generate --prompt "Hello!"
```

### 使用微调模型生成回复

```bash
python3 generate.py generate \
  --prompt "Hello!" \
  --checkpoint checkpoints/best.pt
```

### 开始多轮终端对话

```bash
python3 generate.py chat --checkpoint checkpoints/best.pt
```

输入 `exit`、`quit` 或 Ctrl-D 结束对话。

### 对比微调前后的输出

```bash
python3 generate.py compare \
  --prompt "Hello!" \
  --checkpoint checkpoints/best.pt
```

baseline 与微调模型使用相同的 prompt、seed 和采样参数，以便直接比较。

## 数据如何进入模型

每条样本被组织为：

```text
history[0] + EOS + history[1] + EOS + ... + reply + EOS
```

对应的 labels 为：

```text
history 与 padding → -100
reply 与 reply EOS → 原 token ID
```

计算交叉熵时，输入与标签各错开一个位置：

```python
logits[:, :-1]
labels[:, 1:]
```

因此，history 不直接计入 token-level loss，但仍通过 attention 为回复生成提供上下文，并能从 reply loss 接收梯度。序列最长为 256 tokens；超长时优先删除最旧的完整轮次。batch 使用 EOS 进行右侧填充，history 和 padding 对应的 label 均保持为 `-100`。

## Tokenizer

`tokenizer.py` 实现了 GPT-2 byte-level BPE：

```text
文本 → UTF-8 bytes → byte encoder → 伪文本字符序列
     → BPE 合并 → token 字符串 → token IDs
```

`<|endoftext|>` 的 token ID 为 `50256`，同时用作轮次分隔符和 EOS。微调不会修改 `vocab.json` 或 `merges.txt`，更新的是模型内部的 token embedding，以及与它共享权重的 `lm_head`。

## 模型结构

模型与 DialoGPT-small 的 GPT-2 结构兼容：

| 参数 | 值 |
| --- | ---: |
| Transformer blocks | 12 |
| hidden size | 768 |
| attention heads | 12 |
| MLP hidden size | 3,072 |
| maximum positions | 1,024 |
| vocabulary size | 50,257 |
| dropout | 0.1 |
| activation | GELU-new |

每个 Block 均采用 Pre-LN：

```text
x → LayerNorm → causal self-attention → residual
  → LayerNorm → MLP → residual
```

## 训练

默认训练配置：

| 参数 | 值 |
| --- | ---: |
| device priority | MPS → CUDA → CPU |
| context length | 256 |
| micro-batch size | 2 |
| gradient accumulation | 16 |
| effective batch size | 32 |
| optimizer | AdamW |
| peak learning rate | 2e-5 |
| minimum learning rate | 2e-6 |
| warmup steps | 100 |
| weight decay | 0.01 |
| gradient clipping | 1.0 |
| optimizer steps | 2,500 |
| seed | 1337 |

开始默认训练：

```bash
python3 train.py
```

指定本次训练步数：

```bash
python3 train.py --max-steps 1500
```

训练会在 step 0、每 50 个 optimizer steps 以及最终 step 上，使用固定的 20 个 validation batches 计算 reply-only loss。checkpoint 保存至：

```text
checkpoints/latest.pt
checkpoints/best.pt
```

每个 checkpoint 包含模型参数、optimizer step 和 validation loss。

### W&B 记录

```bash
wandb login
python3 train.py --wandb
```

训练过程记录：

- `train/loss`
- `train/learning_rate`
- `train/gradient_norm`
- `validation/loss`
- `validation/perplexity`

## 完整测试集评估

训练期间的 validation 只用于快速观察收敛趋势。最终评估由 `evaluate.py` 遍历全部 6,740 条 test samples，按有效 reply token 数累计 loss 并计算 perplexity。评估使用 `model.eval()` 与 `torch.no_grad()`，不会修改模型参数或 checkpoint。

评估原始 DialoGPT-small：

```bash
python3 evaluate.py --wandb --run-name baseline-test
```

评估微调后的最佳 checkpoint：

```bash
python3 evaluate.py \
  --checkpoint checkpoints/best.pt \
  --wandb \
  --run-name best-test
```

不需要 W&B 时，可省略 `--wandb` 和 `--run-name`。程序默认每 50 个 evaluation batches 输出一次累计结果；可通过 `--log-interval N` 修改间隔。

W&B 评估指标包括：

- `test/loss`
- `test/perplexity`
- `evaluation/batch`
- `evaluation/samples`
- `evaluation/tokens`

## 已完成实验结果

当前 `results/evaluation.json` 记录了一次 2,500-step 全参数微调及其完整测试集评估：

| 模型 | Test loss | Perplexity |
| --- | ---: | ---: |
| Pretrained DialoGPT-small | 3.745823 | 42.343856 |
| Fine-tuned `best.pt` | 2.659185 | 14.284649 |

两个模型均评估了完整的 6,740 条测试样本和 107,825 个有效 reply tokens。与 baseline 相比，微调模型的 test loss 下降 29.01%，perplexity 下降 66.27%。

结果文件：

- [`results/ACCEPTANCE.md`](results/ACCEPTANCE.md)：训练、测试和固定 prompt 的完整验收说明；
- [`results/evaluation.json`](results/evaluation.json)：机器可读的训练与测试指标；
- [`results/baseline_samples.json`](results/baseline_samples.json)：baseline 的固定 prompt 输出；
- [`results/comparison_samples.json`](results/comparison_samples.json)：baseline 与微调模型的输出对比。

这些结果证明 reply modeling 指标得到改善，但不代表模型能够稳定生成高质量对话。固定 prompt 的输出仍显示出 relevance 和长程连贯性不足。

## 默认生成参数

| 参数 | 值 |
| --- | ---: |
| max new tokens | 50 |
| temperature | 0.75 |
| top-k | 50 |
| repetition penalty | 1.1 |
| seed | 1337 |

以上参数均可通过 `generate.py` 的命令行选项覆盖。生成仅解码新生成的 suffix，并在遇到 EOS 时立即停止。

## 已知限制

- DialoGPT-small 约有 124M 参数，回复质量上限有限；
- 训练与多轮聊天上下文均限制为 256 tokens；
- DailyDialog 主要覆盖英文日常对话；
- 只使用 top-k sampling，没有实现 top-p 或 beam search；
- 没有 KV cache，生成每个 token 时都会重新计算当前上下文；
- 固定 prompt 结果中的英文形式通常可辨认，但相关性和长程连贯性并不稳定。

## 许可证说明

- DailyDialog：CC BY-NC-SA 4.0，仅限非商业用途；
- DialoGPT-small：MIT License；
- 本项目代码：用于学习与研究。
