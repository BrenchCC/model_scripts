# model_scripts

一个用于沉淀常用大模型开发脚本的仓库，重点覆盖模型推理验证、数据预处理、模型转换与 LoRA 相关辅助流程。

## 仓库目标

- 提供可复用、可参数化的 LLM 工具脚本
- 减少项目间重复造轮子
- 用统一 CLI 风格快速完成常见开发与调试任务

## 脚本清单

- `transformers_single_inference_test.py`：基于 HuggingFace Transformers 的单样本推理脚本
- `vllm_single_inference_test.py`：基于 vLLM 的单样本推理脚本
- `data_binary_split.py`：通用二分类数据集 train/test 拆分脚本
- `tokenizer_model_convert.py`：分词器/模型格式转换相关脚本
- `merge_lora.py`：LoRA 合并脚本
- `vllm_server/server.py`：vLLM 推理引擎封装

## 安装依赖

```bash
pip install torch transformers pandas openpyxl
pip install vllm
pip install peft
```

按实际需求安装即可，脚本并不要求一次性安装全部依赖。

## 快速使用

### 1) Transformers 单条推理

```bash
python transformers_single_inference_test.py \
  --model-path /path/to/model \
  --data-path /path/to/data.xlsx \
  --input-format xlsx \
  --label-col label \
  --text-col text \
  --index 0 \
  --max-new-tokens 64
```

常用可选参数：
- `--user-template "Title: {title}\nBody: {body}"`
- `--system-prompt-path /path/to/system_prompt.md`
- `--special-tokens <LABEL_POS> <LABEL_NEG>`
- `--lora-path /path/to/lora --lora-merge`

### 2) vLLM 单条推理

```bash
python vllm_single_inference_test.py \
  --model-path /path/to/model \
  --data-path /path/to/data.csv \
  --input-format csv \
  --label-col label \
  --user-template "Instruction: {instruction}\nInput: {input}" \
  --index 0 \
  --max-tokens 64
```

常用可选参数：
- `--tensor-parallel-size 2`
- `--gpu-memory-utilization 0.9`
- `--special-tokens <YES> <NO>`
- `--lora-path /path/to/lora`

### 3) 二分类数据拆分（比例模式）

```bash
python data_binary_split.py \
  --input-path /path/to/data.xlsx \
  --label-col label \
  --split-mode ratio \
  --train-ratio 0.8
```

### 4) 二分类数据拆分（指定数量模式）

```bash
python data_binary_split.py \
  --input-path /path/to/data.csv \
  --label-col is_positive \
  --split-mode count \
  --train-pos 800 --train-neg 800 \
  --test-pos 200 --test-neg 200 \
  --positive-labels "1,true,yes" \
  --negative-labels "0,false,no"
```

## 输入数据格式建议

推理脚本支持：`xlsx` / `csv` / `jsonl`。

- `--text-col` 模式：直接从单列读取 user 文本
- `--user-template` 模式：按占位符拼接多列
- 未指定时：默认优先尝试 `article + comment`，否则自动拼接整行键值

## 文档

- 代码逻辑说明见：`docs/code_logic/overview.md`

