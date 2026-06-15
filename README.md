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
- `jsonl_to_excel.py`：将 JSONL 转换为 Excel，并嵌入图片或视频首帧预览
- `vllm_infer.sh`：使用可配置参数在后台启动 vLLM API 服务
- `vllm_server/server.py`：vLLM 推理引擎封装

## 安装依赖

```bash
pip install torch transformers pandas openpyxl
pip install vllm
pip install peft
pip install opencv-python Pillow
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

### 5) JSONL 转 Excel

将 JSONL 文件转换为 Excel。脚本会保留原始字段，并自动识别本地图片或视频路径，为对应字段新增 `<字段名>_preview` 预览列。图片会缩放后嵌入 Excel，视频会提取首帧作为预览。

```bash
python jsonl_to_excel.py \
  --input /path/to/results.jsonl \
  --output_dir /path/to/excel_output \
  --frame_size 320
```

常用可选参数：
- `--output_dir /path/to/output`：指定输出目录；未指定时默认在输入文件同级生成带 `_excel` 后缀的目录
- `--frame_size 320`：设置预览图宽度，单位为像素
- `--limit 100`：仅转换前 100 条有效记录，适合快速检查数据

媒体字段支持图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.gif`、`.tiff`、`.webp`；支持视频格式：`.mp4`、`.avi`、`.mov`、`.mkv`、`.flv`、`.wmv`、`.webm`、`.m4v`。媒体路径必须是当前机器可访问的本地路径；列表类型字段仅使用第一个路径生成预览。

### 6) 启动 vLLM API 服务

使用 `nohup` 在后台启动 `vllm serve`，运行日志写入脚本目录下的 `vllm.log`。

```bash
bash vllm_infer.sh \
  --model_path /path/to/model \
  --port 8888 \
  --host :: \
  --dtype auto \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 2 \
  --cuda-visible-devices "0,1"
```

使用默认参数启动：

```bash
bash vllm_infer.sh
```

查看脚本帮助和服务日志：

```bash
bash vllm_infer.sh --help
tail -f vllm.log
```

主要默认参数：
- 模型路径：`../models/Qwen3-VL-8B-Instruct`
- 服务地址：`[::]:8888`
- GPU 显存占用比例：`0.7`
- Tensor 并行大小：`2`
- 使用 GPU：`0,1`
- 本地媒体访问白名单：`/`

## 输入数据格式建议

推理脚本支持：`xlsx` / `csv` / `jsonl`。

- `--text-col` 模式：直接从单列读取 user 文本
- `--user-template` 模式：按占位符拼接多列
- 未指定时：默认优先尝试 `article + comment`，否则自动拼接整行键值
