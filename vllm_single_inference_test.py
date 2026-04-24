#!/usr/bin/env python3
"""Generic single-sample vLLM inference script."""

import argparse
import logging

from inference_data_utils import load_single_row, read_text_file, build_user_content, get_label_text
from vllm_server.server import VLLMEngine


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Generic vLLM single inference script"
    )
    parser.add_argument(
        "--model-path",
        type = str,
        required = True,
        help = "Path to model directory"
    )
    parser.add_argument(
        "--data-path",
        type = str,
        required = True,
        help = "Input data path (xlsx/csv/jsonl)"
    )
    parser.add_argument(
        "--input-format",
        type = str,
        default = "auto",
        choices = ["auto", "xlsx", "csv", "jsonl"],
        help = "Input data format"
    )
    parser.add_argument(
        "--sheet-name",
        type = str,
        default = None,
        help = "Sheet name/index for xlsx input"
    )
    parser.add_argument(
        "--index",
        type = int,
        default = 0,
        help = "Sample index"
    )
    parser.add_argument(
        "--label-col",
        type = str,
        default = None,
        help = "Optional label column used for logging"
    )
    parser.add_argument(
        "--text-col",
        type = str,
        default = None,
        help = "Optional text column for direct user content"
    )
    parser.add_argument(
        "--user-template",
        type = str,
        default = None,
        help = "Optional user content template, e.g. 'Title: {title}\\nBody: {body}'"
    )
    parser.add_argument(
        "--system-prompt-path",
        type = str,
        default = None,
        help = "Optional system prompt text file"
    )
    parser.add_argument(
        "--system-prompt-text",
        type = str,
        default = "",
        help = "Optional raw system prompt text"
    )
    parser.add_argument(
        "--max-tokens",
        type = int,
        default = 128,
        help = "Maximum generation tokens"
    )
    parser.add_argument(
        "--temperature",
        type = float,
        default = 0.0,
        help = "Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type = float,
        default = 1.0,
        help = "Top-p sampling parameter"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type = int,
        default = 1,
        help = "Tensor parallel size"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type = float,
        default = 0.9,
        help = "GPU memory utilization"
    )
    parser.add_argument(
        "--max-model-len",
        type = int,
        default = 4096,
        help = "Maximum model length"
    )
    parser.add_argument(
        "--enable-thinking",
        type = str,
        default = "none",
        choices = ["true", "false", "none"],
        help = "Reasoning toggle"
    )
    parser.add_argument(
        "--special-tokens",
        type = str,
        nargs = "+",
        default = [],
        help = "Special tokens to inspect/check"
    )
    parser.add_argument(
        "--skip-special-tokens",
        action = "store_true",
        help = "Skip special tokens in decoded output"
    )
    parser.add_argument(
        "--lora-path",
        type = str,
        default = None,
        help = "Optional LoRA adapter path"
    )
    parser.add_argument(
        "--lora-name",
        type = str,
        default = "default",
        help = "LoRA name"
    )
    parser.add_argument(
        "--lora-int-id",
        type = int,
        default = 1,
        help = "LoRA integer id"
    )
    parser.add_argument(
        "--max-loras",
        type = int,
        default = 1,
        help = "Maximum LoRAs in a batch"
    )
    parser.add_argument(
        "--max-lora-rank",
        type = int,
        default = 16,
        help = "Maximum LoRA rank"
    )
    parser.add_argument(
        "--lora-extra-vocab-size",
        type = int,
        default = 256,
        help = "Extra vocab hint for LoRA compatibility"
    )
    parser.add_argument(
        "--lora-dtype",
        type = str,
        default = "auto",
        choices = ["auto", "float16", "bfloat16"],
        help = "LoRA dtype"
    )
    parser.add_argument(
        "--max-cpu-loras",
        type = int,
        default = None,
        help = "Max LoRAs cached on CPU"
    )
    parser.add_argument(
        "--fully-sharded-loras",
        action = "store_true",
        help = "Enable fully sharded LoRAs"
    )
    return parser.parse_args()


def normalize_enable_thinking(enable_thinking: str) -> bool | None:
    """Convert CLI value to bool/None.

    Args:
        enable_thinking: One of true/false/none.

    Returns:
        bool | None: Parsed value.
    """
    if enable_thinking == "true":
        return True
    if enable_thinking == "false":
        return False
    return None


def build_messages(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    """Build OpenAI-style messages.

    Args:
        system_prompt: System prompt.
        user_content: User content.

    Returns:
        list[dict[str, str]]: Message list.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def inspect_special_tokens(engine: VLLMEngine, special_tokens: list[str]) -> dict[str, dict[str, object]]:
    """Inspect tokenizer behavior for special tokens.

    Args:
        engine: Initialized engine.
        special_tokens: Token strings.

    Returns:
        dict[str, dict[str, object]]: Token inspection results.
    """
    info: dict[str, dict[str, object]] = {}
    for token in special_tokens:
        token_id = engine.tokenizer.convert_tokens_to_ids(token)
        encoded = engine.tokenizer.encode(token, add_special_tokens = False)
        info[token] = {
            "token_id": token_id,
            "encoded": encoded,
            "is_single_token": len(encoded) == 1,
        }
    return info


def check_special_token(output: str, special_tokens: list[str]) -> dict[str, object]:
    """Check whether special tokens appear in text.

    Args:
        output: Model output text.
        special_tokens: Token strings to search.

    Returns:
        dict[str, object]: Match summary.
    """
    matched_tokens = [token for token in special_tokens if token in output]

    extracted_label = "NONE"
    if len(matched_tokens) == 1:
        extracted_label = matched_tokens[0]
    elif len(matched_tokens) > 1:
        extracted_label = "MULTIPLE"

    return {
        "has_special_token": bool(matched_tokens),
        "matched_tokens": matched_tokens,
        "extracted_label": extracted_label,
    }


def main() -> dict[str, object]:
    """Run one vLLM inference pass.

    Returns:
        dict[str, object]: Summary for this run.
    """
    args = parse_args()
    enable_thinking = normalize_enable_thinking(args.enable_thinking)

    logger.info("=" * 80)
    logger.info("vLLM Single Inference")
    logger.info("=" * 80)

    sample = load_single_row(
        data_path = args.data_path,
        input_format = args.input_format,
        index = args.index,
        sheet_name = args.sheet_name,
    )
    label_text = get_label_text(row = sample, label_col = args.label_col)
    user_content = build_user_content(
        row = sample,
        text_col = args.text_col,
        user_template = args.user_template,
    )

    if args.system_prompt_path:
        system_prompt = read_text_file(args.system_prompt_path)
    else:
        system_prompt = args.system_prompt_text

    messages = build_messages(
        system_prompt = system_prompt,
        user_content = user_content,
    )

    logger.info("data_path = %s", args.data_path)
    logger.info("sample_index = %s", args.index)
    logger.info("sample_keys = %s", list(sample.keys()))
    if label_text is not None:
        logger.info("label_text = %s", label_text)

    engine = VLLMEngine(
        model_path = args.model_path,
        tensor_parallel_size = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len = args.max_model_len,
        trust_remote_code = True,
        lora_path = args.lora_path,
        lora_name = args.lora_name,
        lora_int_id = args.lora_int_id,
        max_loras = args.max_loras,
        max_lora_rank = args.max_lora_rank,
        lora_extra_vocab_size = args.lora_extra_vocab_size,
        lora_dtype = args.lora_dtype,
        max_cpu_loras = args.max_cpu_loras,
        fully_sharded_loras = args.fully_sharded_loras,
    )

    tokenizer_inspect: dict[str, dict[str, object]] = {}
    if args.special_tokens:
        tokenizer_inspect = inspect_special_tokens(engine, args.special_tokens)
        logger.info("tokenizer_inspect = %s", tokenizer_inspect)

    prompt_text = engine.build_prompt(
        messages = messages,
        enable_thinking = enable_thinking,
    )
    logger.info("prompt_preview = %s", prompt_text[:500])

    chat_kwargs = {
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "use_tqdm": False,
        "enable_thinking": enable_thinking,
        "skip_special_tokens": args.skip_special_tokens,
    }
    if args.lora_path is not None:
        chat_kwargs["lora_path"] = args.lora_path
        chat_kwargs["lora_name"] = args.lora_name
        chat_kwargs["lora_int_id"] = args.lora_int_id

    outputs = engine.chat(**chat_kwargs)
    generated_text = outputs[0] if outputs else ""

    generation_check: dict[str, object] = {}
    if args.special_tokens:
        generation_check = check_special_token(generated_text, args.special_tokens)

    print("assistant_content:", generated_text)

    summary = {
        "label_text": label_text,
        "generated_text": generated_text,
        "tokenizer_inspect": tokenizer_inspect,
        "generation_check": generation_check,
    }
    logger.info("summary = %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )
    main()
