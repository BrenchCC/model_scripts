#!/usr/bin/env python3
"""Generic single-sample Transformers inference script.

This script loads a HuggingFace causal LM, builds one chat input from a dataset
row, runs inference once, and optionally inspects/checks special tokens.
"""

import argparse
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from inference_data_utils import load_single_row, read_text_file, build_user_content, get_label_text


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Generic Transformers single inference script"
    )
    parser.add_argument(
        "--model-path",
        type = str,
        required = True,
        help = "Path to HuggingFace model directory"
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
        help = "Optional path to system prompt text file"
    )
    parser.add_argument(
        "--system-prompt-text",
        type = str,
        default = "",
        help = "Optional raw system prompt text (used when path is not set)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type = int,
        default = 128,
        help = "Maximum number of new tokens"
    )
    parser.add_argument(
        "--min-new-tokens",
        type = int,
        default = 1,
        help = "Minimum number of new tokens"
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
        "--enable-thinking",
        type = str,
        default = "none",
        choices = ["true", "false", "none"],
        help = "Reasoning toggle for models that support enable_thinking"
    )
    parser.add_argument(
        "--special-tokens",
        type = str,
        nargs = "+",
        default = [],
        help = "Optional special tokens for tokenizer/output inspection"
    )
    parser.add_argument(
        "--constrain-to-special-tokens",
        action = "store_true",
        help = "Force first generated token to be one of special tokens"
    )
    parser.add_argument(
        "--skip-special-tokens",
        action = "store_true",
        help = "Skip special tokens when decoding assistant_content"
    )
    parser.add_argument(
        "--lora-path",
        type = str,
        default = None,
        help = "Optional PEFT LoRA adapter path"
    )
    parser.add_argument(
        "--lora-merge",
        action = "store_true",
        help = "Merge LoRA weights into base model after loading"
    )
    return parser.parse_args()


def maybe_load_lora(model, lora_path: str | None, lora_merge: bool):
    """Optionally load LoRA adapter.

    Args:
        model: Base model instance.
        lora_path: Optional LoRA adapter path.
        lora_merge: Whether to merge LoRA weights.

    Returns:
        Any: Model with optional LoRA loaded.
    """
    if lora_path is None:
        return model

    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `peft`. Install with: pip install peft"
        ) from exc

    logger.info("Loading LoRA adapter from: %s", lora_path)
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        is_trainable = False,
        torch_dtype = "auto",
        device_map = "auto"
    )
    if lora_merge:
        model = model.merge_and_unload()
        logger.info("LoRA merged into base model")
    else:
        logger.info("LoRA loaded without merge")
    return model


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
    """Build OpenAI-style chat messages.

    Args:
        system_prompt: System prompt.
        user_content: User content.

    Returns:
        list[dict[str, str]]: Message list.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def build_inputs(tokenizer, messages: list[dict[str, str]], enable_thinking: bool | None):
    """Build prompt text and tokenized model inputs.

    Args:
        tokenizer: HuggingFace tokenizer.
        messages: Chat messages.
        enable_thinking: Optional thinking flag.

    Returns:
        tuple[str, Any]: Prompt text and tokenized tensors.
    """
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        **template_kwargs
    )
    model_inputs = tokenizer(
        [prompt_text],
        return_tensors = "pt",
        add_special_tokens = False
    )
    return prompt_text, model_inputs


def build_prefix_allowed_tokens_fn(
    input_ids_len: int,
    allowed_first_token_ids: list[int],
    eos_token_id: int
):
    """Build constrained decoding callback.

    Args:
        input_ids_len: Prompt length.
        allowed_first_token_ids: Allowed first token ids.
        eos_token_id: EOS token id.

    Returns:
        Callable: Callback for generation constraint.
    """

    def _fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        del batch_id
        cur_len = int(input_ids.shape[-1])
        if cur_len == input_ids_len:
            return allowed_first_token_ids
        return [eos_token_id]

    return _fn


def inspect_special_tokens(tokenizer, special_tokens: list[str]) -> dict[str, dict[str, object]]:
    """Inspect tokenizer behavior for special tokens.

    Args:
        tokenizer: HuggingFace tokenizer.
        special_tokens: Token strings.

    Returns:
        dict[str, dict[str, object]]: Token inspection results.
    """
    info: dict[str, dict[str, object]] = {}
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens = False)
        info[token] = {
            "token_id": token_id,
            "encoded": encoded,
            "is_single_token": len(encoded) == 1
        }
    return info


def check_special_token_in_generation(
    tokenizer,
    generated_id_list: list[int],
    decoded_text: str,
    special_tokens: list[str]
) -> dict[str, dict[str, object]]:
    """Check token appearance in generated output.

    Args:
        tokenizer: HuggingFace tokenizer.
        generated_id_list: Generated token ids.
        decoded_text: Decoded text (raw).
        special_tokens: Tokens to search.

    Returns:
        dict[str, dict[str, object]]: Match results.
    """
    result: dict[str, dict[str, object]] = {}
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        result[token] = {
            "found_in_text": token in decoded_text,
            "found_in_ids": (token_id in generated_id_list) if token_id is not None else False,
            "token_id": token_id
        }
    return result


def main() -> dict[str, object]:
    """Run one inference pass.

    Returns:
        dict[str, object]: Summary for this run.
    """
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Transformers Single Inference")
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code = True,
        padding_side = "left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code = True,
        device_map = "auto",
        torch_dtype = "auto"
    )
    model = maybe_load_lora(
        model = model,
        lora_path = args.lora_path,
        lora_merge = args.lora_merge,
    )
    model.eval()

    enable_thinking = normalize_enable_thinking(args.enable_thinking)
    prompt_text, model_inputs = build_inputs(
        tokenizer = tokenizer,
        messages = messages,
        enable_thinking = enable_thinking,
    )

    device = next(model.parameters()).device
    model_inputs = model_inputs.to(device)
    do_sample = args.temperature > 0.0

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    tokenizer_inspect: dict[str, dict[str, object]] = {}
    if args.special_tokens:
        tokenizer_inspect = inspect_special_tokens(tokenizer, args.special_tokens)
        logger.info("tokenizer_inspect = %s", tokenizer_inspect)

    if args.constrain_to_special_tokens:
        if not args.special_tokens:
            raise ValueError("--constrain-to-special-tokens requires --special-tokens")

        allowed_ids = []
        for token in args.special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                allowed_ids.append(token_id)

        allowed_ids = sorted(set(allowed_ids))
        if not allowed_ids:
            raise ValueError("No valid token ids found in --special-tokens")

        gen_kwargs["prefix_allowed_tokens_fn"] = build_prefix_allowed_tokens_fn(
            input_ids_len = len(model_inputs.input_ids[0]),
            allowed_first_token_ids = allowed_ids,
            eos_token_id = tokenizer.eos_token_id,
        )

    logger.info("prompt_preview = %s", prompt_text[:500])
    logger.info("generation_kwargs = %s", gen_kwargs)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **gen_kwargs
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    assistant_content = tokenizer.decode(
        output_ids,
        skip_special_tokens = args.skip_special_tokens,
    )
    assistant_raw_content = tokenizer.decode(output_ids, skip_special_tokens = False)

    print("assistant_content:", assistant_content)
    print("assistant_raw_content:", assistant_raw_content)

    generation_check: dict[str, dict[str, object]] = {}
    if args.special_tokens:
        generation_check = check_special_token_in_generation(
            tokenizer = tokenizer,
            generated_id_list = output_ids,
            decoded_text = assistant_raw_content,
            special_tokens = args.special_tokens,
        )
        logger.info("generation_check = %s", generation_check)

    summary = {
        "label_text": label_text,
        "generated_text": assistant_raw_content,
        "tokenizer_inspect": tokenizer_inspect,
        "generation_check": generation_check,
    }
    logger.info("summary = %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
