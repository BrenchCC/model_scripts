import argparse
import logging
from pathlib import Path
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description = "Merge a LoRA adapter into a base model and save the result."
    )
    parser.add_argument(
        "--base_model_path",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/brench-projects/shenping/tokenized_models/Qwen3-4B_task_binary_template",
        help = "Path to the base HuggingFace model.",
    )
    parser.add_argument(
        "--lora_path",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/brench-projects/shenping/results/v1/qwen3-4B/lora/sft/v1.1_shenping_qwen3_4b_train_lr_5e_5_lora_reason",
        help = "Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--output_path",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/brench-projects/shenping/tokenized_models/Qwen3-4B_reason_merged",
        help = "Path to save the merged model.",
    )
    parser.add_argument(
        "--dtype",
        type = str,
        default = "bfloat16",
        choices = ["float32", "float16", "bfloat16"],
        help = "Data type for loading the base model. Default: bfloat16",
    )
    parser.add_argument(
        "--trust_remote_code",
        action = "store_true",
        default = True,
        help = "Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--device_map",
        type = str,
        default = "auto",
        choices = ["auto", "none"],
        help = "Device map for loading the base model. Default: auto",
    )
    parser.add_argument(
        "--device",
        type = str,
        default = "cuda",
        help = "Device to move the model to when device_map=none. Default: cuda",
    )
    return parser.parse_args()


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    device_map: str = "auto",
    device: str = "cuda",
) -> None:
    """Merge a LoRA adapter into a base model and save the merged weights.

    Args:
        base_model_path: Path to the base HuggingFace model.
        lora_path: Path to the LoRA adapter directory.
        output_path: Path to save the merged model.
        dtype: Data type for loading the base model.
        trust_remote_code: Whether to trust remote code.
        device_map: Device map strategy ("auto" or "none").
        device: Device to load the model on.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    logger.info("=" * 60)
    logger.info("Loading base model from: %s", base_model_path)
    logger.info("=" * 60)

    if device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype = torch_dtype,
            trust_remote_code = trust_remote_code,
            device_map = "auto",
            low_cpu_mem_usage = True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype = torch_dtype,
            trust_remote_code = trust_remote_code,
            device_map = None,
            low_cpu_mem_usage = True,
        )
        model = model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code = trust_remote_code,
        use_fast = True,
    )

    logger.info("Base model loaded. Params: %.2fB", sum(p.numel() for p in model.parameters()) / 1e9)

    logger.info("=" * 60)
    logger.info("Loading LoRA adapter from: %s", lora_path)
    logger.info("=" * 60)

    model = PeftModel.from_pretrained(model, lora_path)

    logger.info("LoRA adapter loaded. Merging weights...")

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    logger.info("Merge complete.")

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents = True, exist_ok = True)

    logger.info("=" * 60)
    logger.info("Saving merged model to: %s", output_path)
    logger.info("=" * 60)

    model.save_pretrained(output_path, safe_serialization = True)
    tokenizer.save_pretrained(output_path)

    # Preserve chat template if present
    chat_template_src = Path(base_model_path) / "chat_template.jinja"
    chat_template_dst = Path(output_path) / "chat_template.jinja"
    if chat_template_src.exists() and not chat_template_dst.exists():
        shutil.copy2(chat_template_src, chat_template_dst)

    logger.info("Merged model saved successfully!")
    logger.info("Output directory: %s", output_path)


def main():
    """Main entry point."""
    args = parse_args()

    merge_lora(
        base_model_path = args.base_model_path,
        lora_path = args.lora_path,
        output_path = args.output_path,
        dtype = args.dtype,
        trust_remote_code = args.trust_remote_code,
        device_map = args.device_map,
        device = args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )
    main()
