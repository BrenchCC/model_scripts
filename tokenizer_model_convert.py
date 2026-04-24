import os
import json
import logging
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    --model_path: path to the base model directory
    --special_token_json: path to the special token template JSON file
    --output_path: (optional) path to save the converted model, default is auto-generated
    """
    parser = argparse.ArgumentParser(description = "Tokenizer Convert: add special tokens and resize model embeddings")
    parser.add_argument(
        "--model_path",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/base_models/Qwen3-4B-Instruct-2507",
        help = "Path to the base model directory"
    )
    parser.add_argument(
        "--special_token_json",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/brench-projects/shenping/special_token_templates/task_binary_template.json",
        help = "Path to the special token template JSON file"
    )
    parser.add_argument(
        "--output_path",
        type = str,
        default = "/mnt/bn/brench-hl-volume-v1/aigc/llm-train-playground/brench-projects/shenping/tokenized_models/Qwen3-4B-Instruct-2507_task_binary_template",    
        help = "Path to save the converted model. If not provided, auto-generated based on model name and template name"
    )
    return parser.parse_args()


def load_special_tokens(json_path):
    """
    Load special tokens from a JSON template file.

    json_path: path to the special token template JSON file
    Returns a list of special token strings
    """
    with open(json_path, "r", encoding = "utf-8") as f:
        token_dict = json.load(f)
    label_list = list(token_dict.values())
    logger.info(f"从模板加载了 {len(label_list)} 个特殊token: {label_list}")
    return label_list


def add_tokens(tokenizer, label_list):
    """
    Add special tokens to the tokenizer.

    tokenizer: the tokenizer instance
    label_list: list of special token strings to add
    """
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": label_list})
    logger.info(f"已向tokenizer添加 {num_added} 个特殊token")
    for token in label_list:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  token '{token}' -> ID: {token_id}")


def find_reference_token(tokenizer):
    """
    Find a reference token ID for initializing new token embeddings.
    Uses the eos_token as reference since it is always present.

    tokenizer: the tokenizer instance
    Returns the token ID of the reference token
    """
    ref_token = tokenizer.eos_token
    ref_id = tokenizer.convert_tokens_to_ids(ref_token)
    logger.info(f"参考token: '{ref_token}' (ID: {ref_id})")
    return ref_id


def change_model_size(model, tokenizer):
    """
    Resize model embeddings to accommodate the expanded tokenizer.

    model: the model instance
    tokenizer: the tokenizer instance (after adding special tokens)
    """
    # Resize embedding and lm_head to fit the expanded tokenizer, preserving existing token weights
    origin_emb_size = model.get_input_embeddings().weight.data.shape[0]
    new_emb_size = max(len(tokenizer), origin_emb_size)
    logger.info(f"原始embedding大小: {origin_emb_size}")
    logger.info(f"新embedding大小: {new_emb_size}")

    model.resize_token_embeddings(new_emb_size)
    logger.info("已完成模型embedding层的大小调整")


def change_model_token_weights(model, tokenizer, labels):
    """
    Initialize new token embeddings by copying weights from a reference token.

    model: the model instance
    tokenizer: the tokenizer instance
    labels: list of new special token strings
    """
    # Get the input embedding layer and lm_head parameters
    embedding_layer = model.get_input_embeddings()
    lm_head_layer = model.get_output_embeddings()

    logger.info("-" * 60)
    logger.info("模型层信息:")
    logger.info(f"Embedding层权重形状: {embedding_layer.weight.data.shape}")
    if lm_head_layer is not None:
        logger.info(f"LM Head层权重形状: {lm_head_layer.weight.data.shape}")
    else:
        logger.info("LM Head层未找到或与Embedding层共享权重")

    # Copy reference token embeddings to new tokens
    logger.info("-" * 60)
    logger.info("正在初始化新token的嵌入权重...")
    reference_token_id = find_reference_token(tokenizer)

    for token in labels:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"初始化token '{token}' (ID: {token_id})")

        # Copy embedding weights
        embedding_layer.weight.data[token_id] = embedding_layer.weight.data[reference_token_id].clone()

        # For tie_word_embeddings == True, embedding and lm_head share weights, no extra init needed
        if not model.config.tie_word_embeddings and lm_head_layer is not None:
            lm_head_layer.weight.data[token_id] = lm_head_layer.weight.data[reference_token_id].clone()
            logger.info(f"  已单独初始化LM Head层权重 for token '{token}'")
        else:
            logger.info(f"  Embedding和LM Head权重绑定或LM Head不存在，已通过embedding初始化完成token '{token}'的权重")


def save_model(model, tokenizer, output_path):
    """
    Save the model and tokenizer to the specified path.

    model: the model instance
    tokenizer: the tokenizer instance
    output_path: directory to save the model
    """
    os.makedirs(output_path, exist_ok = True)
    logger.info("=" * 80)
    logger.info(f"正在保存模型到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"模型和tokenizer已保存到: {output_path}")
    logger.info("=" * 80)


def gen_tokenizer_model(model_path, special_token_json, output_path):
    """
    Main pipeline: load model, add special tokens, resize embeddings, initialize weights, and save.

    model_path: path to the base model directory
    special_token_json: path to the special token template JSON file
    output_path: path to save the converted model
    """
    # Load special tokens from template
    label_list = load_special_tokens(special_token_json)

    # Load tokenizer
    logger.info("=" * 80)
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side = "left",
        trust_remote_code = True
    )
    logger.info(f"原始tokenizer词表大小: {len(tokenizer)}")

    # Load model
    logger.info("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        device_map = "auto"
    )
    logger.info(f"模型加载完成, tie_word_embeddings = {model.config.tie_word_embeddings}")

    # Add special tokens to tokenizer
    logger.info("-" * 60)
    logger.info("正在向tokenizer添加新标签...")
    add_tokens(tokenizer, label_list)
    logger.info(f"扩展后tokenizer词表大小: {len(tokenizer)}")

    # Resize embedding and lm_head
    change_model_size(model, tokenizer)

    # Initialize new token weights
    change_model_token_weights(model, tokenizer, label_list)

    # Save model
    save_model(model, tokenizer, output_path)

    logger.info("*" * 50)
    logger.info("模型扩展完成!")
    logger.info("*" * 50)


def generate_output_path(model_path, special_token_json):
    """
    Auto-generate output path based on model name and template name.

    model_path: path to the base model directory
    special_token_json: path to the special token template JSON file
    Returns the generated output path string
    """
    model_name = os.path.basename(model_path.rstrip("/"))
    template_name = os.path.splitext(os.path.basename(special_token_json))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, "tokenizers", f"{model_name}_{template_name}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    args = parse_args()

    # Auto-generate output path if not specified
    output_path = args.output_path
    if output_path is None:
        output_path = generate_output_path(args.model_path, args.special_token_json)

    logger.info("=" * 80)
    logger.info("Tokenizer Convert 启动")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"特殊token模板: {args.special_token_json}")
    logger.info(f"输出路径: {output_path}")

    gen_tokenizer_model(
        model_path = args.model_path,
        special_token_json = args.special_token_json,
        output_path = output_path
    )