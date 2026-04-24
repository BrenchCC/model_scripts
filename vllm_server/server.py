import os
import json
import inspect
import logging
from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass


from vllm.lora.request import LoRARequest
from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase



logger = logging.getLogger(__name__)


def _ensure_tokenizer_compatibility() -> None:
    """Patch tokenizer compatibility gaps for local environment.

    Some tokenizer classes in the current `transformers` build do not expose
    `all_special_tokens_extended`, while installed `vllm` expects it during
    tokenizer caching. Add a conservative fallback property so vLLM can reuse
    the tokenizer stack without modifying upstream packages.
    """
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    logger.warning(
        "Patching `PreTrainedTokenizerBase.all_special_tokens_extended` for current transformers/vLLM compatibility."
    )

    @property
    def all_special_tokens_extended(self):
        return list(getattr(self, "all_special_tokens", []))

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


@dataclass
class EngineCapabilities:
    """Runtime capability summary for the loaded model stack.

    Args:
        model_type: HuggingFace config model type.
        architectures: Declared architecture names from config.
        supports_reasoning: Whether the model stack appears to support reasoning.
        supports_reasoning_toggle: Whether chat template exposes `enable_thinking`.
        reasoning_start_str: Reasoning start marker string.
        reasoning_end_str: Reasoning end marker string.
        extra_special_tokens: Extra special tokens detected from tokenizer config.
        tokenizer_vocab_size: Tokenizer vocabulary size.
        model_vocab_size: Model config vocabulary size.
        chat_template_available: Whether a chat template is available.
        chat_template_has_enable_thinking: Whether template references `enable_thinking`.
    """

    model_type: str
    architectures: list[str]
    supports_reasoning: bool
    supports_reasoning_toggle: bool
    reasoning_start_str: str
    reasoning_end_str: str
    extra_special_tokens: list[str]
    tokenizer_vocab_size: int
    model_vocab_size: int
    chat_template_available: bool
    chat_template_has_enable_thinking: bool


class VLLMEngine:
    """Generic vLLM engine for OpenAI-style messages.

    Args:
        model_path: Path to the base or merged model directory.
        tensor_parallel_size: Tensor parallel size for vLLM.
        gpu_memory_utilization: GPU memory utilization ratio.
        max_model_len: Optional max model sequence length.
        trust_remote_code: Whether to trust model remote code.
        tokenizer_path: Optional tokenizer path. Defaults to `model_path`.
        lora_path: Optional default LoRA adapter path.
        lora_name: Default LoRA name.
        lora_int_id: Default LoRA integer id.
        enable_lora: Whether to initialize vLLM with LoRA support.
        max_loras: Max number of LoRAs in a batch.
        max_lora_rank: Max supported LoRA rank.
        lora_extra_vocab_size: Deprecated extra vocab size hint kept for compatibility.
        lora_dtype: LoRA dtype passed to vLLM when supported.
        max_cpu_loras: Max number of CPU cached LoRAs.
        fully_sharded_loras: Whether to enable fully sharded LoRAs.
        enforce_reasoning_support: If True, invalid reasoning request raises error.
        enforce_extra_vocab_compatibility: If True, tokenizer/model vocab mismatch raises error.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        tokenizer_path: str | None = None,
        lora_path: str | None = None,
        lora_name: str = "default",
        lora_int_id: int = 1,
        enable_lora: bool | None = None,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        lora_extra_vocab_size: int | None = 256,
        lora_dtype: str = "auto",
        max_cpu_loras: int | None = None,
        fully_sharded_loras: bool = False,
        enforce_reasoning_support: bool = False,
        enforce_extra_vocab_compatibility: bool = True,
    ):
        """Initialize the reusable vLLM engine."""
        _ensure_tokenizer_compatibility()

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        self.default_lora_path = lora_path
        self.default_lora_name = lora_name
        self.default_lora_int_id = lora_int_id
        self.enable_lora = enable_lora if enable_lora is not None else lora_path is not None
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.lora_extra_vocab_size = lora_extra_vocab_size
        self.lora_dtype = lora_dtype
        self.max_cpu_loras = max_cpu_loras
        self.fully_sharded_loras = fully_sharded_loras
        self.enforce_reasoning_support = enforce_reasoning_support
        self.enforce_extra_vocab_compatibility = enforce_extra_vocab_compatibility

        self.last_prompt_text: str | None = None

        self.hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code = self.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code = self.trust_remote_code,
        )
        self.capabilities = self._detect_capabilities()
        self._validate_tokenizer_model_alignment()
        self.llm = self._build_llm()
        self._log_initialization_summary()

    def inspect_capabilities(self) -> dict[str, Any]:
        """Return cached capability information.

        Returns:
            dict[str, Any]: Capability summary.
        """
        return asdict(self.capabilities)

    def build_prompt(self, messages: list[dict[str, Any]], enable_thinking: bool | None = None) -> str:
        """Render OpenAI-style messages into a model prompt.

        Args:
            messages: OpenAI-style chat messages.
            enable_thinking: Optional reasoning toggle.

        Returns:
            str: Rendered prompt text.
        """
        normalized_messages = self._normalize_messages(messages)
        resolved_enable_thinking = self._resolve_reasoning_request(enable_thinking)
        template_kwargs: dict[str, Any] = {}

        if resolved_enable_thinking is not None and self.capabilities.supports_reasoning_toggle:
            template_kwargs["enable_thinking"] = resolved_enable_thinking

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize = False,
                add_generation_prompt = True,
                **template_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to render messages with tokenizer chat template. "
                "Please ensure the target model provides a valid chat template."
            ) from exc

        self.last_prompt_text = prompt_text
        return prompt_text

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        use_tqdm: bool = False,
        enable_thinking: bool | None = None,
        lora_path: str | None = None,
        lora_name: str | None = None,
        lora_int_id: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate texts from OpenAI-style messages.

        Args:
            messages: OpenAI-style chat messages.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum generation tokens.
            use_tqdm: Whether to show tqdm progress.
            enable_thinking: Optional reasoning toggle.
            lora_path: Optional request-level LoRA adapter path.
            lora_name: Optional request-level LoRA adapter name.
            lora_int_id: Optional request-level LoRA adapter id.
            **kwargs: Extra `SamplingParams` arguments.

        Returns:
            list[str]: Generated texts aligned with prompt order.
        """
        prompt_text = self.build_prompt(
            messages = messages,
            enable_thinking = enable_thinking,
        )
        sampling_params = self._build_sampling_params(
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            **kwargs,
        )
        lora_request = self._build_lora_request(
            lora_path = lora_path,
            lora_name = lora_name,
            lora_int_id = lora_int_id,
        )

        logger.info(
            "Running vLLM generation. prompt_chars = %s, reasoning_applied = %s, lora_enabled = %s",
            len(prompt_text),
            enable_thinking if enable_thinking is not None else "default",
            lora_request is not None,
        )

        outputs = self.llm.generate(
            prompts = [prompt_text],
            sampling_params = sampling_params,
            use_tqdm = use_tqdm,
            lora_request = lora_request,
        )
        return self._extract_texts(outputs)

    def _build_llm(self) -> LLM:
        """Initialize the underlying vLLM engine.

        Returns:
            LLM: Configured vLLM LLM instance.
        """
        llm_kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }

        supported_arg_names = self._get_engine_arg_names()
        optional_kwargs = {
            "max_model_len": self.max_model_len,
        }

        if self.enable_lora:
            optional_kwargs.update(
                {
                    "enable_lora": True,
                    "max_loras": self.max_loras,
                    "max_lora_rank": self.max_lora_rank,
                    "lora_dtype": self.lora_dtype,
                    "max_cpu_loras": self.max_cpu_loras,
                    "fully_sharded_loras": self.fully_sharded_loras,
                }
            )
            if "lora_extra_vocab_size" in supported_arg_names and self.lora_extra_vocab_size is not None:
                optional_kwargs["lora_extra_vocab_size"] = self.lora_extra_vocab_size
            elif self.lora_extra_vocab_size is not None:
                logger.info(
                    "Current vLLM version does not expose `lora_extra_vocab_size`; skip passing deprecated argument."
                )

        for key, value in optional_kwargs.items():
            if value is None:
                continue
            if key in supported_arg_names:
                llm_kwargs[key] = value

        return LLM(**llm_kwargs)

    def _get_engine_arg_names(self) -> set[str]:
        """Collect best-effort supported vLLM argument names.

        Returns:
            set[str]: Supported argument names.
        """
        supported_arg_names = set(inspect.signature(LLM.__init__).parameters.keys())

        model_fields = getattr(EngineArgs, "model_fields", None)
        if isinstance(model_fields, dict):
            supported_arg_names.update(model_fields.keys())

        annotations = getattr(EngineArgs, "__annotations__", {})
        if isinstance(annotations, dict):
            supported_arg_names.update(annotations.keys())

        return supported_arg_names

    def _detect_capabilities(self) -> EngineCapabilities:
        """Infer reasoning and tokenizer capabilities.

        Returns:
            EngineCapabilities: Detected capabilities.
        """
        model_type = str(getattr(self.hf_config, "model_type", "") or "")
        architectures = list(getattr(self.hf_config, "architectures", []) or [])
        chat_template_text = self._load_chat_template_text()
        chat_template_available = bool(chat_template_text)
        template_has_enable_thinking = "enable_thinking" in chat_template_text
        template_has_reasoning_tags = "<think>" in chat_template_text and "</think>" in chat_template_text
        reasoning_family = model_type.lower() == "qwen3" or any(
            "Qwen3" in architecture for architecture in architectures
        )

        reasoning_start_ids = self.tokenizer.encode(
            "<think>",
            add_special_tokens = False,
        )
        reasoning_end_ids = self.tokenizer.encode(
            "</think>",
            add_special_tokens = False,
        )
        reasoning_markers_available = bool(reasoning_start_ids) and bool(reasoning_end_ids)

        supports_reasoning_toggle = template_has_enable_thinking
        supports_reasoning = reasoning_family or template_has_reasoning_tags or supports_reasoning_toggle
        if not supports_reasoning and reasoning_markers_available and reasoning_family:
            supports_reasoning = True

        extra_special_tokens = self._load_extra_special_tokens()
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = int(getattr(self.hf_config, "vocab_size", tokenizer_vocab_size))

        return EngineCapabilities(
            model_type = model_type,
            architectures = architectures,
            supports_reasoning = supports_reasoning,
            supports_reasoning_toggle = supports_reasoning_toggle,
            reasoning_start_str = "<think>" if reasoning_markers_available and supports_reasoning else "",
            reasoning_end_str = "</think>" if reasoning_markers_available and supports_reasoning else "",
            extra_special_tokens = extra_special_tokens,
            tokenizer_vocab_size = tokenizer_vocab_size,
            model_vocab_size = model_vocab_size,
            chat_template_available = chat_template_available,
            chat_template_has_enable_thinking = template_has_enable_thinking,
        )

    def _load_chat_template_text(self) -> str:
        """Load chat template text from tokenizer or local file.

        Returns:
            str: Chat template text if available.
        """
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(chat_template, str) and chat_template.strip():
            return chat_template

        chat_template_path = Path(self.tokenizer_path) / "chat_template.jinja"
        if chat_template_path.exists():
            return chat_template_path.read_text(encoding = "utf-8")
        return ""

    def _load_extra_special_tokens(self) -> list[str]:
        """Load extra special tokens from tokenizer config and tokenizer runtime.

        Returns:
            list[str]: Unique extra special tokens.
        """
        tokens: list[str] = []
        tokenizer_config_path = Path(self.tokenizer_path) / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding = "utf-8"))
            config_tokens = tokenizer_config.get("extra_special_tokens", [])
            if isinstance(config_tokens, list):
                tokens.extend([str(token) for token in config_tokens])

        runtime_tokens = getattr(self.tokenizer, "additional_special_tokens", []) or []
        tokens.extend([str(token) for token in runtime_tokens])

        deduplicated_tokens: list[str] = []
        for token in tokens:
            if token not in deduplicated_tokens:
                deduplicated_tokens.append(token)
        return deduplicated_tokens

    def _validate_tokenizer_model_alignment(self) -> None:
        """Validate tokenizer/model vocab alignment."""
        tokenizer_vocab_size = self.capabilities.tokenizer_vocab_size
        model_vocab_size = self.capabilities.model_vocab_size

        if tokenizer_vocab_size > model_vocab_size and self.enforce_extra_vocab_compatibility:
            raise ValueError(
                "Tokenizer vocab is larger than model vocab. This usually means the model directory "
                "does not include resized embeddings for the expanded tokenizer. Please use the model "
                "generated by `scripts/tokenizer_convert.py` or a merged model directory. "
                f"tokenizer_vocab_size = {tokenizer_vocab_size}, model_vocab_size = {model_vocab_size}."
            )

        invalid_tokens: list[tuple[str, int]] = []
        for token in self.capabilities.extra_special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int) or token_id < 0 or token_id >= model_vocab_size:
                invalid_tokens.append((token, int(token_id)))

        if invalid_tokens and self.enforce_extra_vocab_compatibility:
            raise ValueError(
                "Detected extra special tokens whose token ids are outside model vocab range. "
                "Please ensure the base model has been expanded with `scripts/tokenizer_convert.py` before using LoRA. "
                f"invalid_tokens = {invalid_tokens}, model_vocab_size = {model_vocab_size}."
            )

        if tokenizer_vocab_size != model_vocab_size:
            logger.warning(
                "Tokenizer/model vocab sizes differ. tokenizer_vocab_size = %s, model_vocab_size = %s",
                tokenizer_vocab_size,
                model_vocab_size,
            )

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate messages for chat rendering.

        Args:
            messages: Input messages.

        Returns:
            list[dict[str, Any]]: Validated messages.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of OpenAI-style message dictionaries.")

        normalized_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"Message at index {index} must be a dict, got {type(message)!r}.")
            if "role" not in message:
                raise ValueError(f"Message at index {index} is missing required key `role`.")
            if "content" not in message:
                raise ValueError(f"Message at index {index} is missing required key `content`.")
            normalized_messages.append(message)
        return normalized_messages

    def _resolve_reasoning_request(self, enable_thinking: bool | None) -> bool | None:
        """Resolve reasoning toggle against detected capabilities.

        Args:
            enable_thinking: Requested reasoning toggle.

        Returns:
            bool | None: Effective toggle to pass into chat template.
        """
        if enable_thinking is None:
            return None

        if not self.capabilities.supports_reasoning:
            message = (
                "Requested reasoning toggle on a model that does not appear to support reasoning. "
                f"model_type = {self.capabilities.model_type}, architectures = {self.capabilities.architectures}"
            )
            if self.enforce_reasoning_support:
                raise ValueError(message)
            logger.warning(message)
            return None

        if not self.capabilities.supports_reasoning_toggle:
            logger.warning(
                "Model appears to support reasoning, but its chat template does not expose `enable_thinking`; "
                "fall back to template default behavior."
            )
            return None

        return enable_thinking

    def _build_sampling_params(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> SamplingParams:
        """Build sampling params for vLLM.

        Args:
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum generated tokens.
            **kwargs: Extra sampling kwargs.

        Returns:
            SamplingParams: Sampling params.
        """
        return SamplingParams(
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            **kwargs,
        )

    def _build_lora_request(
        self,
        lora_path: str | None,
        lora_name: str | None,
        lora_int_id: int | None,
    ) -> LoRARequest | None:
        """Build a request-level LoRA object when needed.

        Args:
            lora_path: Request-level LoRA path.
            lora_name: Request-level LoRA name.
            lora_int_id: Request-level LoRA id.

        Returns:
            LoRARequest | None: vLLM LoRA request.
        """
        resolved_lora_path = lora_path or self.default_lora_path
        if resolved_lora_path is None:
            return None

        if not self.enable_lora:
            raise ValueError(
                "A LoRA adapter was requested, but the engine was initialized without LoRA support. "
                "Please set `enable_lora = True` or provide `lora_path` during engine initialization."
            )

        if not os.path.isdir(resolved_lora_path):
            raise FileNotFoundError(f"LoRA path does not exist: {resolved_lora_path}")

        self._validate_lora_adapter_config(resolved_lora_path)
        self._validate_lora_extra_vocab_compatibility(resolved_lora_path)

        resolved_lora_name = lora_name or self.default_lora_name
        resolved_lora_int_id = lora_int_id or self.default_lora_int_id
        return LoRARequest(
            lora_name = resolved_lora_name,
            lora_int_id = resolved_lora_int_id,
            lora_path = resolved_lora_path,
        )

    def _validate_lora_extra_vocab_compatibility(self, lora_path: str) -> None:
        """Validate LoRA usage against expanded tokenizer state.

        Args:
            lora_path: LoRA adapter path.
        """
        if not self.capabilities.extra_special_tokens:
            logger.info("LoRA adapter requested without detected extra special tokens. path = %s", lora_path)
            return

        if self.capabilities.tokenizer_vocab_size > self.capabilities.model_vocab_size:
            raise ValueError(
                "LoRA adapter requested together with expanded tokenizer, but model vocab does not cover tokenizer vocab. "
                "Please use the model produced by `scripts/tokenizer_convert.py` or a merged model directory. "
                f"lora_path = {lora_path}"
            )

        logger.info(
            "LoRA adapter requested with validated extra special tokens: %s",
            self.capabilities.extra_special_tokens,
        )

    def _validate_lora_adapter_config(self, lora_path: str) -> None:
        """Validate adapter config against current vLLM limitations.

        Args:
            lora_path: LoRA adapter path.
        """
        adapter_config_path = Path(lora_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter config is missing: {adapter_config_path}"
            )

        adapter_config = json.loads(adapter_config_path.read_text(encoding = "utf-8"))
        modules_to_save = adapter_config.get("modules_to_save")
        if modules_to_save:
            raise ValueError(
                "This LoRA adapter cannot be loaded by vLLM directly because `adapter_config.json` "
                f"contains `modules_to_save = {modules_to_save}`. vLLM 0.11 only supports LoRA adapters "
                "with `modules_to_save = None`. Your adapter saves `embed_tokens` / `lm_head`, which means "
                "it should be used as a merged model for vLLM inference instead of runtime LoRA loading."
            )

        adapter_base_model = adapter_config.get("base_model_name_or_path")
        if adapter_base_model and os.path.abspath(str(adapter_base_model)) != os.path.abspath(self.model_path):
            logger.warning(
                "LoRA adapter base model path differs from current engine model_path. adapter_base_model = %s, model_path = %s",
                adapter_base_model,
                self.model_path,
            )

    def _extract_texts(self, outputs: list[Any]) -> list[str]:
        """Extract plain texts from vLLM outputs.

        Args:
            outputs: Raw vLLM outputs.

        Returns:
            list[str]: Extracted texts.
        """
        generated_texts: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None)
            if not candidates:
                generated_texts.append("")
                continue
            generated_texts.append(getattr(candidates[0], "text", ""))
        return generated_texts

    def _log_initialization_summary(self) -> None:
        """Log concise runtime summary."""
        logger.info("=" * 80)
        logger.info("VLLMEngine initialized")
        logger.info("model_path = %s", self.model_path)
        logger.info("tokenizer_path = %s", self.tokenizer_path)
        logger.info("tensor_parallel_size = %s", self.tensor_parallel_size)
        logger.info("gpu_memory_utilization = %s", self.gpu_memory_utilization)
        logger.info("max_model_len = %s", self.max_model_len)
        logger.info("enable_lora = %s", self.enable_lora)
        logger.info("model_type = %s", self.capabilities.model_type)
        logger.info("architectures = %s", self.capabilities.architectures)
        logger.info("supports_reasoning = %s", self.capabilities.supports_reasoning)
        logger.info("supports_reasoning_toggle = %s", self.capabilities.supports_reasoning_toggle)
        logger.info("extra_special_tokens = %s", self.capabilities.extra_special_tokens)
        logger.info("tokenizer_vocab_size = %s", self.capabilities.tokenizer_vocab_size)
        logger.info("model_vocab_size = %s", self.capabilities.model_vocab_size)
        logger.info("=" * 80)
