"""Hugging Face configuration and causal-LM adapter for Kimi K3."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .config import ModelConfig
from .model import KimiK3Model


class KimiK3HFConfig(PretrainedConfig):
    """Serializable Hugging Face view of :class:`ModelConfig`."""

    model_type = "kimi_k3"

    def __init__(
        self,
        *,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        defaults = ModelConfig().to_dict()
        tie_word_embeddings = bool(
            kwargs.get("tie_word_embeddings", defaults["tie_word_embeddings"])
        )
        for name, default in defaults.items():
            setattr(self, name, kwargs.pop(name, default))
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_model_config(
        cls,
        config: ModelConfig,
        *,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> KimiK3HFConfig:
        return cls(
            **config.to_dict(),
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    def to_model_config(self) -> ModelConfig:
        values = {
            name: getattr(self, name)
            for name in ModelConfig.__dataclass_fields__
        }
        return ModelConfig(**values)


class KimiK3ForCausalLM(PreTrainedModel):
    """Trainer-compatible wrapper around the reference Kimi K3 model."""

    config_class = KimiK3HFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiK3Block"]

    def __init__(self, config: KimiK3HFConfig) -> None:
        super().__init__(config)
        self.model = KimiK3Model(config.to_model_config())
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value) -> None:
        self.model.lm_head = value

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Any = None,
    ) -> None:
        del gradient_checkpointing_func
        if enable:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "Kimi K3 pretraining uses fully packed sequences; padded "
                "attention masks are not supported"
            )
        if use_cache and self.training:
            raise ValueError("use_cache must be false during training")

        logits, _ = self.model(
            input_ids,
            mode="chunk",
            use_cache=bool(use_cache),
        )
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must have the same shape as input_ids, "
                    f"got {tuple(labels.shape)} and {tuple(input_ids.shape)}"
                )
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            return ((loss, logits) if loss is not None else (logits,))
        return CausalLMOutput(loss=loss, logits=logits)  # type: ignore[arg-type]


def register_hf_auto_classes() -> None:
    """Register Kimi K3 with Hugging Face Auto classes in this process."""
    try:
        AutoConfig.register(KimiK3HFConfig.model_type, KimiK3HFConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(KimiK3HFConfig, KimiK3ForCausalLM)
    except ValueError:
        pass


register_hf_auto_classes()
