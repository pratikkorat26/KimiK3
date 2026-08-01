"""
kimi_k3/model.py — KimiK3Model assembly (embed → blocks → norm → lm_head).

Depth: Block AttnRes via DepthHistory (embedding as b_0).
Sequence: AttentionCache for KDA / Gated MLA decode.

Shapes:
    tokens: (B, T) int64
    logits: (B, T, vocab)
    cache:  AttentionCache when use_cache=True
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention.cache import AttentionCache, KDACache, MLACache
from .block import KimiK3Block
from .config import ModelConfig
from .moe.router import QuantileBalancingRouter
from .mtp.heads import MTPHeads
from .norms import rms_norm
from .residuals import BlockAttnRes
from .residuals.depth_history import DepthHistory


class KimiK3Model(nn.Module):
    """Causal LM over the Kimi K3 architecture slots."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [KimiK3Block(cfg, layer_idx=i) for i in range(cfg.n_layer)]
        )
        # Final depth mix over sealed blocks (paper: aggregate all N blocks)
        self.out_res = BlockAttnRes(cfg, layer_idx=cfg.n_layer)
        self.norm = rms_norm(cfg.hidden_size, cfg.eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Multi-Token Prediction heads (training-only aux signal); None when disabled.
        self.mtp = MTPHeads(cfg) if cfg.num_nextn_predict_layers > 0 else None
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Match the released model's initialization policy."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, QuantileBalancingRouter):
            nn.init.kaiming_uniform_(module.w_router.weight, a=5**0.5)

    def _init_history(self, embed: torch.Tensor) -> DepthHistory:
        if self.cfg.use_interim_residual:
            # Additive path: partial is the residual stream, starts at embedding
            return DepthHistory(completed=[], partial=embed, sublayer_count=0)
        return DepthHistory.from_embedding(embed)

    def gradient_checkpointing_enable(self) -> None:
        """Checkpoint decoder blocks during training to reduce activation memory."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @staticmethod
    def _history_tensors(history: DepthHistory) -> tuple[torch.Tensor, ...]:
        tensors = tuple(history.completed)
        if history.partial is not None:
            tensors += (history.partial,)
        return tensors

    def _checkpointed_block(
        self,
        block: KimiK3Block,
        history: DepthHistory,
    ) -> DepthHistory:
        completed_count = len(history.completed)
        has_partial = history.partial is not None
        sublayer_count = history.sublayer_count

        def run(*state: torch.Tensor) -> tuple[torch.Tensor, ...]:
            completed = list(state[:completed_count])
            partial = state[completed_count] if has_partial else None
            local = DepthHistory(
                completed=completed,
                partial=partial,
                sublayer_count=sublayer_count,
            )
            updated, _ = block(local, mode="chunk", cache=None, use_cache=False)
            return self._history_tensors(updated)

        state = self._history_tensors(history)
        updated_state = checkpoint(run, *state, use_reentrant=True)
        if isinstance(updated_state, torch.Tensor):
            updated_state = (updated_state,)

        seals_block = (
            block.layer_idx > 0
            and block.layer_idx % block.attn_res.block_size == 0
            and has_partial
        )
        new_completed_count = completed_count + int(seals_block)
        return DepthHistory(
            completed=list(updated_state[:new_completed_count]),
            partial=updated_state[new_completed_count],
            sublayer_count=sublayer_count + 2,
        )

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must have shape (batch, sequence), got {tuple(tokens.shape)}"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "tokens must use torch.int32 or torch.int64, "
                f"got {tokens.dtype}"
            )
        if tokens.shape[0] == 0:
            raise ValueError("tokens batch dimension must be non-empty")
        if tokens.shape[1] == 0:
            raise ValueError("tokens sequence dimension must be non-empty")
        if tokens.device != self.embed_tokens.weight.device:
            raise ValueError(
                "tokens and model must be on the same device "
                f"({tokens.device} != {self.embed_tokens.weight.device})"
            )
        token_min = int(tokens.min().item())
        token_max = int(tokens.max().item())
        if token_min < 0 or token_max >= self.cfg.vocab_size:
            raise ValueError(
                f"token ids must be in [0, {self.cfg.vocab_size}), "
                f"got min={token_min}, max={token_max}"
            )

    @staticmethod
    def _validate_tensor(
        tensor: torch.Tensor,
        *,
        name: str,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(
                f"{name} device does not match tokens ({tensor.device} != {device})"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")

    def _validate_cache(
        self,
        cache: AttentionCache | None,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> int:
        """Validate a model cache and return its effective prefix length."""
        if cache is None:
            if sequence_length > self.cfg.max_seq_len:
                raise ValueError(
                    f"sequence length {sequence_length} exceeds max_seq_len "
                    f"{self.cfg.max_seq_len}"
                )
            return 0
        if not isinstance(cache, AttentionCache):
            raise TypeError(
                f"cache must be an AttentionCache or None, got {type(cache).__name__}"
            )
        if len(cache) != self.cfg.n_layer:
            raise ValueError(
                f"cache has {len(cache)} layers, expected {self.cfg.n_layer}"
            )
        if (
            not isinstance(cache.tokens_seen, int)
            or isinstance(cache.tokens_seen, bool)
            or cache.tokens_seen < 0
        ):
            raise ValueError(
                f"cache.tokens_seen must be a non-negative integer, got {cache.tokens_seen!r}"
            )

        mla_lengths: set[int] = set()
        K = self.cfg.conv_kernel_size
        H = self.cfg.n_heads
        d_k = self.cfg.head_dim_k
        d_v = self.cfg.head_dim_v
        for layer_idx, layer_cache in enumerate(cache.layers):
            expected_type = self.cfg.attention_type(layer_idx)
            if expected_type == "kda":
                if not isinstance(layer_cache, KDACache):
                    raise TypeError(
                        f"cache layer {layer_idx} must be KDACache, "
                        f"got {type(layer_cache).__name__}"
                    )
                self._validate_tensor(
                    layer_cache.recurrent,
                    name=f"cache layer {layer_idx} recurrent",
                    shape=(batch_size, H, d_k, d_v),
                    device=device,
                )
                self._validate_tensor(
                    layer_cache.conv_q,
                    name=f"cache layer {layer_idx} conv_q",
                    shape=(batch_size, K - 1, H * d_k),
                    device=device,
                )
                self._validate_tensor(
                    layer_cache.conv_k,
                    name=f"cache layer {layer_idx} conv_k",
                    shape=(batch_size, K - 1, H * d_k),
                    device=device,
                )
                self._validate_tensor(
                    layer_cache.conv_v,
                    name=f"cache layer {layer_idx} conv_v",
                    shape=(batch_size, K - 1, H * d_v),
                    device=device,
                )
                conv_dtypes = {
                    layer_cache.conv_q.dtype,
                    layer_cache.conv_k.dtype,
                    layer_cache.conv_v.dtype,
                }
                if len(conv_dtypes) != 1:
                    raise TypeError(
                        f"cache layer {layer_idx} convolution tensors must share a dtype"
                    )
            else:
                if not isinstance(layer_cache, MLACache):
                    raise TypeError(
                        f"cache layer {layer_idx} must be MLACache, "
                        f"got {type(layer_cache).__name__}"
                    )
                prefix_len = layer_cache.seq_len
                self._validate_tensor(
                    layer_cache.kv_latent,
                    name=f"cache layer {layer_idx} kv_latent",
                    shape=(batch_size, prefix_len, self.cfg.kv_latent_dim),
                    device=device,
                )
                self._validate_tensor(
                    layer_cache.k_shared,
                    name=f"cache layer {layer_idx} k_shared",
                    shape=(batch_size, prefix_len, self.cfg.qk_shared_dim),
                    device=device,
                )
                if layer_cache.kv_latent.dtype != layer_cache.k_shared.dtype:
                    raise TypeError(
                        f"cache layer {layer_idx} MLA tensors must share a dtype"
                    )
                mla_lengths.add(prefix_len)

        if len(mla_lengths) > 1:
            raise ValueError(
                f"MLA cache layers disagree on prefix length: {sorted(mla_lengths)}"
            )
        inferred_tokens_seen = next(iter(mla_lengths), cache.tokens_seen)
        if cache.tokens_seen not in (0, inferred_tokens_seen):
            raise ValueError(
                "cache.tokens_seen disagrees with MLA cache length "
                f"({cache.tokens_seen} != {inferred_tokens_seen})"
            )
        prefix_len = inferred_tokens_seen if cache.tokens_seen == 0 else cache.tokens_seen
        total_length = prefix_len + sequence_length
        if total_length > self.cfg.max_seq_len:
            raise ValueError(
                f"cached sequence length {total_length} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        return prefix_len

    def forward(
        self,
        tokens: torch.Tensor,
        mode: str = "chunk",
        cache: AttentionCache | None = None,
        use_cache: bool = False,
        return_mtp: bool = False,
    ) -> (
        tuple[torch.Tensor, AttentionCache | None]
        | tuple[torch.Tensor, list[torch.Tensor], AttentionCache | None]
    ):
        """tokens (B, T) → (logits (B, T, vocab), AttentionCache | None).

        With return_mtp=True (chunk mode only), returns
        (logits, mtp_logits, cache) where mtp_logits is a list of D auxiliary
        logit tensors from the Multi-Token Prediction heads (empty if disabled).
        """
        self._validate_tokens(tokens)
        prefix_len = self._validate_cache(
            cache,
            batch_size=tokens.shape[0],
            sequence_length=tokens.shape[1],
            device=tokens.device,
        )
        x = self.embed_tokens(tokens)            # x: (B, T, d)
        history = self._init_history(x)
        new_cache = AttentionCache.empty(self.cfg.n_layer) if use_cache else None
        if new_cache is not None:
            new_cache.tokens_seen = prefix_len + tokens.shape[1]

        for i, block in enumerate(self.blocks):
            assert isinstance(block, KimiK3Block)
            layer_cache = cache.get(i) if cache is not None else None
            if (
                self.gradient_checkpointing
                and self.training
                and mode == "chunk"
                and not use_cache
                and cache is None
            ):
                history = self._checkpointed_block(block, history)
                c = None
            else:
                history, c = block(
                    history, mode=mode, cache=layer_cache, use_cache=use_cache
                )
            if new_cache is not None and c is not None:
                new_cache.set(i, c)

        if self.cfg.use_interim_residual:
            x = history.partial
        else:
            history.seal_remainder()
            x = self.out_res.mix(history)

        x = self.norm(x)
        logits = self.lm_head(x)
        if return_mtp:
            if mode != "chunk":
                raise ValueError("return_mtp is only supported in mode='chunk'")
            mtp_logits = (
                self.mtp(x, tokens, self.embed_tokens, self.lm_head)
                if self.mtp is not None
                else []
            )
            return logits, mtp_logits, new_cache
        return logits, new_cache

    def _select_next(
        self,
        logits_last: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Pick the next token from last-step logits (B, vocab) → ids (B, 1).

        Greedy when do_sample is False. Otherwise apply temperature, then
        optional top-k and top-p (nucleus) filtering, then multinomial sample.
        """
        if not do_sample:
            return logits_last.argmax(dim=-1, keepdim=True)

        logits = logits_last.float() / temperature
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = probs.cumsum(dim=-1)
            # Drop the tail whose *preceding* cumulative mass already exceeds p;
            # always keep the single most-probable token.
            remove = (cumprobs - probs) > top_p
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits
            )
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive decode: chunk prefill, then recurrent steps with cache.

        Greedy by default (do_sample=False). With do_sample=True, sample with
        temperature / top_k / top_p. If eos_token_id is given, finished
        sequences keep emitting EOS and decoding stops early once all are done.
        """
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError(
                "max_new_tokens must be a non-negative integer, "
                f"got {max_new_tokens!r}"
            )
        if max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be non-negative, got {max_new_tokens}"
            )
        if do_sample and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or temperature <= 0
        ):
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        ):
            raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
        if top_p is not None and (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not 0.0 < top_p <= 1.0
        ):
            raise ValueError(f"top_p must be in (0, 1] or None, got {top_p!r}")
        self._validate_tokens(tokens)
        requested_length = tokens.shape[1] + max_new_tokens
        if requested_length > self.cfg.max_seq_len:
            raise ValueError(
                f"requested sequence length {requested_length} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        if max_new_tokens == 0:
            return tokens

        batch_size = tokens.shape[0]

        was_training = self.training
        self.eval()
        try:
            logits, cache = self(tokens, mode="chunk", use_cache=True)
            next_tok = self._select_next(
                logits[:, -1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
            if eos_token_id is not None:
                finished |= next_tok.squeeze(-1) == eos_token_id
            seq = [tokens, next_tok]
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and bool(finished.all()):
                    break
                logits, cache = self(
                    next_tok, mode="recurrent", cache=cache, use_cache=True
                )
                next_tok = self._select_next(
                logits[:, -1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
                if eos_token_id is not None:
                    # Already-finished rows keep emitting EOS (no drift).
                    next_tok = torch.where(
                        finished.unsqueeze(-1),
                        next_tok.new_full(next_tok.shape, eos_token_id),
                        next_tok,
                    )
                    finished |= next_tok.squeeze(-1) == eos_token_id
                seq.append(next_tok)
            return torch.cat(seq, dim=1)
        finally:
            self.train(was_training)
