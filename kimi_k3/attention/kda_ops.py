"""
kimi_k3/attention/kda_ops.py — the two KDA core algorithms.

Paper: "Kimi Linear: An Expressive, Efficient Attention Architecture"
       (arXiv:2510.26692).

  1) kda_recurrence — the per-token form of Eq. 1. This is the decode path
     (one token at a time, a fixed-size state S instead of a KV cache) and
     the correctness reference for the chunkwise form.

  2) kda_chunkwise — the hardware-efficient chunkwise-parallel form used for
     training (Listing 8b and Eqs. 2-9 of the paper).

KDA extends Gated DeltaNet by replacing the per-head scalar decay with a
per-channel diagonal gate Diag(alpha_t): every channel of the d_k state
dimension forgets at its own learned rate (paper Section 3, Table 6):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

Both functions work on per-head tensors and keep the state S in fp32.

Notation (identical to the paper):
    B  batch size            H  number of heads
    T  sequence length       C  chunk size (paper: 64)
    N  number of chunks      d_k / d_v  key / value head dims (paper: 128)
    g  per-channel log-decay, g = log(alpha) <= 0
    gc within-chunk cumulative log-decay (the paper's gamma, Eqs. 3-9)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kda_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-token KDA recurrence (Eq. 1 of the paper).

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    Intuition: S is an associative memory. Each step first decays every
    channel of S by its own rate (Diag(alpha_t) — the fine-grained KDA gate),
    then performs one online gradient step on the reconstruction loss
    (beta_t/2) ||v_t - S^T k_t||^2, i.e. it erases what S currently stores
    under key k_t and writes v_t in its place (the classical delta rule).

    Shapes:
        q:             (B, H, T, d_k)    L2-normalized queries (already scaled)
        k:             (B, H, T, d_k)    L2-normalized keys
        v:             (B, H, T, d_v)    values
        g:             (B, H, T, d_k)    per-channel log-decay, g = log(alpha) <= 0
        beta:          (B, H, T)         per-head delta-rule step size in (0, 1)
        initial_state: (B, H, d_k, d_v)  optional S_0 (e.g. from a cache), fp32

    Returns:
        o: (B, H, T, d_v)       per-token outputs, cast back to q.dtype
        S: (B, H, d_k, d_v)     final recurrent state, fp32
    """
    dtype = q.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))

    B, H, T, d_k = q.shape
    d_v = v.shape[-1]

    S = initial_state.float() if initial_state is not None else q.new_zeros(B, H, d_k, d_v)
    # S: (B, H, d_k, d_v) — fixed-size recurrent state, fp32 for stability

    outs = []
    for t in range(T):
        q_t = q[:, :, t]                              # q_t:     (B, H, d_k)
        k_t = k[:, :, t]                              # k_t:     (B, H, d_k)
        v_t = v[:, :, t]                              # v_t:     (B, H, d_v)
        alpha_t = g[:, :, t].exp()                    # alpha_t: (B, H, d_k)   diagonal of Diag(alpha_t)
        beta_t = beta[:, :, t]                        # beta_t:  (B, H)

        # --- Eq. 1, rightmost factor first: fine-grained decay Diag(alpha_t) S ---
        S = alpha_t.unsqueeze(-1) * S                 # S: (B, H, d_k, d_v)   decay each d_k channel independently

        # --- Eq. 1, delta rule: (I - beta k k^T) acts on the decayed state ---
        kv = torch.einsum('bhk,bhkv->bhv', k_t, S)    # kv: (B, H, d_v)       value S currently stores under k_t
        S = S + beta_t[..., None, None] * torch.einsum('bhk,bhv->bhkv', k_t, v_t - kv)
        # S: (B, H, d_k, d_v)   S += beta_t * k_t (v_t - S^T k_t)^T  (erase + write)

        # --- Eq. 1: read out o_t = S_t^T q_t ---
        o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)   # o_t: (B, H, d_v)
        outs.append(o_t)

    o = torch.stack(outs, dim=2)                      # o: (B, H, T, d_v)
    return o.to(dtype), S


def kda_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Chunkwise-parallel KDA (Listing 8b and Eqs. 2-9 of the paper).

    Same math as kda_recurrence, but the sequence is processed in chunks of
    C tokens: all intra-chunk work is done in parallel with matmuls, while a
    compact state S is carried recurrently across chunks. Because KDA binds
    the DPLR low-rank vectors to the keys (a = b = k), only two intra-chunk
    matrices (A_qk, A_kk) are needed and the reciprocal-decay division of the
    general DPLR form disappears (paper Section 6.2) — every exp() that
    survives the masks is a decay <= 1, so no log-domain secondary chunking
    is required.

    Shapes:
        q:             (B, H, T, d_k)    L2-normalized queries (already scaled)
        k:             (B, H, T, d_k)    L2-normalized keys
        v:             (B, H, T, d_v)    values
        g:             (B, H, T, d_k)    per-channel log-decay, g = log(alpha) <= 0
        beta:          (B, H, T)         per-head delta-rule step size in (0, 1)
        chunk_size:    int               C (paper: 64)
        initial_state: (B, H, d_k, d_v)  optional S_0, fp32

    Returns:
        o: (B, H, T, d_v)       per-token outputs, cast back to q.dtype
        S: (B, H, d_k, d_v)     final recurrent state after the last chunk, fp32
    """
    dtype = q.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))

    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    C = chunk_size

    # ------------------------------------------------------------------
    # Step 1: pad T to a whole number of chunks, reshape into chunks, and
    #         compute the within-chunk cumulative log-decay (gamma, Eqs. 3-5).
    #         Padded positions get k = 0, so they never touch the state.
    # ------------------------------------------------------------------
    N = (T + C - 1) // C
    T_pad = N * C
    if T_pad != T:
        pad = (0, 0, 0, T_pad - T)                    # zero-pad along T (dim 2)
        q, k, v, g = (F.pad(t, pad) for t in (q, k, v, g))
        beta = F.pad(beta, (0, T_pad - T))

    q = q.reshape(B, H, N, C, d_k)                    # q:    (B, H, N, C, d_k)
    k = k.reshape(B, H, N, C, d_k)                    # k:    (B, H, N, C, d_k)
    v = v.reshape(B, H, N, C, d_v)                    # v:    (B, H, N, C, d_v)
    g = g.reshape(B, H, N, C, d_k)                    # g:    (B, H, N, C, d_k)
    beta = beta.reshape(B, H, N, C)                   # beta: (B, H, N, C)

    gc = g.cumsum(dim=-2)                             # gc:   (B, H, N, C, d_k)
    # gc[i] = g[1] + ... + g[i]; the decay between positions j <= i inside one
    # chunk is exp(gc[i] - gc[j]), always in (0, 1] since g <= 0.

    # ------------------------------------------------------------------
    # Step 2: intra-chunk matrices (Listing 8b, lines 9-15).
    #   A_qk[i, j] = <q_i * exp(gc_i - gc_j), k_j>   for j <= i   (causal)
    #   A_kk[i, j] = <k_i * exp(gc_i - gc_j), k_j>   for j <  i   (strictly
    #              lower; the strict-tril mask in Step 3 drops the rest)
    # Both use the SAME between-position decay exp(gc_i - gc_j) — the
    # coefficient k_i^T Diag(gamma_{j->i}) k_j of the w/u recurrences
    # (Eqs. 4-5); the two matrices differ only in their masks. Every
    # exponential that survives masking is a decay (<= 1), never a
    # division by a small gamma.
    # ------------------------------------------------------------------
    A_qk_rows, A_kk_rows = [], []                     # rows stacked below (functional: keeps autograd intact)
    pos = torch.arange(C, device=q.device)
    for i in range(C):
        q_i = q[:, :, :, i:i + 1, :]                  # q_i:  (B, H, N, 1, d_k)
        k_i = k[:, :, :, i:i + 1, :]                  # k_i:  (B, H, N, 1, d_k)
        gc_i = gc[:, :, :, i:i + 1, :]                # gc_i: (B, H, N, 1, d_k)

        decay_i = (gc_i - gc).exp()                   # decay_i: (B, H, N, C, d_k)  decay of position j as seen at i
        s1_i = decay_i.masked_fill((pos > i).view(1, 1, 1, C, 1), 0.0)   # causal: keep j <= i
        A_qk_rows.append((q_i * k * s1_i).sum(-1))    # row i: (B, H, N, C)
        A_kk_rows.append((k_i * k * decay_i).sum(-1)) # row i: (B, H, N, C)

    A_qk = torch.stack(A_qk_rows, dim=3)              # A_qk: (B, H, N, C, C)
    A_kk = torch.stack(A_kk_rows, dim=3)              # A_kk: (B, H, N, C, C)

    A_kk = A_kk * beta.unsqueeze(-1)                  # A_kk: (B, H, N, C, C)  left Diag(beta) of Eq. 6

    # ------------------------------------------------------------------
    # Step 3: UT transform (Eq. 6, Listing 8b lines 16-20).
    #   M = (I + StrictTril(A_kk))^{-1} Diag(beta)
    # The inverse is found by forward substitution on the strictly-lower
    # part — no matmul inverse is ever formed. Row i is final once rows
    # < i are, so the loop is exact after C-1 passes.
    # ------------------------------------------------------------------
    strict_upper = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    A = -A_kk.masked_fill(strict_upper, 0.0)          # A: (B, H, N, C, C)  strictly-lower, negated
    for i in range(1, C):
        # (paper's in-place Neumann accumulation; .clone() guards aliasing)
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = A + torch.eye(C, device=q.device)             # A = (I + StrictTril(A_kk))^{-1}: (B, H, N, C, C)
    M = A * beta.unsqueeze(-2)                        # M: (B, H, N, C, C)   right Diag(beta) of Eq. 6

    # ------------------------------------------------------------------
    # Step 4: WY representation (Eq. 7). The C rank-1 updates of one chunk
    #         are packed into two dense factors.
    # ------------------------------------------------------------------
    W = torch.einsum('bhnij,bhnjd->bhnid', M, gc.exp() * k)   # W: (B, H, N, C, d_k)   M (Gamma ⊙ K)
    U = torch.einsum('bhnij,bhnjd->bhnid', M, v)              # U: (B, H, N, C, d_v)   M V

    # ------------------------------------------------------------------
    # Step 5: inter-chunk recurrence (Eqs. 8-9, Listing 8b lines 24-29).
    #   For each chunk, the output is the sum of
    #     inter = (Gamma ⊙ Q) S          — read from the incoming state
    #     intra = A_qk (U - W S)         — causal in-chunk correction with
    #                                      "pseudo-values" (U - W S)
    #   then the state advances:
    #     S <- Diag(gamma_C) S + (Gamma_{i->C} ⊙ K)^T (U - W S)
    # ------------------------------------------------------------------
    S = initial_state.float() if initial_state is not None else q.new_zeros(B, H, d_k, d_v)
    # S: (B, H, d_k, d_v) fp32 — the only quantity carried across chunks
    O_chunks = []                                     # chunks stacked below (functional: keeps autograd intact)
    for n in range(N):
        q_decayed = q[:, :, n] * gc[:, :, n].exp()    # q_decayed: (B, H, C, d_k)      Gamma ⊙ Q

        # --- Eq. 9: chunk output (uses the state S from the chunk's start) ---
        inter = torch.einsum('bhck,bhkv->bhcv', q_decayed, S)
        # inter: (B, H, C, d_v)   (Gamma ⊙ Q) S
        pseudo = U[:, :, n] - torch.einsum('bhck,bhkv->bhcv', W[:, :, n], S)
        # pseudo: (B, H, C, d_v)  U - W S, the "pseudo-value" term of Eq. 9
        intra = torch.einsum('bhcj,bhjv->bhcv', A_qk[:, :, n], pseudo)
        # intra: (B, H, C, d_v)   Tril((Gamma ⊙ Q)(K Gamma)^T) (U - W S)
        O_chunks.append(inter + intra)                # O_n: (B, H, C, d_v)

        # --- Eq. 8: state update (same chunk-start S inside `pseudo`) ---
        gc_last = gc[:, :, n, -1, :]                  # gc_last: (B, H, d_k)           full-chunk decay gamma_C
        decay_to_end = (gc_last.unsqueeze(-2) - gc[:, :, n]).exp()
        # decay_to_end: (B, H, C, d_k)   decay of key i up to the chunk end, Gamma_{i->C}
        S = gc_last.exp().unsqueeze(-1) * S + torch.einsum('bhck,bhcv->bhkv', k[:, :, n] * decay_to_end, pseudo)
        # S: (B, H, d_k, d_v)   Diag(gamma_C) S + (Gamma_{i->C} ⊙ K)^T (U - W S)

    out_all = torch.stack(O_chunks, dim=2)            # out_all: (B, H, N, C, d_v)
    o = out_all.reshape(B, H, T_pad, d_v)[:, :, :T]   # o: (B, H, T, d_v)   drop the padding
    return o.to(dtype), S
