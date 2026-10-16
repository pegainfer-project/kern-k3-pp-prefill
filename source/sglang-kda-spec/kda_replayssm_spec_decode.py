# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM speculative-decode state commit for KDA (Kimi Delta Attention).

KDA keeps its own recurrent verify kernel for the per-step OUTPUT (unchanged),
so — unlike the GDN spec kernel (gdn_replayssm_spec_decode.py) — this module does
NOT reconstruct the verify output. It only replaces the per-draft-token full-SSM
snapshot cache (``intermediate_ssm``, ``max_running×(γ+1)×[HV,V,K]``, the memory
hog that collapses dspark concurrency) with a small per-request input window +
an exact-fold on commit.

Scheme (fold-every-commit, no circular ring / periodic flush):
  * during verify, the KDA backend stores the draft window's raw inputs
    (raw v, raw pre-norm k, per-K log-decay gate ``gk`` (fp32), beta (fp32)) into
    the per-slot ring at positions ``0..spec_len``;
  * on commit, :func:`commit_kda_replayssm_spec` replays the *accepted* prefix
    ``0..accept_len`` from the persistent checkpoint ``h0`` into ``h0`` in place —
    ``h0`` is always the current committed state, so the next verify reads it
    directly as its initial state (no lag, no chunked reconstruction).

The exact-fold is a BITWISE CLONE of ``fused_recurrent_gated_delta_rule_fwd_kernel``'s
``IS_KDA`` branch (fused_recurrent.py): same [BK, BV] fp32 tile (K rows, V cols),
same division-form L2 norm (eps inside sqrt), same per-K gate decay
``h *= exp(gk)``, same decay→delta→rank-1 op order. Given identical inputs the
folded checkpoint is bit-identical to the recurrent baseline's committed state
(the delta-rule recurrence is contractive, so no length-dependent error). Do NOT
reorder into tl.dot / reciprocal-multiply; keep num_warps=1 so the reduction
trees match.

Linear chain only (dspark γ chain, topk<=1).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, V, K] fp32 checkpoint (folded in place)
    rawv_cache,  # [num_slots, HV, L, V]  raw v
    rawk_cache,  # [num_slots, H,  L, K]  raw pre-norm k
    gk_cache,  # [num_slots, HV, L, K]  fp32 per-K log-decay gate
    beta_cache,  # [num_slots, HV, L]     fp32 beta
    ssm_state_indices,  # [B] int  physical slot per request
    accept_lens,  # [B] int  committed prefix length per request
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    # program_id(2) packs (layer, v-head): layer-major so a single launch folds
    # all KDA layers. num_layers=1 launches (per-layer entry) keep i_layer == 0.
    i_hvl = tl.program_id(2)
    # int64: layer stride * i_layer overflows int32 at K3 scale (69 layers x
    # ~34M-element per-layer stride > 2^31).
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    # Shift the layer-indexed bases once; every pointer below is layer-relative.
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return
    ring_idx = i_n.to(tl.int64)

    # extra_buffer: snapshot the interval-crossing state into the track ping-pong
    # slot on the step it crosses (mask-gated: -1 track step / NULL slot = skip).
    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # [BK, BV] tile: K rows / V cols, matching the recurrent baseline's memory
    # offset (v * K + k, K contiguous).
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + ring_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + ring_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_gk = tl.load(
            gk_cache
            + ring_idx * stride_gk_slot
            + (i_hv * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + ring_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)

        # --- verbatim recurrent update, IS_KDA branch (see module docstring) ---
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])  # per-K gate decay, broadcast over V
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        # Interval-crossing snapshot -> track slot (state AFTER step `track_step`).
        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


