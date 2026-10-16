# SGLang KDA speculative verify and ReplaySSM fold (vendored)

Upstream: https://github.com/sgl-project/sglang
Commit: `c55790d3` (2026-10-10)
License: Apache-2.0, `LICENSE` in this directory.

SGLang serves Kimi-K3 with DSpark at TP8 with two kernels per KDA layer:
- `fused_kda_decode_mtp_dspark` (CuTe DSL) verifies a block of eight tokens per
  sequence (an anchor and seven drafts) from the committed state. It writes
  each step's output and its raw inputs to a ring, and leaves the state as it
  was.
- After acceptance, `kda_replayssm_exact_fold_kernel` (Triton) replays the
  kept steps into the state.

D's speculative round runs both, AOT-compiled here into the two cubins
`kernels.toml` pins:
- `k3_kda_verify+HEADS=12`
- `k3_kda_fold+HEADS=12`

## What is vendored

- `kda_decode_mtp.py`, from `python/sglang/kernels/ops/attention/`. Modified:
  - The torch host wrapper, its JIT cache and `_block_threads` are cut. The
    kernel and its launcher (`_run_kda_decode_mtp_dspark`) are upstream.
  - A sequence's rows are its eight consecutive ones (`bos = i_n * T_LOOP`),
    in place of `cu_seqlens`.
  - The per-step conv windows and the ring are addressed by the batch row
    `i_n`, in place of `intermediate_state_indices` and the state slot. Both
    are scratch of one round, not per-slot state.
  - The per-step snapshot path (`ht`, the non-ring mode) is removed.
- `kda_replayssm_spec_decode.py`, from
  `python/sglang/kernels/ops/attention/fla/`. Modified:
  - The torch wrappers are cut.
  - The rings are read at the batch row (`ring_idx = i_n`), in place of the
    state slot. The fold itself is upstream.

## What is ours

- `build.py` is the cubin recipe, run in an environment with
  nvidia-cutlass-dsl 4.8.0 and Triton 3.8.0 (SGLang's image). It needs no
  GPU. Every operand is a fixed view of a buffer the D manifest owns, so
  every layout is static and both kernels take bare pointers:

  ```sh
  HOME=/tmp python3 -B build.py --tp 8 --out <dir>
  ```

  Both cubins rebuild byte for byte.
- `../k3_kda_spec.cu` holds the two kernels around them: the forget path's
  `f_b` projection, which the verify reads as `g`, and the commit of the
  kept step's conv window into the line.

## ABI (TP8, 12 heads)

A member's KDA line is `k3_kda_core`'s:
- recurrent state f32 `[12][128 v][128 k]`, then
- the conv window bf16 `[3 stream q|k|v][3 tap][1536]`.

`line` is its byte size, 814,080.

The qkvg partial is the fused projection's f32 row of 6400 columns:
`q | k | v | g`, then beta's 12 columns and `f_a`'s 128.

The round's scratch has one row per sequence:
- conv windows bf16 `[8 step][3][3][1536]`;
- the ring:
  - raw v bf16 `[12][8][128]`
  - raw k bf16 `[12][8][128]`
  - log decay f32 `[12][8][128]`
  - beta f32 `[12][8]`

**`k3_kda_verify+HEADS=12`**: 25 pointers, in this order. Offsets are in
elements of the operand's type.

| # | Operand | Points at |
|---|---|---|
| 0 | h0 | line |
| 1–3 | x_q, x_k, x_v | partial + 0, 1536, 3072 |
| 4–6 | w_q, w_k, w_v | conv weights f32 `[3][4 tap][1536]`, + 0, 6144, 12288 |
| 7–9 | cs_q, cs_k, cs_v | line window, + 0, 4608, 9216 |
| 10 | A_log | |
| 11 | g | `f_b` output f32 `[rows, 1536]` |
| 12 | dt_bias | |
| 13 | beta | partial + 6144 |
| 14 | o | bf16 `[rows, 1536]` |
| 15–17 | step windows q, k, v | + 0, 4608, 9216 |
| 18–21 | ring | raw v, raw k, log decay, beta |
| 22 | output gate | partial + 4608 |
| 23 | gamma_o | |
| 24 | line index | i32 `[seqs]` |

Launch:
- grid `(12, seqs)`, block 256;
- 95,008 B of dynamic shared memory;
- a sequence's rows are `8n .. 8n + 7`.

The kernel waits on and releases programmatic dependents, so it also runs
without PDL.

**`k3_kda_fold+HEADS=12`**: the state slot stride, the ring row stride and
`MAX_CACHE_LEN = 8` are compiled in.

| # | Operand |
|---|---|
| 0 | line |
| 1–4 | ring: raw v, raw k, log decay, beta |
| 5 | line index |
| 6 | kept steps i32 `[seqs]` |
| 7–8 | unread (`HAS_TRACK = 0`) |
| 9–10 | Triton's scratch pointers, 0 |

Launch:
- grid `(4, seqs, 12)`, block 32;
- 128 B of dynamic shared memory.

`harness/kda_spec.cu` checks the round against eight steps of
`k3_kda_core` (`harness/run.sh spec`).
