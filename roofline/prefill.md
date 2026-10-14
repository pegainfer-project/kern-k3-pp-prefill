# Prefill roofline: PP8 stage 0 at production shapes

Every kernel full Kimi-K3 prefill runs on kern, measured at the shapes the service sees, priced against a
floor built from measured ceilings, and explained where it misses. Measured 2026-10-14 on one GB300
(152 SMs, 1400 W board cap), kern master `304c0d4`, the PP8 capacity deployment's stage 0 manifest.

**Summary.**

- **The stage runs at 1.29× its floor.** One 8192-row chunk over an empty cache takes 141.4 ms against a
  109.8 ms floor. The last chunk of a 32k prompt is 164.6 ms (1.26×); four 2k prompts packed are 140.7 ms (1.31×).
- **Dense bf16 GEMMs are half the stage and sit on the roof (1.04–1.05×).** The tensor pipe is 91–99% active.
  What is left there is precision: the same FLOPs at the measured fp8 ceiling floor at 37.9 ms instead of 66.1 ms per chunk.
- **MoE is 31–38% of the stage at 1.53–1.59×.** Most of the gap is the 128-row token tile against a
  heavy-tailed expert histogram: 29% of the executed MMA rows are padding, at full price.
- **KDA is 6–9% at 2.2–2.6×.** The bytes are right, but prepare and the recurrence are separate kernels with a
  1.36 GB workspace round trip between them, and the recurrence is a serial `mma.sync` chain.
- **MLA attention grows with context: 3.7% over an empty cache, 18% at a 24k prefix.** The FMHA is compute-bound at 71–80% of the bf16 ceiling.
- **One bug, off the measured shapes.** The absorbed short path (≤128 rows) has an unsigned
  underflow that gives every row 16 KV splits instead of 1.

## Setup

**What runs.** PP8 stage 0, layers 0–11:
- the embedding;
- layer 0's dense FFN, then MoE in layers 1–11 (896 experts, top-16, on the 3584-wide latent);
- nine KDA layers and three MLA layers (3, 7, 11).

Stage 0 is used rather than a middle stage because it starts from token ids. A middle stage benched alone gets zeroed
arrivals: every router logit is 0, every token picks the same 16 experts, and the MoE numbers are meaningless.

**Shapes.** One `kern bench` workload:

```toml
samples = 128   # 32 bracketed samples per call
[[sweep]]
rows = [8192]
context = [0, 24576]
[[sweep]]
groups = [4]
rows = [2048]
context = [0]
```

| shape | what it is |
|---|---|
| `g1-r8192-kv0` | one 8k chunk over an empty cache |
| `g1-r8192-kv24576` | the last 8k chunk of a 32k prompt |
| `g4-r2048-kv0` | four 2k prompts packed into one chunk |

**Measured.** `kern bench` brackets every call inside the CUDA graph. A call's time is its bracket's p50 at the program's own clock, power and cache state.

- Shares use one denominator: the sum of the shape's call brackets, 141.39 / 164.56 / 140.68 ms.
- The un-bracketed graph is 0.2–0.5% shorter; a bracket breaks PDL overlap.
- An nsys program span is ~1% longer, because it includes the inter-launch gaps.

**Launches.** The manifest declares 200 launches. 180 fire at these shapes; the other 20 are gated off by `when` on the call's row count:

- 11 small-batch `moe_route` variants;
- 3 `mla_fmha` split launches (129–512 rows);
- 3 `mla_absorb_short` and 3 `mla_attn_short` launches (≤128 rows).

**Floor.** floor = max(FLOP / peak, bytes / HBM).
- FLOPs are the useful work.
- Bytes are the minimum HBM traffic: every input read once, every output written once, weights included.

**Hardware counters.** ncu in a single pass over the same program, so the counters come from the
program's own buffers; ncu serializes launches, so its clocks run higher than in the program.

**Ceilings (measured on the same GPU).**

| key | measured | how | datasheet |
|---|---:|---|---:|
| HBM | 7332 GB/s | 16 GiB streaming read, 2070 MHz | 8000 |
| HBM copy | 6685 GB/s | 16 GiB streaming copy (read + write counted) | |
| L2 | 18.9 TB/s | 48 MiB L2-resident sweeps | |
| bf16 | 1868 TFLOP/s | the best cuBLAS GEMM in the program (`qkvg`, 8192 × 49408 × 7168); alone, sustained: 1764 | 2500 |
| fp8 | 3257 TFLOP/s | cuBLASLt e4m3, 16384³, 1392 MHz | 5000 |
| mxf8f6f4 | 3058 TFLOP/s | cuBLASLt block-scaled e4m3 (the `kind::mxf8f6f4` MMA), 1291 MHz | 5000 |
| mma.sync | 604.5 TFLOP/s | m16n8k16 bf16 throughput, 1957 MHz (prices FlashKDA) | |
| FFMA | 78.5 TFLOP/s | CUDA-core FMA throughput, ~2.02 GHz | |

**Why bf16 is an in-program number.**
- At the 1400 W cap a GEMM that runs alone for seconds settles at ~1380–1480 MHz.
- Inside the program the memory-bound kernels between GEMMs leave power headroom, and the board runs 1450–1590 MHz at 1375 W.
- 17 of the 33 (shape, GEMM) groups beat the sustained 1764, so the ceiling is the faster of the two.
- Per clock the two agree with the datasheet: 1764 / 1477 MHz = 1.19 TFLOP/s/MHz; 2500 / 2070 = 1.21.
- **Read the GEMMs' 1.0× as "at the best rate cuBLAS reaches on this board", not "at the datasheet".** The gap to 2.5 PF is the power-capped clock.

**Clock.** The whole stage runs at the power cap, at ~1.45–1.49 GHz in the issue-bound kernels. The same kernels run at 1.9–2.1 GHz serialized under ncu. Under this cap, fewer instructions and fewer bytes per row pay off; more parallelism does not.

## Where the time goes

Groups (ms per chunk, share of the stage, measured / floor):

| group | g1-r8192-kv0 | g1-r8192-kv24576 | g4-r2048-kv0 |
|---|---:|---:|---:|
| dense GEMM (cuBLAS bf16, incl. MLA `q_b` / `expand`) | 69.73 (49.3%, 1.05×) | 70.20 (42.7%, 1.04×) | 69.69 (49.5%, 1.05×) |
| MoE (route, FC1, FC2, finalize) | 52.37 (37.0%, 1.57×) | 50.87 (30.9%, 1.53×) | 52.84 (37.6%, 1.59×) |
| MLA attention (FMHA + glue) | 5.17 (3.7%, 1.44×) | 29.64 (18.0%, 1.26×) | 2.31 (1.6%, 2.08×) |
| KDA (span_gather, prepare, recurrence) | 10.35 (7.3%, 2.22×) | 10.13 (6.2%, 2.17×) | 12.07 (8.6%, 2.56×) |
| residual / norm glue | 3.78 (2.7%, 1.76×) | 3.72 (2.3%, 1.73×) | 3.77 (2.7%, 1.76×) |
| **stage** | **141.39** (floor 109.82, **1.29×**) | **164.56** (floor 131.10, **1.26×**) | **140.68** (floor 107.38, **1.31×**) |

Every call site, sorted by time at `g1-r8192-kv0`. Each cell is µs per call, measured / floor, share of the stage.

| label | op | kernel | calls | bound | g1-r8192-kv0 | g1-r8192-kv24576 | g4-r2048-kv0 |
|---|---|---|---:|---|---:|---:|---:|
| `fc1` | `moe_fc1` | TRT-LLM gen bmm mxf8×mxf4, SiTU-GLU | 11 | mxf8f6f4 | 3087.6, 1.64×, 24.0% | 2994.2, 1.59×, 20.0% | 3117.0, 1.65×, 24.4% |
| `qkvg` | `gemm_bf16` | cublaslt_bf16_tn | 9 | bf16 | 3185.6, 1.03×, 20.3% | 3131.6, 1.01×, 17.1% | 3180.4, 1.02×, 20.3% |
| `fc2` | `moe_fc2` | TRT-LLM gen bmm mxf8×mxf4, bf16 out | 11 | mxf8f6f4 | 1402.0, 1.49×, 10.9% | 1361.7, 1.44×, 9.1% | 1415.7, 1.50×, 11.1% |
| `front` | `gemm_bf16` | cublaslt_bf16_tn | 11 | bf16 | 1054.4, 1.06×, 8.2% | 1040.9, 1.04×, 7.0% | 1052.5, 1.05×, 8.2% |
| `o_proj` | `gemm_bf16` | cublaslt_bf16_tn | 12 | bf16 | 837.9, 1.08×, 7.1% | 826.2, 1.07×, 6.0% | 841.1, 1.09×, 7.2% |
| `span_kda` | `flash_kda` | flash_kda prepare + k3_kda_rec | 9 | hbm | 869.1, 2.47×, 5.5% | 850.2, 2.41×, 4.6% | 1058.1, 2.96×, 6.8% |
| `back` | `gemm_bf16` | cublaslt_bf16_tn | 11 | bf16 | 646.8, 1.06×, 5.0% | 639.2, 1.05×, 4.3% | 647.4, 1.06×, 5.1% |
| `attn` | `mla_fmha` | trtllm_fmha_ctx_h192_v128 | 3 | bf16 | 1546.3, 1.40×, 3.3% | 9696.5, 1.26×, 17.7% | 587.8, 2.13×, 1.3% |
| `wgu` | `gemm_bf16` | cublaslt_bf16_tn | 1 | bf16 | 4372.0, 1.03×, 3.1% | 4307.7, 1.01×, 2.6% | 4359.7, 1.03×, 3.1% |
| `wfu` | `gemm_bf16` | cublaslt_bf16_tn | 3 | bf16 | 947.4, 1.05×, 2.0% | 936.8, 1.03×, 1.7% | 948.6, 1.05×, 2.0% |
| `span_gather` | `span_gather` | k3_span_gather | 9 | hbm | 280.8, 1.69×, 1.8% | 274.9, 1.66×, 1.5% | 282.9, 1.70×, 1.8% |
| `w_dn` | `gemm_bf16` | cublaslt_bf16_tn | 1 | bf16 | 2253.7, 1.06×, 1.6% | 2172.4, 1.02×, 1.3% | 2244.5, 1.06×, 1.6% |
| `res_mlp` | `land_add_attnres_rms_bf16` | k3_prefill_glue | 12 | hbm | 145.2, 1.84×, 1.2% | 143.4, 1.82×, 1.0% | 145.2, 1.84×, 1.2% |
| `finalize` | `moe_finalize_rms_ld` | k3_moe_route | 11 | hbm | 158.0, 1.16×, 1.2% | 157.8, 1.16×, 1.1% | 157.9, 1.16×, 1.2% |
| `res_in` | `land_add2_attnres_rms` | k3_prefill_glue | 11 | hbm | 145.8, 1.82×, 1.1% | 143.2, 1.79×, 1.0% | 145.6, 1.82×, 1.1% |
| `route` | `moe_route` | k3_moe_route | 11 | hbm | 112.8, 1.95×, 0.9% | 111.3, 1.92×, 0.7% | 113.1, 1.95×, 0.9% |
| `router` | `gemm_f32` | cublas_bf16_tn_f32 | 11 | bf16 | 81.9, 1.45×, 0.6% | 81.0, 1.44×, 0.5% | 81.9, 1.45×, 0.6% |
| `q_b` | `gemm_bf16` | cublaslt_bf16_tn | 3 | bf16 | 271.7, 1.09×, 0.6% | 269.1, 1.08×, 0.5% | 271.8, 1.09×, 0.6% |
| `expand` | `gemm_bf16_long` | cublaslt_bf16_tn | 3 | bf16 | 196.7, 1.19×, 0.4% | 709.2, 1.12×, 1.3% | 196.0, 1.19×, 0.4% |
| `span_g` | `gemm_bf16` | cublaslt_bf16_tn | 9 | hbm | 57.3, 2.03×, 0.4% | 56.0, 1.99×, 0.3% | 57.3, 2.03×, 0.4% |
| `gate` | `mla_gate` | k3_mla_glue | 3 | hbm | 129.4, 1.57×, 0.3% | 129.6, 1.57×, 0.2% | 129.6, 1.57×, 0.3% |
| `situ` | `situ_n33792` | k3_prefill_glue | 1 | hbm | 304.6, 1.34×, 0.2% | 294.4, 1.30×, 0.2% | 304.1, 1.34×, 0.2% |
| `prep` | `mla_prep_gather` | k3_mla_glue | 3 | hbm | 47.8, 4.42×, 0.1% | 55.5, 2.99×, 0.1% | 51.8, 4.79×, 0.1% |
| `res_in` | `embed_rms_setup` | k3_prefill_glue | 1 | hbm | 74.9, 1.55×, 0.1% | 74.5, 1.54×, 0.0% | 74.3, 1.54×, 0.1% |
| `hidden` | `land_add2` | k3_prefill_glue | 1 | hbm | 53.2, 1.11×, 0.0% | 53.2, 1.11×, 0.0% | 52.7, 1.10×, 0.0% |

## Headroom, ranked

Savings in ms per 8k chunk at `g1-r8192-kv0` unless noted. Estimates are marked; everything else is measured or
is a floor difference.

| # | change | saves | kind |
|---:|---|---:|---|
| 1 | fp8 (e4m3, block- or channel-scaled) for the K = 7168 projections: qkvg, front, back, wfu, o_proj, wgu, w_dn | ≤ 28 ms (20%): GEMM floor 66.1 → 37.9 | an accuracy decision first, then a kernel swap |
| 2 | MoE padding: split experts by row count (≤ 64 rows on a tile-64 kernel); structurally, more rows per expert (16k chunks) | 2.5 ms (split, estimate); all padding is ≤ 14.4 ms (10%) | route emits two CTA tables / a pipeline decision |
| 3 | fp8 context FMHA (TRT-LLM gen ships e4m3 variants of the same family) | at a 24k prefix, the floor gap is 3 × 3.3 = 9.9 ms; grows with context | `kern test` agreement |
| 4 | fuse FlashKDA prepare into the recurrence (no 2 × 680 MB workspace round trip) | ≤ 5.2 ms at the `mma.sync` chain floor; more on tcgen05 UMMA with a chunk-64 recurrence | kernel rewrite |
| 5 | MoE per-tile residual: FC1's SiTU-GLU + mxfp8 epilogue not hidden behind a 14-step mainloop | ≤ 5.3 ms | hard |
| 6 | MLA `expand` re-expands the whole cached context every chunk | 1.5 ms at a 24k prefix, linear in context | memory trade (60 KiB/token/layer K/V) or the absorbed path |
| 7 | finalize: fold the deterministic top-16 combine + RMS into `back`'s A load | ≤ 1.7 ms | fusion |
| 8 | K1b / K1d: 2–4 rows in flight per CTA | ~1.4 ms | kernel tune |
| 9 | span_g: produce the gate inside FlashKDA prepare | ~0.8 ms | fusion |
| 10 | route: the table phase after the grid barrier | ≤ 0.6 ms | kernel tune |
| 11 | g4: give the KDA gate CTAs the lowest block indices (the 201 MB `raw` stays in L2) | ~0.5 ms (estimate) | kernel tune |
| 12 | router: fold its 896 rows into `front` | ~0.3 ms | needs bf16 logits or a split epilogue |
| 13 | g4: size the FMHA grid by the longest sequence, not the call's rows | 0.2 ms (−11% FMHA, measured) | needs a call var |
| — | **bug**, ≤128-row calls: `nsm / 2 - B` underflows in [`k3_mla_glue.cu`](../source/k3_mla_glue.cu) | absorbed layer 1122 µs → ~545 µs at 128 rows over 24.6k (estimate, with item below) | one-line fix |

Without a precision change (items 2 and 4–13), the kernels have roughly 15–30 ms of a 141 ms chunk to give. Most
of it is in MoE padding, the KDA fusion and FC1's per-tile residual. None of it is in the dense GEMMs.

## Dense GEMMs (cuBLAS)

There are 74 cuBLAS GEMMs per chunk:
- 63 `cublaslt_bf16_tn` with bf16 out: 60 projections and the 3 MLA kv_b expansions;
- 11 f32-out routers on `cublasGemmEx`.

Every one is C[m, n] = A[m, k] · W[n, k]ᵀ with an f32 accumulator. The cuBLASLt ones take the heuristic's top-1 algorithm with a 32 MiB workspace. ncu sees exactly one `nvjet_sm103_*` tcgen05 kernel per call, never a split-K reduction.

| label | where | n × k | what |
|---|---|---|---|
| `qkvg` | 9 KDA layers | 49408 × 7168 | fused q, k, v, output gate (4 × 12288) and the β / gate-rank heads |
| `span_g` | 9 KDA layers | 12288 × 128 | the forget gate's rank-128 up-projection |
| `o_proj` | 12 layers | 7168 × 12288 | attention output |
| `wgu`, `w_dn` | layer 0 | 67584 × 7168, 7168 × 33792 | dense FFN gate+up, down |
| `router` | 11 MoE layers | 896 × 7168, f32 out | expert logits |
| `front` | 11 MoE layers | 15872 × 7168 | routed-latent down (3584) + shared expert gate/up (12288) |
| `back` | 11 MoE layers | 7168 × 9728 | routed-latent up + shared expert down, over the concatenated input |
| `wfu` | 3 MLA layers | 14400 × 7168 | q_a + kv_a + output gate |
| `q_b` | 3 MLA layers | 18432 × 1536 | q up-projection, 96 heads × 192 |
| `expand` | 3 MLA layers | 30720 × 576, m = ⌈ctx/128⌉·128 + 512 | kv_b expansion of the whole cached + new latent |

**The model.**
- FLOP = 2mnk as executed. For `expand` this includes re-expanding the context, which is why it grows 3.6× from kv0 to kv24576.
- Bytes = A + W + C once.
- The ridge is 1868 / 7.33 = 255 FLOP/B. Everything is compute-bound except `span_g` (125 FLOP/B).

**ncu evidence** (`g1-r8192-kv0`, medians over layers):

| label | ncu µs | SM MHz | tensor pipe | DRAM / min | DRAM % of peak |
|---|---:|---:|---:|---:|---:|
| qkvg | 2898 | 1629 | 99% | 3.8× | 27% |
| wgu | 4046 | 1597 | 99% | 3.8× | 26% |
| front | 937 | 1659 | 97% | 3.4× | 28% |
| wfu | 850 | 1683 | 96% | 2.9× | 24% |
| back | 596 | 1649 | 93% | 2.9× | 26% |
| o_proj | 728 | 1744 | 91% | 2.7× | 23% |
| w_dn | 2039 | 1686 | 93% | 3.7× | 26% |
| q_b | 236 | 1720 | 92% | 1.1× | 23% |
| expand | 172 | 1742 | 84% | 1.2× | 50% |
| router | 71 | 1840 | 69% | 1.2× | 34% |
| span_g | 44 | 1939 | 27% | 0.94× | 56% |

- **The large projections are at the MMA roof per clock.** Their re-read bytes (2.7–3.8× the minimum, A and W
  re-read across 128 × 256 tiles) cost nothing at 23–28% of DRAM bandwidth.
- **Their time is set by the power-capped clock, not the kernel.**
  - Alone at the cap they are 3–8% *slower* than in the program.
  - The best of cuBLAS's first 8 candidates beats top-1 by only 0–3.5%, mostly via a tile that draws less power.
- **The router is the exception (1.45×).** `cublasGemmEx` with f32 output picks a 1-CTA `tss_128x192` tile:
  301 CTAs in two waves over 152 SMs, no 2-CTA cluster, tensor pipe 69% active.
- **`span_g` is memory-bound, and its bytes are already the minimum.** 201 MB of its 207 MB is writing the
  8192 × 12288 gate, which FlashKDA reads once. Fusing it into prepare removes both.
- **`expand` at a 24k prefix** is m = 33280, 709 µs per call. It re-expands the full context every chunk.
  - Expanding only the new rows into a persistent per-head K/V cache makes it the kv0 cost (197 µs) at every
    depth, for 60 KiB of cache per token per MLA layer instead of 1.1 KiB of latent.
  - Attending over the latent (the decode path) removes it entirely.

## MoE

Four launches per MoE layer:

| launch | kernel | what |
|---|---|---|
| `route` | `k3_moe_route` | top-16 of sigmoid(logits) + bias, latent quantized to mxfp8, shared-expert SiTU, TRT-LLM gen's routing tables; one launch with a grid barrier between the per-row and the table phase |
| `fc1` | TRT-LLM gen `bmm_MxE4m3_MxE2m1MxE4m3_…_siTuGlu` | mxfp8 latent × mxfp4 w13, K = 3584, N = 2 × 3072, SiTU-GLU epilogue re-quantized to mxfp8, rows gathered with LDGSTS |
| `fc2` | TRT-LLM gen `bmm_Bfloat16_MxE2m1MxE4m3_…` | mxfp8 × mxfp4 w2, K = 3072, N = 3584, bf16 out in permuted order |
| `finalize` | `k3_moe_route` | each token's 16 rows weighted, summed and RMS-normed |

**The histogram decides the cost, so it was measured.**
- A rewritten manifest has every layer's `route` write its top-k ids into its own slice of a carry buffer.
- A patched `kern bench` reads that buffer back. Its timings reproduce the main run within 0.7%.

Means over the 11 MoE layers; each layer routes 131,072 rows (146 per expert if routing were uniform):

| corpus | active experts | rows / active expert p10 / p50 / p90 | hottest expert | experts < 128 rows | padded share at tile 128 |
|---|---:|---|---:|---:|---:|
| `kern bench` prose (behind every number above) | 736 | 5 / 100 / 394 | 6794 rows (83% of tokens) | 423 | **29.1%** |
| diverse chat prose, 1M tokens | 893 | 32 / 116 / 274 | 7196 rows (88%) | 489 | 31.3% |
| uniform | 896 | 146 | 147 | 0 | 42.9% |

- Every layer has a hot expert on both corpora: in the hottest layer one expert takes 70–88% of the tokens.
- On real prose FC1 is 3.3% and FC2 3.5% slower, and the stage takes 146.5 ms instead of 142.5 ms, because more experts are read.

**Model.**
- FC1 is 5.77 TFLOP and FC2 2.89 TFLOP of useful work per layer at 8192 rows.
- Bytes: packed fp4 weights + UE8M0 scales for active experts only (11.7 MB and 5.85 MB per expert), the latent once per token, outputs once per routed row.
- Both GEMMs are compute-bound at `mxf8f6f4`; route and finalize are HBM-bound.

**FC1 / FC2: time is a straight line in executed 128-row tiles** over the 33 (shape, layer) points:

| kernel | µs per tile | at the ceiling's rate | intercept | r² |
|---|---:|---:|---:|---:|
| FC1 | 2.074 | 1.843 | 76 µs | 0.961 |
| FC2 | 0.991 | 0.922 | −35 µs | 0.977 |

- **FC1's 1.65× = 1.41 (padding) × 1.13 (per-tile rate) × 1.03 (fixed cost); FC2's 1.50× = 1.41 × 1.07.**
  A padded tile costs the same as a full one.
- **What the gap is not:**
  - not memory: DRAM is 1.00–1.11× the minimum, at 47–62% of the HBM ceiling;
  - not imbalance or wave tails: about 600 waves of one CTA per SM, every SM busy 100% of the kernel, the busiest ending 3–4% after the mean;
  - not a starved tensor pipe: 78% (FC1) / 82% (FC2) active at 1670–1750 MHz.

| kernel | ncu µs | DRAM / min | DRAM TB/s | tensor pipe | executed MMA PF/s |
|---|---:|---:|---:|---:|---:|
| FC1 | 2925 | 1.11 | 3.54 | 78% | 2.80 |
| FC2 | 1327 | 1.05 | 4.57 | 82% | 3.09 |
| route | 87 | 0.95 | 4.63 | — | — |
| finalize | 151 | 1.01 | 6.66 (91%) | — | — |

**Headroom.**
- **Smaller tiles alone lose.** Padding falls to 16% at tile 64, but an earlier sweep measured tile 64 25% slower per layer at 8k rows (1.45× the per-row cost).
- **Splitting by row count gains about 2.5 ms per stage.** Experts with ≤ 64 rows (238–286 per layer) go to a tile-64 kernel, the rest stay on tile 128; padding falls to 22%.
- **The structural lever is more rows per expert.** In the same earlier sweep, MoE time per token fell 17% from 8k to 16k chunks.
- **finalize is at the floor (1.16×, 91% of HBM).** Its cost is FC2's 940 MB output round trip; only a deterministic fusion removes it.
- **route is 1.95×.** A bandwidth phase, then a latency-bound table phase behind the grid barrier; worth ≤ 0.6 ms per stage.

## MLA attention

Each MLA layer issues `prep` (`k3_mla_glue`), `expand` and `q_b` (cuBLAS, above), then one attention path, then `gate`
(`k3_mla_glue`). The path is chosen by the call's row count:

- **≤ 128 rows: the absorbed path.** `absorb` (`k3_mla_absorb`) and FlashInfer's `mla_decode_h96_p64` over the latent.
- **129–512 rows: the split FMHA launch.** The heaviest sequence is split into three causal pieces over `seqs + 2`
  FMHA sequences, so a 1–2 Q-tile call still fills 152 SMs; `gate` merges the pieces by their softmax statistics.
- **≥ 513 rows: the plain context FMHA.** `trtllm_fmha_ctx_h192_v128`, Q256 × KV128, causal, persistent.

At all three measured shapes only the plain FMHA fires.

**Model.**
- FLOP = pairs · 96 heads · 2 · (192 + 128), where pairs = Σ_seq (L·P + L(L+1)/2): 33.6 M, 234.9 M and 8.39 M.
- Bytes = q + K|V over the context (61,440 B per row) + o.
- **The causal accounting is exact.** HMMA-active cycles × 8192 FLOP/clk/SM × 152 reproduce the model's
  tile-executed FLOPs to 0.0–0.2% at every shape. So no key crosses a packed sequence boundary, and no tile is skipped or double-counted.

| shape | FMHA µs | floor µs | × | PF/s in program | % of bf16 ceiling | tensor pipe (ncu) | DRAM / min |
|---|---:|---:|---:|---:|---:|---:|---:|
| g1-r8192-kv0 | 1546.3 | 1103.8 | 1.40 | 1.33 | 71% | 70.0% | 1.03 |
| g1-r8192-kv24576 | 9696.5 | 7725.6 | 1.26 | 1.49 | 80% | 75.7% | 1.04 |
| g4-r2048-kv0 | 587.8 | 276.0 | 2.13 | 0.88 | 47% | 49.7% | 1.03 |

**The g1 shapes are compute-bound, and no single unit is saturated.**
- MUFU is 28–31% busy and its instruction count matches the exponentials, so none are emulated. FMA is 17%.
- The missing 24–30% is the softmax → MMA dependency inside a CTA.

**g4 is 2.13× for three reasons:**
1. 12% of the executed MMA is masked diagonal work: 256-row Q tiles on 2048-row triangles.
2. The grid is sized by the call's rows, `ceil(8192/256) = 32` CTAs per head per sequence, while each sequence has 8 Q tiles. 3/4 of the CTAs find no rows. A per-sequence grid measured 592.8 → 526.2 µs in the program (−11%).
3. The mean KV loop is 9 tiles, so per-tile fixed costs weigh about four times more than at kv0.

**The lever is precision.** At a 24k prefix the fp8 FLOP floor is 4.43 ms against 7.73 ms in bf16. Attention's
share grows with context: an earlier stage study put the FMHA at 57% of the stage at a 200k prefix.

**Glue.**
- `gate` moves 0.96× its minimum at 5.6 TB/s. Its in-program 1.57× is the power cap; it could only go away by fusing into a prebuilt FMHA epilogue or o_proj prologue.
- `prep` (3.0–4.8×) is launch-granularity-bound: one row per 512-thread block, ~27 waves. 4–8 rows per block would save ~30 µs per layer.

**The absorbed short path, measured off the main shapes at 128 rows over a 24.6k prefix.** It costs 1122 µs per
layer, more than the expanded path's 967 µs at twice the rows. By FLOPs it should win below ~240 rows. Three causes:

1. **The split plan underflows.**
   - In [`k3_mla_glue.cu`](../source/k3_mla_glue.cu), `long long budget = nsm / 2 - B` is computed with
     `nsm` unsigned (`%nsmid` reads 152 here). For B > 76 rows it wraps to ~4.3 × 10⁹, so every row gets the maximum 16 KV splits instead of 1.
   - Evidence: the decode kernel moves 0.39 GB against a 0.068 GB minimum, and the merge (`v_up` + gate) moves 10.8× its minimum through DRAM.
   - The decode step's split plan in `k3_ar_fused.cu` already computes it as `long long`.
2. **`absorb` and `v_up` are CUDA-core FMA kernels tuned for decode's ≤ 64 rows.** At 128 rows they run at 19 TFLOP/s, 24% of the FFMA ceiling. As tensor-core batched GEMMs each is a ~5 µs bandwidth problem.
3. **The decode kernel pads 96 heads to 128 MMA rows.** That is 1.33× of its 1.80×.

With 1–2 fixed, the layer is an estimated ~545 µs. That matters for agent-style follow-up turns: short extends over a long cached prefix.

## KDA

Nine KDA layers, three non-GEMM launches each:

1. **`span_gather`** (`k3_span_gather`): the 4-tap causal conv + SiLU of q | k | v, transposes of β and the gate flow, and FlashKDA's tile prefix.
2. **FlashKDA prepare** (`flash_kda_d128_varlen`, chunk 16, d = 128). Per (tile, head): norms, gate cumsum, decayed q / k, the 16×16 inverse. It writes a 13,824 B workspace.
3. **`k3_kda_rec`**: the inter-chunk delta-rule recurrence.
   - One CTA per (sequence, head) carries an f32 state through the sequence's tiles in order.
   - 56 more CTAs compute the output gate.

Per-launch times below come from an nsys graph-node trace, because `kern bench` brackets the two FlashKDA launches as one call.

| kernel | g1-kv0 µs | g1-kv24576 µs | g4 µs | HBM floor µs | × (g1) | ncu evidence (g1, isolated) |
|---|---:|---:|---:|---:|---:|---|
| span_gather | 278.5 | 274.8 | 280.2 | 165.9 | 1.68 | 210 µs, 5.70 TB/s, issue 59%, MUFU 43% |
| prepare | 350.2 | 346.0 | 359.6 | 175.3 | 2.00 | 278 µs, 4.44 TB/s, issue 64%, top stall `barrier` |
| k3_kda_rec | 523.2 | 516.5 | 706.2 | 177.0 (g4 182.4) | 2.96 | 426 µs, HMMA 32%, occupancy 19%, DRAM 38% |

**The bytes are right.**
- ncu's DRAM reads match the byte model for every kernel (0.93–0.99).
- The one exception is the recurrence at g4 (1.27×). Its gate CTAs have the highest block indices and wait three recurrence waves for an SM, so the ungated 201 MB `raw` round-trips through HBM.

**The issue-bound kernels pay the clock.** They run 1.23–1.33× slower in the program (1.44–1.49 GHz) than isolated (1.93–2.09 GHz).

**The recurrence's real floor is its serial chain, not HBM.**
- A tile is 52 dependent m16n8k16 HMMAs per warp, so at least 832 cycles at the measured 20.2-cycle `mma.sync` latency.
- g1 runs 96 chains of 512 tiles on 96 SMs. That is 291 µs at the in-program clock, above the 177 µs HBM floor; the kernel takes 1.8× it.
- About half of each tile is the non-MMA serial work between the products: decay, the u / U steps, barriers.

**Headroom.**
1. **Fuse prepare into the recurrence.**
   - The workspace costs 2 × 680 MB of HBM traffic. As one kernel, flash_kda's HBM floor is 167 µs against 873 µs today.
   - Prepare does not depend on the state, so a producer warpgroup (or the 56 otherwise-idle SMs at g1) can run it ahead of the chain.
2. **Shorten the chain.**
   - No `mma.sync` design gets under ~184 µs here.
   - tcgen05 UMMA (measured bf16 peak 3.1× `mma.sync`) with a chunk-64 recurrence takes 4× fewer serial steps.
   - The cost is a 64×64 inverse and the precision of the longer intra-chunk decay.
3. **g4: schedule the gate with the chain.** `raw` then stays in L2 (−0.4 GB).
4. **span_gather: fewer instructions per element.** It is issue-bound under the cap, not HBM-bound. A cheaper SiLU (approximate exp + rcp) and vectorized tap loads would help.

## Residual and norm glue

| kernel | calls | µs | floor µs | × | why |
|---|---:|---:|---:|---:|---|
| K1b `land_add_attnres_rms` | 12 | 145 | 79 | 1.84 | one row per CTA; the reduction's dependent loads expose latency (`long_scoreboard`), DRAM 51–61% |
| K1d `land_add2_attnres_rms` | 11 | 146 | 80 | 1.82 | same |
| `situ` (layer 0) | 1 | 305 | 227 | 1.34 | isolated it reaches 97% of the copy ceiling; the in-program gap is the clock |
| `embed_rms` | 1 | 75 | 48 | 1.55 | token → row gather latency; 0.05% of the stage |
| `land_add2` | 1 | 53 | 48 | 1.11 | at the floor |

Keeping 2–4 rows in flight per CTA in K1b / K1d is worth ~1.4 ms per stage. Folding them into the neighbouring
GEMM epilogues removes the partial's 235 MB round trip.

## Method and caveats

- **Counters.**
  - ncu ran in a single pass per program, so `sm_pct` / `mem_pct` are max(issue, tensor, FMA) and max(DRAM, L2) from that pass; ncu's own SOL needs 6–9 replays, which a program this size can't afford.
  - ncu serializes launches at free clocks, so its µs are 4–25% under the in-program brackets. Read ncu for bytes and ratios, the brackets for time.
- **fp8 / mxf8f6f4 ceilings are sustained isolated values.** A floor priced with them may be 5–7% optimistic for a short MMA kernel between memory-bound neighbours (the in-program uplift bf16 shows).
- **mxf8 × mxf4 has no cuBLASLt reference.** cuBLASLt 13.7 returns NOT_SUPPORTED, so `mxf8f6f4` is measured with e4m3 on both sides: the same `kind::mxf8f6f4` instruction.
- **The pipeline is deterministic.** Re-running the CPU side (call table, models, roofline join) reproduces the tables byte for byte.
