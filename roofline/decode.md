# Decode roofline: the TP8 / EP8 / DCP8 step at production shapes

Every kernel of the full Kimi-K3 decode step on kern, measured at the A/B batch points, priced against a floor
built from measured ceilings, and explained where it misses.

**Setup.** Measured 2026-10-14:
- Hardware: eight GB300 over two 4-GPU trays of one NVL72 rack.
- Software: the D manifest of the PP8 capacity deployment (93 layers, TP8 / EP8 / DCP8, 64 sequences max), run by `k3_step`.

**Summary.**

- **Rank 0's busy time is 1.7–2.05× its floor.** That is 11.2 ms at 1 row and 24.7 ms at 64 rows over an 8k context. Decode is
  much further from its floor than prefill (1.26–1.31×), and the reasons are spread over more kernels.
- **Dense projections (21–42%) are weight streams at 1.4–1.5×.** Their bytes are exactly the minimum.
  Three that have their weights prefetched into L2 by the all-reduce before them run at 1.1–1.2×; the rest do not get that prefetch.
- **The routed MoE (17–41%) streams each touched expert at 98% of HBM bandwidth.** The gap is a fixed 6–12 µs per launch, plus the EP skew:
  the busiest rank touches more experts, and every rank waits for it.
- **The two all-reduces per layer (20–23%) are 2.3–4.5× their latency floor.** Part of that time is the EP skew landing in the MoE all-reduce's wait.
- **The benchmark's pseudo-random tokens under-report the MoE.** They touch two thirds of the experts real text does. At 64 rows, real prose
  adds 4.2–6.0 ms of MoE time per step (corpus-dependent) to the 24.7 ms measured here.

## Setup

**Shapes.** These are the A/B points: closed batches of 1, 8, 32 and 64 sequences at 8192 context, and 32 at 32768.
DCP8 spreads each sequence's KV over the eight ranks.

**Measured.**
- `k3_step --bench`: one CUDA graph per shape, 8 ranks, pseudo-random token ids.
- An nsys graph-node trace on ranks 0 and 4 gives each launch's busy time, the time it adds to the step: end − max(start, previous end).
- Times are rank 0's unless stated. Rank 4, on the other tray, is within ±4% per call.
- Shares use rank 0's summed busy time (11.22 / 13.64 / 19.15 / 24.67 / 19.81 ms). The nsys step span is 1% longer (11.37 / 13.79 / 19.34 / 24.81 / 20.11 ms) because it includes the idle gaps.
- `k3_step`'s own step p50 is 11.77 / 14.10 / 19.60 / 24.56 / 20.23 ms.

**Real routing.**
- Teacher-forced runs replay real text through the same graph and dump every MoE layer's picks.
- One is 64 essays of the public Paul Graham essays set, 1280 tokens each; the other is chat-style prose.
- Only the MoE depends on the tokens: the dense projections take the same time on either (4713 / 5158 / 5243 µs against 4686 / 5167 / 5219 µs at 8 / 32 / 64 rows).

**Floor.** floor = max(FLOP / peak, bytes / HBM, bytes over NVLink / NVLink bandwidth, latency).
- Bytes are the minimum HBM traffic: weights, KV and state read once, outputs written once.
- Every decode kernel here is HBM-, NVLink- or latency-bound; none is compute-bound.

**Ceilings (measured on these GPUs).**

| key | measured | how | datasheet |
|---|---:|---|---:|
| HBM | 7332 GB/s | 16 GiB streaming read | 8000 |
| HBM copy | 6685 GB/s | 16 GiB streaming copy | |
| NVLink | 715 GB/s | SM stores across the trays (714 within a tray) | 900 / dir |
| NVLink all-to-all | 700 GB/s | 8 GPUs at once | |
| NVLink one-way flag | 2.09 µs | across the trays | |
| empty 8-rank one-shot exchange | 3.90 µs | `k3_ar_fused`'s Lamport protocol with no payload, kernel time | |
| one-shot all-reduce, 1 × 7168 bf16 | 6.0 µs | per call in a graph | |

One-shot all-reduce per call, by rows and width 7168 / 10752: 7.3 / 10.2 µs at 8 rows, 12.0 / 16.0 µs at 32 rows,
18.4 / 26.2 µs at 64 rows.

## Where the time goes

Groups, ms per step on rank 0 (share of busy time, measured / floor):

| group | 1 × 8k | 8 × 8k | 32 × 8k | 64 × 8k | 32 × 32k |
|---|---:|---:|---:|---:|---:|
| MoE (front, FC1, FC2) | 1.89 (17%, 2.7×) | 3.75 (27%, 1.5×) | 7.31 (38%, 1.3×) | 10.10 (41%, 1.2×) | 7.50 (38%, 1.3×) |
| dense GEMM (cuBLAS) + head | 4.67 (42%, 1.4×) | 4.70 (34%, 1.4×) | 5.20 (27%, 1.5×) | 5.28 (21%, 1.5×) | 5.19 (26%, 1.5×) |
| all-reduce + residual rows (`k3_ar_fused`, 186 exchanges) | 2.60 (23%, 2.3×) | 3.15 (23%, 2.8×) | 3.90 (20%, 3.5×) | 4.97 (20%, 4.5×) | 4.00 (20%, 3.6×) |
| residual rows, no exchange (92 closing adds) | 0.66 (6%, 294×) | 0.61 (4%, 43×) | 0.64 (3%, 11×) | 0.68 (3%, 6.1×) | 0.63 (3%, 11×) |
| KDA (`k3_kda_core`, 69 layers) | 0.66 (6%, 33×) | 0.66 (5%, 5.1×) | 0.84 (4%, 1.7×) | 1.57 (6%, 1.6×) | 0.84 (4%, 1.7×) |
| MLA (prep, absorb, attention, 24 layers) | 0.45 (4%, 8.7×) | 0.48 (4%, 4.5×) | 0.92 (5%, 3.8×) | 1.48 (6%, 3.7×) | 1.31 (7%, 2.1×) |
| DCP exchange | 0.29 (3%, 2.0×) | 0.29 (2%, 2.0×) | 0.34 (2%, 2.4×) | 0.59 (2%, 3.1×) | 0.35 (2%, 2.4×) |
| **busy, rank 0** | **11.22** (floor 5.48, **2.05×**) | **13.64** (floor 7.40, **1.84×**) | **19.15** (floor 11.18, **1.71×**) | **24.67** (floor 14.58, **1.69×**) | **19.81** (floor 11.78, **1.68×**) |

**Under real routing** (rank 0, teacher-forced; the context is shorter, which only touches MLA and DCP):

| rows | MoE, bench | MoE, prose | step busy, bench | step busy, prose |
|---:|---:|---:|---:|---:|
| 8 | 3.75 ms | 4.61 ms | 13.7 ms | 14.7 ms |
| 64 | 10.10 ms | 14.35 ms | 24.7 ms | 29.5–29.9 ms |

**Every call site** that costs at least 20 µs per step at 64 rows. Each cell is µs per call and measured / floor.

| call | op | kernel | n | bound | 1 × 8k | 8 × 8k | 32 × 8k | 64 × 8k | 32 × 32k |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `fc1` | `moe_fc1` | TRT-LLM gen bmm, FC1 SiTU-GLU | 92 | hbm | 8.7, 2.9× | 23.1, 1.5× | 49.7, 1.3× | 70.4, 1.2× | 51.1, 1.3× |
| `reduce_mlp` | `ar_finalize_rms` | k3_ar_fused | 92 | latency | 15.5, 2.6× | 21.5, 3.6× | 27.5, 4.6× | 33.6, 5.6× | 28.5, 4.8× |
| `fc2` | `moe_fc2` | TRT-LLM gen bmm, FC2 bf16 out | 92 | hbm | 5.3, 3.6× | 11.2, 1.4× | 23.2, 1.2× | 32.7, 1.1× | 24.0, 1.2× |
| `res_mlp` | `ar_attnres_rms` | k3_ar_fused | 93 | latency | 12.5, 2.1× | 12.5, 2.1× | 14.6, 2.4× | 20.0, 3.3× | 14.6, 2.4× |
| `front` | `gemm_f32` | cublas_bf16_tn_f32 | 92 | hbm | 13.8, 1.2× | 13.6, 1.1× | 16.0, 1.3× | 17.5, 1.4× | 16.0, 1.3× |
| `kda_core` | `kda_core` | k3_kda_core | 69 | hbm | 9.5, 33× | 9.6, 5.1× | 12.2, 1.7× | 22.8, 1.6× | 12.2, 1.7× |
| `qkvg` | `gemm_f32` | cublas_bf16_tn_f32 | 69 | hbm | 17.9, 1.4× | 17.8, 1.4× | 19.1, 1.5× | 18.4, 1.4× | 19.0, 1.5× |
| `attn` | `mla_attn` | mla_decode_h96_p64 | 24 | hbm | 8.8, 21× | 10.5, 4.2× | 27.2, 3.7× | 48.1, 3.7× | 43.7, 1.9× |
| `lat_up` | `gemm_bf16` | cublaslt_bf16_tn | 92 | hbm | 7.6, 1.1× | 8.3, 1.2× | 8.6, 1.2× | 8.1, 1.1× | 8.6, 1.2× |
| `res_in` | `land_add2_attnres_rms_snap` | k3_ar_fused | 91 | hbm | 7.1, 291× | 6.6, 43× | 6.8, 11× | 7.3, 6.1× | 6.8, 11× |
| `moe_front` | `moe_front` | k3_moe_front | 92 | latency | 6.5, 2.0× | 6.4, 2.0× | 6.5, 2.0× | 6.6, 2.0× | 6.5, 2.0× |
| `dcp` | `dcp_exchange` | k3_dcp | 24 | NVLink a2a | 12.0, 2.0× | 12.0, 2.0× | 14.2, 2.4× | 24.7, 3.1× | 14.5, 2.4× |
| `o_proj` | `gemm_bf16` | cublaslt_bf16_tn | 93 | hbm | 5.2, 1.7× | 5.2, 1.7× | 5.9, 1.9× | 5.8, 1.8× | 5.8, 1.9× |
| `sh_down` | `gemm_bf16` | cublaslt_bf16_tn | 92 | hbm | 3.8, 2.5× | 3.7, 2.4× | 4.0, 2.5× | 4.0, 2.4× | 4.0, 2.6× |
| `wfu` | `gemm_f32` | cublas_bf16_tn_f32 | 24 | hbm | 12.0, 1.7× | 12.2, 1.7× | 13.2, 1.8× | 13.8, 1.9× | 13.2, 1.8× |
| `q_b` | `gemm_f32` | cublas_bf16_tn_f32 | 24 | hbm | 10.7, 1.4× | 10.6, 1.4× | 11.2, 1.4× | 11.5, 1.4× | 11.2, 1.4× |
| `absorb` | `mla_absorb` | k3_mla_absorb_mma | 24 | hbm | 6.5, 3.7× | 6.3, 3.3× | 7.9, 3.1× | 10.1, 3.0× | 7.8, 3.1× |
| `mla_prep` | `mla_prep` | k3_mla_prep | 24 | launch latency | 3.5 | 3.1 | 3.1 | 3.5 | 3.0 |
| `argmax` | `head_argmax` | k3_head | 1 | NVLink | 11.7, 2.0× | 15.1, 2.3× | 35.2, 1.4× | 64.0, 1.2× | 35.2, 1.4× |
| `lm_head` | `gemm_f32` | cublas_bf16_tn_f32 | 1 | hbm | 48.6, 1.2× | 50.6, 1.3× | 48.0, 1.2× | 48.5, 1.2× | 47.8, 1.2× |
| `wgu` | `gemm_bf16` | cublaslt_bf16_tn | 1 | hbm | 17.9, 1.1× | 18.2, 1.1× | 20.7, 1.2× | 22.0, 1.3× | 20.8, 1.2× |

## Headroom, ranked

Savings in µs per step on rank 0. MoE items are at 64 / 8 rows of prose. Items marked *estimate* come from the
model and the measurements, not from a measured change.

| # | change | saves | kind |
|---:|---|---:|---|
| 1 | balance touched experts per rank (replicate each layer's hot experts, let `moe_front` pick the lighter holder) | ≤ 1700 / 1200 (the measured slowest − mean rank) | placement + routing |
| 2 | hide the MoE GEMMs' fixed per-launch tail: run `sh_down` beside FC1 / FC2, or FC2 behind each FC1 tile in one persistent kernel | ≤ 1400 / 900 | kernel / schedule |
| 3 | prefetch every dense projection's weights into L2 in the launch before it (`k3_ar_fused` already does it for three) | 740 / 680 / 660 / 520 at 1 / 8 / 32 / 64 rows | manifest + kernel args |
| 4 | 1 row: split-K variants of FC1 / FC2 so 152 SMs stream the 2–4 touched experts | ~700 at 1 row (estimate) | kernel variant |
| 5 | MLA attention: launch only the split slots the plan uses, grid z = min(16, ⌈64 / rows⌉) | 266 / 566 / 179 at 32 × 8k / 64 × 8k / 32 × 32k (measured with regenerated manifests) | needs `min` in launch expressions |
| 6 | shorten the attention-residual row chain shared by the attention all-reduce and the closing add | ≤ 420 on the all-reduce side (estimate), more on the closing add | kernel |
| 7 | `k3_kda_core`: start the state loads at entry (≤ 8 rows); a single balanced wave (≥ 32 rows) | ~300 at 1–8 rows, ~320 at 64 rows (estimate) | kernel tune |
| 8 | `k3_moe_front`: build the routing tables in FC1's prologue | ~300 (estimate) | kernel |
| 9 | pin the n = 7168 bf16 GEMMs (`o_proj`, `sh_down`) to an algorithm that fills 152 SMs | ~300 at 1 row (overlaps item 3) | `algo` pin |
| 10 | two-shot all-reduce for the 10752-wide MoE message from 12–24 rows, not 33 | 130–190 at 32 rows | constant |
| 11 | `k3_mla_absorb_mma` and `k3_mla_prep` | 106–163 and ≤ 50 | kernel tune |
| 12 | `k3_head`: skip the full-logits gather when only the token is needed | 58 at 64 rows | caller option |

**Two floors that kernels cannot move:**
- **The exchange floor.** Two exchanges per layer cost 186 × 3.9 ≈ 726 µs per step. Only merging or removing exchanges goes below it.
- **The MoE byte floor.** The bytes of the touched experts are 11.95 ms per step at 64 rows of prose. Only touching fewer experts moves it.

## Dense projections (cuBLAS)

Every projection is C[m, n] = A[m, k] · W[n, k]ᵀ with m = 1–64 rows and W the rank's TP slice: 25.2 GB of weights per step on every rank.
At ~m FLOP/byte against a 255 FLOP/byte ridge, all of them are weight streams.

| call | n × k | W per rank | layers | 1 × 8k | 64 × 8k | floor (1 → 64) |
|---|---|---:|---:|---:|---:|---:|
| `qkvg` (KDA, f32 out) | 6400 × 7168 | 87.5 MiB | 69 | 17.9 (1.43×) | 18.4 (1.43×) | 12.5 → 12.9 |
| `front` (router + latent down + shared up, f32 out) | 6016 × 7168 | 82.2 MiB | 92 | 13.8 (1.17×) | 17.7 (1.47×) | 11.8 → 12.1 |
| `lat_up` | 7168 × 3584 | 49.0 MiB | 92 | 7.6 (1.08×) | 8.1 (1.12×) | 7.0 → 7.2 |
| `o_proj` | 7168 × 1536 | 21.0 MiB | 93 | 5.2 (1.72×) | 5.8 (1.84×) | 3.0 → 3.2 |
| `sh_down` | 7168 × 768 | 10.5 MiB | 92 | 3.7 (2.47×) | 4.0 (2.42×) | 1.5 → 1.6 |
| `wfu` (MLA, f32 out) | 3648 × 7168 | 49.9 MiB | 24 | 11.9 (1.67×) | 13.8 (1.87×) | 7.1 → 7.4 |
| `q_b` (MLA, f32 out) | 18432 × 1536 | 54.0 MiB | 24 | 10.7 (1.38×) | 11.5 (1.37×) | 7.7 → 8.4 |
| `lm_head` (f32 out) | 20480 × 7168 | 280 MiB | 1 | 48.6 (1.21×) | 48.5 (1.19×) | 40.1 → 40.9 |

**Bytes are the minimum.** DRAM read / model bytes is 1.00 for every projection that nothing prefetches: every weight byte is read once.

**The prefetched ones show what the rest could be.**
- The all-reduce launched before `front`, `wgu` and `lat_up` is given the next GEMM's weight. Its blocks past the rows it reduces issue `prefetch.global.L2` over it while the exchange waits on the peers.
- Those GEMMs read only 0.34–0.80 of their bytes from DRAM and run at 1.08–1.17× at 1–8 rows.
- Everything without a prefetching predecessor sits at 1.38–2.5×.
- At 32–64 rows `front` climbs to 1.35–1.47× anyway: cuBLAS switches to slower 2-CTA tiles there.

**Two algorithm choices leave bandwidth idle.**
- **The n = 7168 bf16 projections launch 112 CTAs on 152 SMs.** For `o_proj`, `sh_down`, `lat_up` and `w_dn`, the heuristic picks a 64 × 8 tile, n / 64 = 112 CTAs, and no split-K, so 40 SMs stream nothing. `o_proj` reaches 4.25 TB/s and `sh_down` 2.97 TB/s, against 6.0 TB/s for the 756-CTA `lm_head`.
- **The f32-out path pays a second kernel.** `cublasGemmEx` picks split-K for `qkvg`, `front`, `wfu` and `lm_head` at 1–32 rows. That adds a `splitKreduce_kernel` of 2.5–3.6 µs each, 186 extra launches per step at 1 row. The f32 output is needed; the reduce is not.

The SM clock is 1970–2020 MHz under these streams, so power is not the limiter in decode.

**`k3_head`** copies each rank's `[m, 20480]` logits slice to rank 0 and exchanges one argmax key per row.
- At 32–64 rows it is bound by rank 0's NVLink ingress: 7 × m × 80 KiB is 35 MiB at 64 rows, 51 µs at 715 GB/s; it measures 64 µs.
- At 1–8 rows it is the small-message exchange: 11.7–15.1 µs against the 6.0 µs one-shot.

## MoE (`k3_moe_front`, FC1, FC2)

Rank r holds the 112 experts e with e % 8 == r. Every rank routes every row of the replicated batch through its own experts, and the partials meet in the all-reduce after FC2.

| launch | what | grid |
|---|---|---|
| `k3_moe_front` | one block per row: sigmoid of 896 logits, biased top-16, weights; the latent quantized to mxfp8; the shared expert's SiTU. The last block builds the routing tables (counts, 16-row CTA tiles, `route_map`). | rows × 1024 threads |
| FC1, TRT-LLM gen `bmm_MxE4m3_MxE2m1MxE4m3 … t128x16x256 … siTuGlu` | per touched expert: [up; gate] (6144 × 3584, mxfp4 + UE8M0) × its rows' mxfp8 latent, fused SiTU gate | 48 × (rows + 112) CTAs, 2-CTA clusters |
| FC2, `bmm_Bfloat16_MxE2m1MxE4m3 … t128x16x256` | per touched expert: down (3584 × 3072, mxfp4) × the FC1 rows, bf16 out | 28 × (rows + 112) CTAs |

**Model.**
- One expert is 11,698,176 B for FC1 (mxfp4 + scales) and 5,849,088 B for FC2. At the measured HBM rate that is **1.596 µs** and **0.798 µs**.
- A launch's floor is touched experts × expert bytes, plus < 0.1% for activations, over HBM. The FLOPs are 1–2% of the time even counting the 16-row tile padding.
- Touched experts come from dumped routing per layer, for rank 0 and for the busiest rank.

**Touched experts per rank (of 112), prose, mean over 92 layers:**

| rows | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| mean rank | 2.00 | 13.19 | 22.83 | 36.77 | 53.94 |
| busiest rank | 4.18 | 18.19 | 28.99 | 43.82 | 61.32 |
| uniform routing | 2.00 | 15.03 | 28.05 | 49.08 | 76.65 |
| bench routing (busiest) | 2.0 (4.1) | 9.8 (14.2) | | 24.3 (30.5) | 36.1 (43.1) |

- Real routing is concentrated: 64 rows touch 54 of a rank's 112 experts, 70% of uniform.
- The bench's tokens route narrower still: 0.67× the weight bytes at 64 rows. That is where the 4.2 ms of the summary comes from.

**Measured against the floor (bench routing, rank 0):**

| kernel | 1 × 8k | 8 × 8k | 32 × 8k | 64 × 8k | 32 × 32k |
|---|---:|---:|---:|---:|---:|
| FC1 ms/step (×) | 0.80 (2.95) | 2.12 (1.49) | 4.58 (1.30) | 6.48 (1.24) | 4.70 (1.27) |
| FC2 ms/step (×) | 0.49 (3.62) | 1.03 (1.44) | 2.13 (1.20) | 3.01 (1.14) | 2.21 (1.19) |
| `moe_front` ms/step (×, latency) | 0.60 (2.0) | 0.59 (2.0) | 0.60 (2.0) | 0.61 (2.0) | 0.60 (2.0) |

On prose at 64 rows: FC1 9.31 ms (1.17×), FC2 4.43 ms (1.11×). At 8 rows: 2.69 ms (1.39×) and 1.31 ms (1.35×).

**Why it sits there.**
- **Each extra expert costs the floor rate.** Fitting busy = a + b · touched over 92 layers × 3 replays gives FC1 b = 1.63–1.67 µs per expert and FC2 b ≈ 0.81: 98% of HBM bandwidth.
- **The gap is the intercept a.** FC1 a = 6–12 µs, FC2 a = 3–5 µs per launch. A CTA streams one 244 KB panel, the first and last waves run partly empty, and at 1 row only ~90 (FC1) and ~53 (FC2) CTAs have work for 152 SMs.
- **ncu at 64 prose rows:**
  - DRAM read / (touched × expert bytes) = 1.001 (FC1) and 1.002 (FC2): a second 16-row tile reads its expert through L2.
  - The tensor pipe is 7–8% active, so the 16-row tile padding is free.
  - Achieved DRAM rate over the whole launch is 6.53 / 5.96 TB/s.
- **EP skew sets the step.** Measured on all eight ranks with prose, the per-layer slowest rank's FC1 + FC2 sums to 15.27 ms per step against a rank mean of 13.58 ms at 64 rows, and 4.99 against 3.82 ms at 8 rows.
  - Rank 0 is an average rank, so its own times don't show this.
  - The wait lands in the MoE all-reduce on the faster ranks; see below.

## All-reduce and the residual rows (`k3_ar_fused`)

Each layer has two exchanges, and every row-wise residual operation is fused around them:

- **`ar_attnres_rms`** (94 per step): all-reduce of the attention output's bf16 partials `[B, 7168]`. It then adds the residual prefix, mixes the attention-residual snapshots (a softmax over 1–8 RMS-scored candidates), and applies the next RMS norm.
- **`ar_finalize_rms`** (92): weights the rank's routed expert outputs, all-reduces `[B, 3584 latent | 7168 shared]`, and applies the latent's RMS.
- **`land_add2_attnres_rms`** (92): the layer's closing add with the next layer's mix and RMS; no exchange.

**The protocol.**
- The exchange is a Lamport protocol over peer memory: a 16-byte bf16 −0.0 word means "not arrived", and three stages rotate with no barrier.
- Below 33 rows it is one-shot, with 7× the message of egress. From 33 rows it is two-shot.
- Its spare blocks prefetch the next GEMM's weight into L2.

**The floor is latency.** The epilogue's HBM traffic is ≤ 0.5 µs and the optimal NVLink egress ≤ 2.2 µs even at 64 rows. The floor is therefore the measured empty-payload 8-rank one-shot: 3.90 µs per call.

**Where rank 0's all-reduce time goes** (µs per step, 186 launches):

| shape | floor | alone (protocol + fused work) | in the step, last-arriving rank | rank 0 | rank 0 waiting on later ranks |
|---|---:|---:|---:|---:|---:|
| 1 × 8k | 726 | 1862 | 2201 | 2747 | 545 |
| 8 × 8k | 726 | 1862 | 2402 | 3308 | 906 |
| 32 × 8k | 726 | 2521 | 2793 | 3992 | 1199 |
| 64 × 8k | 726 | 2221 | 3196 | 4551 | 1355 |

1. **Waiting for later ranks: 545–1355 µs.** Nearly all of it is on the MoE all-reduce, where the first-arriving rank waits up to 28 µs per launch longer than the last. That is the MoE's EP skew, not the protocol; the attention all-reduce's spread is 1–2 µs.
2. **Fused work over the bare protocol: 1.5–4.5 µs per launch.**
   - On the attention all-reduce, the attention-residual row routine adds 4.5 µs (11.1 against 6.6 µs at one row).
   - The same routine alone costs 4.3 + 0.55 × snapshots µs, flat in rows: one 1024-thread block per row walks a chain of dependent steps.
3. **Protocol over the empty floor: 2.7–8 µs per launch.**
   - At ≤ 8 rows it is a flat 3.3 µs: latency, not bytes.
   - From 32 rows the one-shot's 7× egress shows. At 32 rows the MoE message's one-shot is 11.94 µs, while a two-shot of the same message is 10.53 µs. The switch at 33 rows is late.

**The closing add without an exchange** (`land_add2_attnres_rms`) is 6.3–6.9 µs at every row count against a 0.02–1.6 µs HBM floor. It is the same dependent row chain, and shortening it (e.g. scoring all candidates in one pass with one block reduction) saves time in both kernels.

## KDA (`k3_kda_core`)

One KDA decode step per row and head for the rank's 12 heads, with the short conv fused. Grid (rows, 12), 128 threads per block.

- Each row's state is 786 KB of f32 recurrent state plus 28 KB of conv window, read and written once.
- The minimum traffic is 2.1 / 13.7 / 53.5 / 106.5 MB at 1 / 8 / 32 / 64 rows. At ~1 FLOP/byte it is HBM-bound.

| rows | blocks | waves at 4 / SM | floor µs | measured µs | × | achieved | ncu DRAM read |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.02 | 0.29 | 9.18 | 32 | 0.23 TB/s | 1.3 MB |
| 8 | 96 | 0.16 | 1.87 | 9.42 | 5.0 | 1.46 TB/s | 7.2 MB |
| 32 | 384 | 0.63 | 7.29 | 12.13 | 1.7 | 4.41 TB/s | 27.4 MB |
| 64 | 768 | 1.26 | 14.52 | 22.66 | 1.6 | 4.70 TB/s | 54.2 MB |

- **The bytes are the minimum.** At 64 rows ncu reads 54.2 MB against 52.1 MB of state plus 2.1 MB of inputs.
- **1.6× at 32–64 rows is waves.** 128 registers a thread allow 4 blocks per SM. At 64 rows the 768 blocks are one full wave plus a 160-block tail, and at 32 rows a third of the resident slots never fill. A 64-register build or a persistent grid makes it one balanced wave.
- **9 µs at 1–8 rows is a chain of serial memory round trips** before the state is touched: line index → conv window → partial → decay tile → the state in two passes. Moving 2 MB or 14 MB takes the same time. Issuing the state loads at entry, since they depend only on the line index, would roughly halve it.

## MLA attention and the DCP exchange

Each of the 24 MLA layers runs:
1. `k3_mla_prep`;
2. `k3_mla_absorb_mma`: q_abs = W_UKᵀ q_nope for all 96 heads;
3. `mla_decode_h96_p64`: FlashInfer's CuTe-DSL split-KV decode over this rank's eighth of the context;
4. `k3_dcp`: the exchange that merges the eight ranks' partial outputs and applies W_UV.

**Attention, floor = the rank's latent rows once** (1152 B per position) plus q and the split partials.

| | 1 × 8k | 8 × 8k | 32 × 8k | 64 × 8k | 32 × 32k |
|---|---:|---:|---:|---:|---:|
| measured µs per call | 8.8 | 10.5 | 27.1 | 48.0 | 43.7 |
| floor µs | 0.42 | 2.48 | 7.36 | 12.99 | 22.80 |
| with the split slots the plan uses (16 / 8 / 2) | 8.8 | 10.5 | 16.0 | 24.4 | 36.2 |

- **Empty split slots cost more than the attention at 32–64 rows.**
  - The grid always has 16 split slots per row, whatever the split plan asks for. At 64 rows, 64 of 1024 slot clusters do work and 960 launch and exit, at about 25 ns each.
  - Regenerating the manifest at 8 and 2 slots, every other split count equal, measured 48.0 → 34.0 → 24.4 µs at 64 rows.
  - The best slot count is the plan's own maximum: 16 at 1 row, 8 at 8 rows, 2 at 32–64 rows.
  - The fix is a grid z of min(16, ⌈64 / rows⌉), which needs a `min` in the launch expression grammar.
- **After that fix it is 1.6–1.9× its floor.** ncu reads exactly the minimum (155.8 MB against 154.6 MB at 32 × 32k).
  - About 7 µs per launch is fixed.
  - Each 128-position tile takes ~1.70 µs, against 1.45 µs for the 2-SM MMA at its measured peak.
  - The kernel's own structural floor (24.7 µs) sits above the HBM floor (22.8 µs), for two reasons: 96 heads pad to 128 MMA rows, and 32 rows × 2 splits fill 64 of 76 clusters.
  - Closing the rest needs a different tiling: a 96-row tile, or positions as the M dimension.
- **`k3_dcp`** is 12.0–24.7 µs: 1.2–1.35× a same-size one-shot all-reduce at 32–64 rows, 1.7–2× at 1–8 rows. The extra is the split merge before the push and the merge + W_UV after the poll.
- **`k3_mla_absorb_mma`** reads exactly its 12.6 MB weight (q comes from L2) in one wave plus a serial tail: 3.0–3.7×, 106–163 µs per step of headroom.
- **`k3_mla_prep`** is 3.0–3.6 µs of launch latency on ≤ 0.2 µs of bytes. The TP8 build takes the kernel's generic path because its grid has 3 blocks in y and the fast path needs 4.

## Method and caveats

- **Counters.**
  - ncu ran single-pass on rank 0. One full replay at 1 row covered all 1211 launches.
  - At the other shapes ncu profiled the first three launches of each kernel configuration, and the rest take the median of their call site.
  - ncu serializes the step, so its µs are not the step's. Read it for bytes and utilization, nsys for time.
- **The latency floor is a measured lower bound.** The empty-payload exchange is not a hardware constant: a different protocol could go lower. It is the floor of this protocol's design.
- **`moe_front`'s floor is an estimate:** two dependent passes at the step's shortest kernel (1.63 µs).
- **Activations charged to HBM in the MLA model mostly stay in L2.** ncu shows this, so those floors are conservative.
- **The pipeline is deterministic.** Re-running the CPU side (call table, models, roofline join) reproduces the tables byte for byte.
