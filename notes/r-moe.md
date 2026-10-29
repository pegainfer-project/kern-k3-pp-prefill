# r-moe: D's routed MoE at its floor (round 3)

Target: `k3_moe_front`, FC1 / FC2 (TRT-LLM gen mxfp8 x mxfp4 batched GEMMs) and the launches around
them in the TP8 / EP8 / DCP8 decode step. Main at the start: 3.486 ms/step weighted, 12.9 launches a layer.

## Where it stands

- **Rebased on main e343e79 (06:45 UTC; 124 commits of the other runs).** Against that main in one lease,
  3 rounds: 3.2123 -> 3.1640 ms/step (-1.50%), every shape faster (8 / 16 / 24 / 36 / 48 rows at 64k:
  -0.05 / -0.04 / -0.05 / -0.07 / -0.01 ms); launches 327 -> 312 a step. loop/d/check bit-identical;
  teacher-forced real-text logits byte-identical to that main at 8 / 24 / 36 / 48 rows. Commit hashes
  below are the rebased ones.
- **For r-attn: moe_front is this run's target** (`loop/rounds/r3/r-moe.md`), not unowned. Main's f285eb9
  (bitonic sort32 + merge tree, r-attn) is what this branch had as its first step (`956fdd7`); the branch
  replaces it with the threshold top-k (tau from the warp maxima, the candidates compacted and merged 32 at
  a time by sort32 / merge32, exact for any candidate count: no O(n^2) ranking, so tie-heavy rows cost one
  more chunk, not 24 us) and the rest below. On this branch the threshold step measured 5.9 -> 5.1-5.4 us
  in the harness and 6.5 -> 5.6 in the step against the bitonic one (below).
- **For r-gemm:** the shared expert's down projection now runs inside moe_front (before FC1), so
  800c859's L2 pin of `sh_down` for after the MoE finds no `.sh_down` GEMM and stays off; nothing to do.

- Committed: `956fdd7` (moe_front top-k by warp bitonic sorts and a merge tree, bit-identical),
  `586631a` (moe_front sorts only the keys above the 16th largest warp maximum, bit-identical),
  `ec0a1bd` (tau by REDUX of the score bits, table counts zeroed early, bit-identical),
  `d77eb09` (the shared expert's down projection runs beside moe_front as a non-waiting programmatic
  dependent fed by per-row flags, bit for bit cuBLASLt's), `9622918` (sh_down reads each fragment once
  up to 24 rows: two warps x 32 rows; eight warps x 16 rows above, bit-identical), `84f6c1a` (sh_down
  from 25 rows: four warps x 16 rows x every tile), `11f29d0` (sh_down merged into moe_front's grid,
  moe_front a programmatic dependent of the front GEMM: one launch less a MoE layer, bit-identical),
  `9399a42` (above 24 rows the shared-expert blocks multiply on tcgen05 into TMEM, bit-identical),
  `3854ef6`, `3fba6c7`, `9e77304`, `0cf0afa` (FC1's two lowest touched experts prefetched into L2 while
  moe_front runs, below; bit-identical), `b5c3e3b` (after the rebase: tcgen05 at every row count, the
  mma.sync path gone; equal in the step, bit-identical).
  moe_front 6.5-6.6 -> 5.5-5.6 us a call in the step at 8-64 rows (before sh_down moved beside it).
  D 16 layers weighted, against the commit before each in one lease: 3.4857 -> 3.4733 (moe_front
  top-k), 3.4690 -> 3.4665 (tau), 3.4710 -> 3.4540 (sh_down), 3.4497 -> 3.4403 (sh_down layout).
  HEAD (9622918) against main in one lease, 3 rounds each: 3.4873 -> 3.4467 ms/step (-1.16%);
  then 3.4413 -> 3.4393 (84f6c1a), 3.4460 -> 3.4380 (11f29d0), 3.4340 -> 3.4317 (9399a42) and the
  prefetch steps 3.4353 -> 3.4323, 3.4270 -> 3.4220, 3.4257 -> 3.4197, 3.4220 -> 3.4170: about -2.0%
  against main.
  Launches 207 -> 192 a step, 12.9 -> 12.0 a layer (11f29d0).
- FC1 / FC2: no variant, bucket or grid change beats main's pair (below); the remaining gap is the
  cubins' one CTA per SM (wave quantization, ~46 GB/s a CTA) and FC1's wait at entry.
- The largest MoE cost left is EP skew (ar_finalize_rms waits for the busiest rank), a placement
  question; measured estimate under Proposals.

## Tools (work/, not committed)

- `work/dprof.sh OUT MANIFEST [ROWS] [CTX]`: the D bench with rank 0 under nsys (graph nodes); 20 s a run
  when GPUs are free. `work/dprof2.sh OUT MANIFEST <k3_step args>`: the same for teacher-forced runs.
- `work/kt.py`, `work/sum.py DIR...` (FC1 / FC2 / moe_front / ar_finalize busy us per call by rows),
  `work/tl.py SQLITE ROWS` (one step's per-layer timeline around the MoE).
- `work/ab.sh OUT ROUNDS tag=manifest...`: one TP8 lease, manifests benched interleaved (loop/d/bench's
  shapes); the noise between leases is larger than most effects here.
- Real routing: `work/mkrows.py` (R rows of the check's real text, distinct tokens per step: row r
  reads the four rows concatenated from offset r * 2560 / R), `work/peridx.py` (each layer's top-k ids
  in its own buffer, for `--dump`), `work/touched.py` (touched experts per rank from the dumps).
- `work/mf/` (moe_front harness: any cubin, CUDA graph, outputs dumped for bit compares, %globaltimer
  phase stamps), `work/shd/` (sh_down harness vs cuBLASLt and fp64), `work/pdl/` (PDL ordering test).
- Main's stored manifests in `$K3_LOOP_DATA/base/` no longer verify with this k3_step (`span_at`):
  A/B against a manifest generated from HEAD's scripts (`git archive HEAD scripts kernels.toml`).

## Floors (real text, 16-layer bench, rank 0)

Touched experts per rank (of 112), layers 1-15, real text (`work/touched.py`; 48 rows x 160 steps):

| rows | 1 | 8 | 16 | 24 | 36 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| rank 0 | 2.03 | 12.75 | 22.56 | 29.97 | 41.28 | 50.27 |
| mean rank | 2.00 | 12.60 | 22.14 | 29.76 | 40.97 | 49.98 |
| busiest rank | 4.25 | 17.58 | 28.12 | 36.28 | 48.18 | 57.44 |

An expert is 11,698,176 B (FC1) + 5,849,088 B (FC2): 1.596 + 0.798 us at 7.33 TB/s.

| kernel (us/call, rank 0) | 1 row (bench) | 8 rows real | 24 rows real | 48 rows real |
|---|---|---|---|---|
| FC1 measured / floor | 8.8 / 3.2 (2.75x) | 28.1 / 20.4 (1.38x) | 60.6 / 47.8 (1.27x) | 91.8 / 80.2 (1.14x) |
| FC2 measured / floor | 5.4 / 1.6 (3.4x) | 13.7 / 10.2 (1.34x) | 29.1 / 23.9 (1.22x) | 43.8 / 40.1 (1.09x) |
| moe_front (main) | 6.9 | 6.2 | 6.5 | 6.5 |

Bench routing (pseudo-random tokens) at 8 / 24 / 48 rows: FC1 26.6 / 53.0 / 79.6, FC2 12.7 / 25.3 / 37.6,
moe_front 6.6 / 6.5 / 6.6, ar_finalize_rms 22.5 / 29.0 / 32.1 (most of it waiting for the busiest rank).

## Done table (rank 0, us per call; floors from real-text touched experts; bench = pseudo-random tokens)

Real text, teacher-forced (`work/dprof2.sh`, `work/mkrows.py` rows), 16-layer step, layers 1-15:

| kernel | 1 row | 8 rows | 24 rows | 32 rows | 48 rows | 64 rows | limit |
|---|---|---|---|---|---|---|---|
| FC1 meas / floor (x) | 8.8 / 3.2 (2.75) | 28.1 / 20.4 (1.38) | 60.6 / 47.8 (1.27) | 72.9 / 57.6 (1.27) | 91.8 / 80.2 (1.14) | 105.9 / 93.4 (1.13) | cubin waves + entry wait |
| FC2 meas / floor (x) | 5.4 / 1.6 (3.4) | 13.7 / 10.2 (1.34) | 29.1 / 23.9 (1.22) | 34.9 / 28.8 (1.21) | 43.8 / 40.1 (1.09) | 50.7 / 46.7 (1.09) | cubin waves |
| moe_front now / est. floor | 5.9 / 4.3 | 5.6 / 4.3 | 5.5 / 4.3 | 5.6 / 4.3 | 5.6 / 4.3 | 5.7 / 4.3 | latency chain (below) |
| moe_front + shared projection (`9399a42`, bench) | | 7.2 | 7.5 | | 11.1 | | 36 rows 8.3; see sh_down below |

Rebased branch against main e343e79 (bench, nsys, rank 0, mean of 15 MoE layers; us from the end of the
front GEMM, r-gemm's own kernel up to 32 rows, cuBLAS + split-K reduce above):

| section | 8 rows main / branch | 24 rows main / branch | 36 rows branch | 48 rows branch |
|---|---|---|---|---|
| moe_front (branch: + shared projection + FC1 prefetch) | 5.87 / 6.98 | 5.84 / 8.29 | 7.48 | 12.16 |
| FC1 after moe_front | 27.55 / 26.46 | 52.55 / 50.75 | 66.51 | 77.74 |
| FC2 after FC1 | 13.07 / 13.02 | 25.12 / 25.15 | 32.88 | 38.68 |
| front end -> FC2 end | 46.49 / 46.47 | 83.51 / 84.20 | 106.87 | 128.57 |
| sh_down after FC2 (main only) | 4.2 | 6.0 | | |

Real text (teacher-forced, `work/mkrows.py` rows, 16 layers, `work/dprof2.sh` + `work/span.py`: rank 0's
median GPU span a step over 161 steps), main e343e79 -> branch: 8 rows 2178.8 -> 2135.6 us (-2.0%), 24 rows
3079.6 -> 3028.3 (-1.7%), 36 rows 3667.4 -> 3593.2 (-2.0%); the MoE section (`work/sect.py`) as in the
bench: moe_front 5.80 / 5.86 / 5.85 -> 7.25 / 8.55 / 7.60 us, FC1 after it 28.19 / 59.08 / 75.92 -> 26.73 /
57.17 / 73.07, FC2 unchanged, and main's sh_down after FC2 gone.

The branch's moe_front includes ~1-2 us of waiting for its FC1 prefetches to land (a grid retires once
they have), which FC1 gets back; main's separate sh_down after FC2 is what the step saves. Without the
prefetch (warp 31 exits) the branch measures 3.1643 -> 3.1810 ms/step: the prefetch is -0.5% here.

(1-row FC numbers are bench routing: 2.0 touched experts on rank 0 either way. 32 and 64 rows: touched
counts from the first rows of the 64-row dump, the profile from its own 32 / 64-row file.)
Bench (8k context) 1 / 8 / 32 / 64 rows after the commits: FC1 8.8 / 26.3 / 61.3 / 91.3, FC2 5.4 / 12.5 /
29.1 / 43.4, moe_front 5.9 / 5.6 / 5.6 / 5.7. At 64k / 128k the same within 1%.

moe_front's floor estimate is its chain of dependent steps, each measured in the harness: launch to
first instruction ~0.8 us, the front row from L2 ~0.6, the selection ~1.0 (warp max, tau, compaction,
one sort + merge), the cross-block hand-off (atomic acq_rel round trip) ~0.8, the ids from L2 ~0.6,
the tables ~0.5. Measured 5.5-5.7 in the step, 5.1-5.4 in the harness: 1.2-1.3x of that chain.

## Why FC1 / FC2 sit where they are (the named limit)

- Both cubins run one CTA an SM: 512 threads x 128 registers is the whole register file, and 223-225 KB
  of shared memory. A CTA streams one 128-row panel of one expert (FC1 244 KB, FC2 196 KB).
- FC1's first instruction is `griddepcontrol.wait` (SASS: ACQBULK at 0x140, before both PREEXITs):
  nothing of FC1 overlaps moe_front, and its HBM stream starts only after moe_front's grid completes.
  FC2's weight loader runs before its wait (no ACQBULK on that path): it streams during FC1's tail.
- A wave model fits every measurement: FC1 ~ ceil(48 E / 152) waves x 5.2-5.9 us + ~2 us
  (E = touched experts; 5.06 us is a full wave's 152 panels at 7.33 TB/s): 8 / 24 / 48 rows real
  5 / 10 / 16 waves -> 28.1 / 60.6 / 91.8 us measured. The intercept is the last wave's quantization
  (half a wave on average, partly refilled by FC2's early weight loads) plus the wait at entry.
- No other variant does better (measured in the step, nsys, rank 0; `work/list*.txt`):
  - FC1 tile 16, stages 3-6, 1- or 2-CTA MMA, u2 or not, Pd scheduler: main's (s6, 2-CTA) is best;
    s5 +0.3-0.6 us, s4 +2-4 us, 1-CTA +4-17 us. The static scheduler (schedS) is 2x slower everywhere.
  - FC2 likewise: main's u2_s6 2-CTA = non-u2 s6 (12.9 / 25.1 / 37.2); fewer stages or 1-CTA worse.
  - Tile 8 (needs moe_front's tables at TILE 8: built and checked): s6 = main at 8-48 rows
    (27.2 / 53.2 / 80.7), others worse; split-K 2 (tile 8 only) worse at every row count including
    1-4 rows (FC1 10.4-11.2 vs 8.8 at 1 row): its 2-deep cluster along K leaves 2 x 48 CTAs a touched
    expert, which spills into a second wave already at 2 experts.
  - At 1 / 2 / 4 rows main's pair is also the best (FC1 8.8 / 12.2 / 17.9, FC2 5.4 / 6.4 / 8.8).
  - Grid: the batch dim is launched at its bound (112 + T CTA rows, mostly exiting at once). Cutting the
    bound to 40 + T (a timing-only hack) changed FC1 / FC2 by <= 1%: exiting CTAs are cheap.
- Bytes are the minimum (ncu, single-pass metrics inside the TP8 step, 8 rows real text, one layer;
  `work/ncu2.sh`: kernel replay cannot save a 200 GB device): FC1 read 198.9 MB = 17.00 touched
  experts x 11.70 MB, FC2 99.6 MB = 17.02 x 5.85 MB. Alone (no PDL overlap) FC1 took 41.3 us (4.8 TB/s,
  1.52x) and FC2 25.3 us (3.9 TB/s, 1.86x); in the step FC2's early weight loads hide part of that.
- So per-bucket routing (`when`) has nothing to route to; the generator change that would do it
  (`--moe-buckets`, moe_front's TILE define) was written, measured, and not committed.

## moe_front (one block a row, the last one builds the tables)

Harness (`work/mf`, CUDA graph of 20 calls, inputs from L2): 6.0-6.2 us at 8-64 rows. Phase stamps
(%globaltimer, 24 rows, per block): load + sigmoid + top-k 2.65 us, waiting for quant 0.4, the
last-block hand-off (atomic acq_rel round trip + barriers) 0.8, the tables (L2 read of the ids, smem
histogram, scan, writes) 1.3; launch and teardown ~0.8 more between back-to-back calls.

Tried:
- **Threshold, then sort** (kept, `586631a`): each warp's maximum 64-bit key (one shuffle butterfly), tau =
  the 16th largest of the 28 maxima (one warp sort), the keys >= tau compacted (ballot + one shared
  atomic a warp; 16-30 of them typically), one warp's running top 16 over them 32 at a time. Exact for
  any candidate count. 5.9 -> 5.1-5.4 us in the harness, 6.5 -> 5.6 in the step. Bit-identical.
- **Bitonic top-k** (kept, `956fdd7`, superseded by the above): warp w of 28 sorts its 32 (score key, ~expert) 64-bit keys by a bitonic
  network, 14 warps merge the warps' top 16 pairwise in 5 levels. Bit-identical picks, order and
  weights (the same total order); 6.1 -> 5.9 us in the harness.
- One 8-CTA cluster doing every row's top-k and the tables from DSMEM after a cluster barrier
  (FlashInfer's routingIndicesClusterKernel shape), row blocks doing quant / SiTU: bit-identical but
  the top-k is throughput-bound on the SM, ~1.6 us per extra row a CTA: 5.7 (8 rows) / 7.3 (24) /
  10.6 (48) us. With 8 slice warps a row (<= 4 rows a CTA): 5.9 / 7.3 (16) / 8.9 (24). Dropped.
- Ballot-based rounds (cached lane best, the winner lane rescans): slower (6.6), the divergent winner
  branch costs more than the second REDUX. 16 slice warps: slower (merge over 256 candidates).
- Generation-tagged picks polled by an extra block instead of the atomic hand-off: no faster.
- Quant moved to 4 spare warps from the start, hand-off issued right after the picks: slower (6.5:
  four warps cannot hide the quant's L2 loads); hand-off early alone: no change.
- **L2 prefetch of FC1's first wave from moe_front** (blocks prefetch their picks' FC1 weights below a
  row-count-dependent expert id): FC1 got 1.6-3.5 us faster, as predicted, but moe_front took 50-58 us:
  `prefetch.global.L2` from a few SMs runs at ~0.25 TB/s each, and `cp.async.bulk.prefetch.L2` the
  same, because a grid does not complete before its prefetches land, and FC1 waits for the grid.
  Prefetching from the kernel before FC1 cannot overlap FC1's wait.

## The launches around: sh_down (r-gemm's GEMM; r-gemm: it is ours now, see below)

**Committed (`d77eb09`): sh_down beside moe_front.** During moe_front (~5.5 us) ~140 SMs only host FC1's
CTAs parked at their griddepcontrol.wait. sh_down's input (the SiTU row) exists ~1.2 us into
moe_front. `k3_sh_down.cu` launches right after moe_front with `pdl: true` and never waits on it:
moe_front's row block b raises ready[64 c + b] for each of sh_down's 112 CTAs (st.release.gpu after
its SiTU warps' barrier); CTA c acquires its T flags, clears them (a carry; no cross-CTA reset),
stages the rows by cp.async and multiplies. FC1 (the next launch) waits for both grids (measured,
work/pdl). The manifest rule says a pdl kernel waits before reading its predecessor's output; here
the flags carry that ordering (noted in the commit and the kernel).
- Bit-identical to cuBLASLt: one mma.sync m16n8k16 chain per output over contiguous k16 blocks in
  ascending order reproduces cuBLASLt's nvjet sums exactly (100% of outputs at 1-64 rows, random and
  real layer-3 weights; step logits byte-identical at 24 and 48 rows; loop/d/check bit-identical).
  Two near misses on the way, worth knowing: a K permutation (two k16 steps fed from one 16-byte load)
  or K split into halves / accumulator sets gave 99.95-99.99% identical outputs (one bf16 ulp, never
  less accurate than cuBLAS), and that alone FAILed loop/d/check: 105 flips vs main, 3 confident flips
  vs the oracle. A one-ulp difference in 2-5 of 28672 hidden elements after one layer grows through
  routing near-ties over four layers. In this step only bit-identity is safe.
- mma.sync on sm_103 (work/mmalat): ~20 cycles dependent latency, one every ~8.4 cycles per warp.
- In the step (nsys and in-kernel %globaltimer stamps via `--dump` of a debug buffer, work/dbgrun.sh;
  24 rows, us from moe_front's entry): moe_front's SiTU done 1.2, sh_down entry 1.0, flags seen
  2.2-4.1 (median 2.9), act landed +0.8, GEMM +2.0 (smem-bandwidth-bound: the two tile-parity warps of a
  row group load the same A fragments), end ~6.5 against moe_front's ~5.5. sh_down ends 1.4-1.6 us
  after moe_front at 8-24 rows instead of 3.7-3.8 after FC2; at 48 rows its 112 CTAs (207 KB of shared
  memory each) do not all fit beside moe_front's 48 blocks and it ends 4 us after.
- Tried after the commit, no better: weight loads issued after the flags (the polls no longer queue
  behind 6144 cp.async: flags seen 0.4 us earlier, but the weights land 2.6 us later); weights by TMA
  bulk copies from two warps with two other warps only polling, and one gpu fence + relaxed flag
  stores in moe_front instead of 112 st.release: same as committed.

- After `9622918` (nsys, bench, sh_down end minus moe_front end): 0.2 / 0.8 / 2.4 / 3.2 us at 8 / 24 /
  36 / 48 rows, and moe_front itself runs 0.6-1.2 us longer from 24 rows (5.5 -> 6.3-6.85: sh_down's
  11 MB of weight loads contend with its L2 round trips). Still ahead of the old 3.7 us tail after FC2
  at every row count. Tried, not better: four warps x 32 rows x alternate tiles for every row count
  (harness 8.2 vs 7.6 us at 36 rows).
- In-step stamps at 36 rows (HEAD): flags stored by moe_front 1.6-1.9 us, seen by sh_down 3.3-4.6 (its
  polls queue behind the CTA's 6144 weight cp.async), act +0.8, the 8-warp GEMM +2.5, end 7.5-7.9.
  Tried, A/B in one lease (3 rounds): weight cp.async only from the warps of sub-partitions 2-3 (polls on
  0-1 then see the flags ~0.6 us earlier) 3.4430 -> 3.4463, worse; plus four warps x 32 rows x alternate
  tiles above 24 rows 3.4717, worse. FC1 waits for every grid launched before it (the transitive wait
  above), so sh_down's end, not moe_front's, now sets FC1's start at every row count: moe_front's own
  tail (hand-off, tables) only pays once sh_down is shorter.
- **Committed as `11f29d0`: the merge works once moe_front itself is a programmatic dependent.** With
  `pdl: true` on moe_front and griddepcontrol.wait at the entry of every block (it waits for the front
  GEMM's split-K reduce, then triggers FC1), FC1 runs as fast as before and the merged grid starts
  ~1.1-1.5 us before the reduce ends: 3.4460 -> 3.4380 ms/step against 84f6c1a, 12.0 launches a layer.
  Measured from the reduce's end the merged grid ends at 6.0 / 6.0 / 8.1 / 10.0 us at 8 / 24 / 36 / 48
  rows (HEAD before it: max(moe_front, sh_down) 5.9 / 7.3 / 9.2 / 10.3). In-kernel stamps at 36 rows:
  SiTU done 1.2 us, the 112 st.release flags stored by 2.25, seen 3.0-3.2, act landed +0.9, GEMM 2.6
  (4 + 1 tile chunks), end ~6.9. Tried on top, no gain: one gpu fence + relaxed flag stores with
  3-tile chunks (36 / 48 rows a little better, 16 / 24 a little worse: 3.4360 = 3.4360), 3-tile chunks
  alone (3.4353 -> 3.4350).
- The first attempt (below) without that: blocks T .. T + 111 of moe_front run the projection;
  launch_bounds(1024) -> 54-64 registers, no spills; row blocks dispatch first, so no hang): one launch
  less a MoE layer, 207 -> 192 a step (12.9 -> 12.0 / layer), bit-identical (loop/d/check). But
  3.4480 -> 3.4740 ms/step (+0.75%): FC1 runs ~2.2 us longer after the merged kernel, at every layer
  and row count (layer-by-layer nsys, `work/fc1gap.py`), although the merged grid ends earlier than
  HEAD's sh_down at 24-36 rows. Bisected: merged with the shared blocks idle -> FC1 normal; without
  their GEMM -> normal; with the GEMM (even on garbage weights, even HEAD's exact GEMM code, even
  delayed 0.9 us) -> slow. Not the smem carveout (HEAD with moe_front's dynamic smem raised to 207 KB:
  equal), not the block size (HEAD's sh_down at 1024 threads: FC1 normal). With FC1 not PDL, merged and
  separate give the same FC1 (29.6 / 29.5 us): it was the PDL chain. Fixed by launching moe_front as a
  programmatic dependent too (above).
- **Committed as `9399a42`: tcgen05 above 24 rows.** One thread issues 48 `tcgen05.mma.cta_group::1.kind::f16`
  (M = 64 W rows, N = rows rounded to 8, K = 16) into 64 TMEM columns; operands K-major in 128-byte swizzle
  atoms (8 rows x 64 k, SBO 1 KB, the descriptor advanced 32 B a K step inside an atom); warp w reads tile
  w / 4 of TMEM lane quarter w % 4 (`32x32b.x8`; M = 64 puts D row 16 q + l in lane 32 q + l). Bit for bit
  cuBLASLt at 1-64 rows in the harness (`work/umma`), teacher-forced logits byte-identical at 32 / 36 / 48.
  What it took (harness stamps): the no-swizzle core-matrix layouts land the operands 3-5 us late (either
  the global reads or the shared writes of a warp scatter over 8 lines; cp.async wants both contiguous),
  the 128-byte swizzle lands them as fast as the pitched rows. One M = 64 MMA occupies the tensor pipe
  ~25 ns whatever N (two interleaved accumulator chains take twice as long), and the issuing thread's
  descriptor arithmetic must be constant offsets on a precomputed descriptor (else ~100 ns an MMA): 48
  MMAs ~1.2 us at any row count, against mma.sync's 1.4 us at 8 rows and 2.6 at 36. In the step: 3.4340
  -> 3.4317 ms/step; moe_front 9.1 -> 8.3 us at 36 rows, unchanged at 48 (11.1).
- Tried on `9399a42`, not better:
  - Wsh stored as the swizzled shared-memory images (a load-time derive kernel), the row blocks writing their
    SiTU rows into one image, the blocks fetching both by `cp.async.bulk` (12 x 8 KB, mbarrier tx counts)
    and tcgen05 at every row count: bit-identical, 5.9-6.0 us at 8-48 rows in the harness (HEAD 5.4-7.8),
    but in the step moe_front +0.6 us at 8 / 24 rows (-0.7 at 48): 3.4327 -> 3.4353. The same images
    gathered by cp.async into the pitched rows with mma.sync up to 24 rows: 3.4347 -> 3.4330, noise; not
    worth a second weight layout. Probably the bulk copies' 11 MB burst slows the row blocks' L2 round trips.
  - At 41+ rows (48 + 112 blocks > 152 SMs: 8 shared-expert blocks wait for a second wave) a grid of 152
    (`when`) with row blocks 0..7 taking the leftover column tiles after their row: moe_front 11.1 -> 12.4
    us at 48 rows with the weights prefetched at entry (they queue ahead of the row's own loads), 11.4 with
    the prefetch from the SiTU warps after publishing. The second wave is not what costs at 48 rows.
  - In-step stamps (`work/dbg3.sh`, `work/dbg4.sh`: a u64 buffer dumped per step, %globaltimer from one
    thread; each stamp also loads the buffer's pointer, so gaps carry ~0.5 us of error), HEAD, us from the
    first block's entry: rows' SiTU 1.3-1.5, flags stored 2.0-2.8, picks 3.0, hand-off (barrier + the
    acq_rel atomic) 4.0-4.9, the last block's tables 5.4-6.1; shared-expert blocks see the flags at 3.1-3.3
    (max 3.8-4.5), data 4.0-4.1, product 5.5-5.8, end 5.8-6.0 (max 6.5-7.3) at 36 rows; at 16 rows the
    tables (5.6-6.1) and the projection (end max 5.7) finish together; at 48 rows the 8 blocks of the
    second wave enter at 5.3 and end at 9.7.
  - The hand-off replaced by per-row id flags (warp 0's st.release after the picks) and block 0 building the
    tables once it acquires all T: moe_front -0.17 us at 8 rows, +0.14 at 24, -0.57 at 48; 3.4280 -> 3.4263
    in one lease, noise. Block 0's stamps put its tables 2.6 us after its polls succeed (HEAD's last block:
    0.2-1.2 after the hand-off), not understood. With row blocks 1..8 taking the 48-row leftover tiles on
    top (weights prefetched by their SiTU warps): moe_front 9.8 us at 48 rows (HEAD 11.1), but that build
    measured 3.4350 -> 3.4500 in another lease, +0.02-0.03 ms at 8-24 rows where it should equal the
    first. Polls without `__nanosleep`: no change (3.4350 -> 3.4345).
- Tried on `11f29d0`: the shared-expert blocks land the SiTU rows themselves from the f32 front rows (no
  flags, no `sh_ready`, row blocks without SiTU warps): bit-identical, but 3.4350 -> 3.4757 ms/step
  (+1.2%), the loss growing with rows (+0.05 ms at 24-36, +0.1 at 48): each of the 112 blocks reads T x
  6 KB of f32 gate | up instead of T x 1.5 KB of bf16 rows (24 MB of L2 reads at 36 rows).
- Verification caveat: main itself gave different logits at 24 rows teacher-forced across two leases
  once (48 rows reproducible); compare a candidate against main run in the same batch.

Before that, as a dependent of FC2:

- cuBLAS's sh_down already launches as a programmatic dependent (it starts ~2 us before FC2 ends) and
  ends 3.7-3.8 us after FC2.
- sh_down before FC2 (FC1 -> sh_down -> FC2): 3.489 -> 3.508 ms (worse).
- PDL ordering (work/pdl, measured): a programmatic dependent B that never executes
  griddepcontrol.wait runs fully overlapped with its primary A, and the launch after B (normal, or
  PDL with griddepcontrol.wait) still starts only after A completes. So a sh_down of ours launched
  `pdl: true` after FC2 that never waits is safe, and takes the SMs FC2's last wave leaves.
- `source/k3_sh_down.cu` (written for this, not committed): bf16 mma.sync, weights by one bulk copy a
  warp, the activation staged once a CTA. vs cuBLASLt alone: 3.4 / 3.4 / 3.9 / 4.2 / 7.0 us at
  1 / 8 / 16 / 24 / 48 rows against 3.0 / 2.9 / 2.8 / 3.7 / 3.8; 99.97% of outputs bit-identical to
  cuBLAS, the rest one bf16 ulp either way, same max / mean error vs fp64. mma.sync's latency on
  sm_103 (~140 ns a dependent mma) bounds it: it needs 8-16 independent accumulator chains a warp.
  In the step (three builds by `when`): 3.485 -> 3.482 ms, noise. Not committed.

## FC1's first experts prefetched during moe_front

HBM idles during moe_front once the shared weights have landed (~2.5 us), and FC1's first instruction waits
for moe_front's grid. FC1 launches expert-major (grid 48 panels x CTA rows, x fastest), so its first wave
reads the touched local experts of the lowest indices. Committed in four steps, each bit-identical and
measured against its predecessor in one lease (3 rounds):
- `3854ef6`: every row's SiTU warps also publish the row's ids to each shared-expert block (`sh_ids`,
  after a named barrier with the picking warp); warp 31 of each shared-expert block, after the block's
  activations land, takes the lowest touched local expert and prefetches its 1/112 share of that
  expert's FC1 weights and scales (`cp.async.bulk.prefetch.L2`, 98 KB + 6 KB). -0.1% (three leases).
  moe_front then ends 1-2 us later (a grid retires once its prefetches land), FC1 needs 1.4-1.8 us less.
- `3fba6c7`: warp 31 leaves the projection at entry (the other warps sync on a named barrier of 992) and
  prefetches as soon as the ids are in: -0.15%.
- `9e77304`: each row sends its lowest local pick as the flag value (relaxed), so the prefetcher reads no
  ids from L2 and the SiTU warps no longer release: -0.18%.
- `0cf0afa`: the two lowest (two picks a row in the flag): -0.15%.
Tried, worse or equal: half or a quarter of one expert (the flag machinery costs more than the bytes it saves),
three or four experts (3.4113 two -> 3.4187 / 3.4197), the touched set as a 128-bit mask (slower to
build than a min-reduction, so the prefetch starts later).
Together 3.4317 (9399a42) -> ~3.411-3.417 ms/step depending on the lease. On main e343e79's front
(r-gemm's own GEMM up to 32 rows) the prefetch is worth more: without it 3.1643 -> 3.1810 ms/step.
After the rebase, also tried: a speculative prefetch (each expert below j0 = 2 x 112 / E(T), E the
expected touched experts, issued as soon as any row reports it, the two least once all rows are in):
3.1573 -> 3.1587, nothing gained; moe_front without PDL at <= 32 rows (behind r-gemm's front kernel):
3.1643 -> 3.1813, worse (the early launch pays).
At 41+ rows (48 + 112 blocks > 152 SMs) a grid of 152 whose first 8 shared-expert blocks take a second
column tile after their first (its weights prefetched into L2 at entry, the product into the same TMEM
columns, mbarrier phase 1): bit-identical, 48 rows -0.02 / -0.03 ms in two leases, but 24 rows at 128k
+0.01 in both (the narrow path's code is the same logic, refactored): 3.1617 -> 3.1637, 3.1603 -> 3.1613.
Not kept (`work/umma/k3_moe_front.twotile.cu`).

## Proposals

- **For r-ar: launch `ar_finalize_rms` (and `ar_attnres_rms`) as programmatic dependents** (measured,
  not committed: the kernels are r-ar's). Since `11f29d0` nothing sits between FC2 and `ar_finalize_rms`
  (sh_down used to). In `kern_k3_ar_finalize_rms` add `asm volatile("griddepcontrol.wait;" ::: "memory");`
  right after `lamport_open(...)` (it touches only the all-reduce's own state; everything after reads
  FC2's or earlier outputs); in `kern_k3_ar_attnres_rms` the same wait before its entry trigger; and
  `"pdl": True` on both launches in gen_stage.py. Against HEAD in one lease, 3 interleaved rounds:
  ar_finalize alone 3.4373 -> 3.4183 ms/step (-0.55%), both 3.4380 -> 3.4207 (-0.5%); loop/d/check
  bit-identical for both (run it 3 times before merging: Lamport kernels).
- **Finalize + all-reduce against FlashInfer / TRT-LLM (direction 3).** r-ar's `kern_k3_ar_finalize_rms` is
  already TRT-LLM's MoE finalize-allreduce fusion in structure: the combine is TRT-LLM's finalize (FC2 rows
  gathered by expanded index -> permuted index, `exp2perm` here, `expandedIdxToPermutedIdx` in
  FlashInfer's `trtllm/fused_moe/DevKernel.h` finalize, the 16 picks summed in order in f32 with the
  routing weights, rounded once), then the allreduce_fusion Lamport one-shot (two-shot from 33 rows) and
  the RMSNorm, plus the shared expert's partial in the same exchange. What FlashInfer's launch has and ours
  does not is `mUsePdl` (cudaGridDependencySynchronize at entry): the proposal above, -0.5% measured. In
  the step the kernel takes 17-43 us at 48 rows, most of it waiting for the busiest rank's FC2 (EP skew,
  below); no finalize variant changes that.
- **For r-attn: PDL on `kda_core`, `mla_prep`, `mla_absorb` drifts; on `dcp_exchange` it is clean.** With
  griddepcontrol.wait at entry (before any memory access) and `pdl: true`, loop/d/check: `kda_core` alone
  12 flips vs main (relRMS max 1.1%), `mla_prep` + `mla_absorb` 124 flips (median 1.0%), both still inside
  the band; `dcp_exchange` alone bit-identical. Together with the AR pair: 3.4347 -> 3.4143 and a FAIL.
  All three drifting kernels follow a cuBLAS `gemm_f32` (GEMM + split-K reduce): waiting on that pair
  as a programmatic dependent apparently does not always see its output. `moe_front` follows the same
  kind of call and is a programmatic dependent since `11f29d0`: 6 of 6 loop/d/check runs and
  teacher-forced logits at 8 / 16 / 24 / 36 / 48 rows are bit-identical to main (a stale front row would
  change routing and show at once). Why it differs from kda_core is not understood: keep an eye on it.
- **EP balance (largest MoE lever, outside the fixed layout).** Simulated on the dumped real routing
  (`work/epsim.py`, 64 rows x 160 steps x 15 layers; a replicated expert goes to the holder with fewer
  touched experts so far, which every rank can compute from the same picks): busiest rank's touched
  experts at 8 / 24 / 64 rows

  | placement | 8 | 24 | 64 |
  |---|---:|---:|---:|
  | EP8 today (mean rank) | 17.32 (12.48) | 35.38 (29.07) | 65.59 (58.23) |
  | each layer's 8 / 16 / 32 hottest experts also on the next rank | 16.68 / 16.24 / 15.63 | 34.65 / 33.93 / 32.81 | 64.69 / 63.81 / 62.43 |
  | every expert on two ranks (r, r + 1) | 13.75 | 30.17 | 59.07 |

  At 2.4 us an expert (FC1 + FC2 at HBM rate), every expert on two ranks saves 8.6 / 12.5 / 15.6 us a
  layer on the critical rank, but doubles the expert memory (182 GB more a rank at 93 layers: does not
  fit). The 32 hottest a layer (+29%) save 4.1 / 6.2 / 7.6 us a layer.
  Why it matters: the busiest rank touches 17.6 / 36.3 /
  57.4 experts at 8 / 24 / 48 rows of real text against a mean of 12.6 / 29.8 / 50.0; every rank waits
  for it in ar_finalize_rms (22.5 / 29.0 / 32.1 us a call on rank 0 against ~10 of its own). The skew
  is binomial spread in the count of distinct experts touched, not a few hot experts, so it needs each
  expert reachable on two ranks; the batch is replicated, so every rank can compute the same balanced
  assignment from the picks with no exchange. Memory is the cost (a second copy of a share of the
  experts).
- **A persistent MoE kernel** (FC1 -> FC2 per expert panel, the shared projection filling the drain).
  References: DeepGEMM's `sm100_fp8_fp4_mega_moe.cuh` (fp8 x fp4 tcgen05, FC1 + SiTU + FC2 in one
  persistent kernel, but built around EP dispatch / combine over symmetric buffers), and the kernel
  index's `k3_mega_moe` (a K3 build of it, not for the full 896 experts, r1-r16). Our layout (replicated batch, every
  rank its experts over every row, then the all-reduce) would need its dispatch removed and the routing
  tables built in it. What it could take back, from the waves above: FC1's and FC2's last-wave
  quantization (half a wave each, 2-3 us), the FC1 -> FC2 boundary, FC1's entry wait behind the tables
  (only if it builds them itself) and the shared projection's tail past the row blocks (0-1 us at 8-36
  rows, ~4.5 at 48): roughly 5-8 us a layer at 8-48 rows, 2-3.5% of the 16-layer bench. Needs
  block-scaled tcgen05 MMA and, for numerics, the GEMM.md proof (FC1's mxfp8 output requantizes any
  change of summation order).
- **The shared projection on a graph branch beside FC1 / FC2 (direction 4): not worth a runtime change.**
  The runtime cannot express a branch: a launch is module / entry / params / grid / block / shared_mem /
  cluster / pdl / args (+ `when`) in the kern checkout's `tools/kern_manifest.py` (its `docs/runtime.md`
  is not in this checkout), one stream, PDL the only overlap; the merged moe_front is what PDL allows.
  Beside moe_front the projection costs only its tail past the row blocks (above). On a parallel branch its 112 CTAs
  (one an SM, ~3.5 us each, bound by their 98 KB weight loads) would take ~2.6 us of whole-GPU SM time from
  FC1's / FC2's waves, more than that tail at 8-36 rows; only FC1's last-wave idle SMs could absorb it for
  free, and no launch-level scheduler can aim there (the persistent kernel above can).
