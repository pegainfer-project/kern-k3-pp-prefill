# DCP microbench：2×EP36 形态下 context 条带化的代价（想法 → 测法 → 结果）

> Moved from kern's `docs/` on 2026-11-08. Paths in backticks (`tools/`, `docs/`,
> `crates/`, `examples/`) are kern's as they were when this was written.

状态：2026-09-01 起，tray13/14。前置：`agent-workload.md`（负载）、`multi-gpu.md`
（形态）。代码与原始输出：本地存档 `bench-results/2026-09-01-dcp-bench/`（不在仓库）。

## 0. 我们相信什么（待证伪的命题）

设计把一切 attention 归结为 **"context 条带 + query 本地 + partial merge"**，每步是一批
span（decode n=1、extend n=chunk）。它成立要靠下面五条，每条对应一个测法：

| # | 命题 | 若为假的后果 | 测法 |
|---|---|---|---|
| H1 | 吸收式 MLA decode（96 头、FP8 latent 576 B/token）在 GB300 上**算力绑定**，每 (q, ctx-token) ≈ 209 kFLOP，attention 的实际效率 ≥ 1 PF/s | 成本模型（attention 占 89%）整体失真；若带宽绑定则 DCP 不减每 rank 时间、只改延迟 | B1：cuBLAS FP8/bf16 GEMM 复刻 QKᵀ / PV 形状（M=b·96, K=576/512, N=ctx），扫 b、ctx，看 TFLOP/s 与字节 |
| H2 | DIST_CTX 每层的 **q 广播 + partial 回收 + merge** 在 lockstep 里的附加 ≪ 一层的算时（几 µs 级/层，24 层 < 0.5 ms/步） | DCP 变成每步几 ms 的税，条带宽度不敢开 | B2：w ∈ {1,2,4}（tray 内）真实搬字节 + doorbell，算时用标定过的 spin 代替，测步时增量 |
| H3 | **extend chunk 的 DCP 流量**（q: c×96×192×2 B，partial: c×96×128×4 B，每 peer 每层）在 w=4、c=2k 时每步 ≈ 几 ms——与 5 ms 预算同量级，**chunk 上限受通信而非算力约束** | "chunk 由量子和 P 派生"的公式要把通信项加进去；或 partial 要在 helper 侧先 merge 再回传 | B3：同 B2 的骨架，query 换成 c 行，扫 c ∈ {128…8k}、P ∈ {64k…630k}、w |
| H4 | EP dispatch 的 fan-out 成本随 world 线性但基数小：decode 尺寸消息（每 peer 几十–几百 KB）在 8 rank 上一步 ≤ 几十 µs，外推 36 仍 ≪ 1 ms | EP36 的每步同步税不可忽略，cell 该缩到 EP16/24 | B4：`a2a.cu` 4 rank / 8 rank（跨 tray），消息 16 KB–4 MB |
| H5 | session 睡/醒（6 GB 经 C2C 200 GB/s）与本 tray decode **互不干扰**（C2C 与 HBM 是两条路） | 唤醒要排到 decode 空隙里，策略变复杂 | B6：decode 代理步 + 并发 H2D/D2H 流，看步时抖动 |

间接测法的原则：**字节搬真的，算时用标定 spin 代**。attention kernel 本身（FlashMLA/
FlashKDA）不在 kern 里，直接写 fused 核不值；GEMM 复刻只用来标定 FLOP/s（H1），
其余实验的算时按 H1 的标定值用 `globaltimer` spin 占位，这样测出的增量就是通信与
同步本身。

## 1. 骨架（B2/B3 共用）

一个进程、一个 host 线程、tray 内 4 张卡各一个 stream 与一张图（上午已证 1 线程 ==
N 线程）。每 rank 同时是自己 S 条 session 的 owner 和其他 rank 的 helper（对称）。
每层（24 层 MLA；69 层 KDA/dense/MoE 用一个 `T_other` spin 代）：

1. owner：把本 rank 的 q（S 行）写进每个 peer 的 inbox[r]（peer 指针 D2D 拷贝核）→
   release-store doorbell；
2. helper：等 w 个 owner 的 doorbell（本地 flag）→ 对每个 owner 的 q 扫本地条带
   （spin，时长 = t_attn(C/w, S)）→ partial 写回 owner 的 return[r] → doorbell；
3. owner：等 w 个 return → merge（真实 elementwise，S×96×512×w）；
4. `T_other` spin。

flag 用 generation 计数（设备侧 step 计数器由图尾 kernel 自增，等待目标从它读），
所以图可整张 replay。w=1 时 1–3 退化成本地扫全量 C。指标：每步墙钟 vs w，
减去 w=1 即 DCP 税；再拆 doorbell 等待 vs 拷贝时间（nsys 或 event）。

## 2. 结果（tray13，4×GB300，2026-09-01）

### B1：MLA attention 形状的 GEMM 标定（`mla_gemm.cu`，`b1-mla-gemm.txt`）

`scores = Q[b·96, 576] @ K[n, 576]ᵀ`、`out = P[b·96, n] @ Vt[512, n]ᵀ`，cuBLAS，
K/V 分开读（proxy 读 1088 B/token FP8、2176 B bf16；真 fused 核读 576 B）。

| dtype | b | n=262k：每层 µs | TFLOP/s | µs / (ktok·b) |
|---|---:|---:|---:|---:|
| fp8 | 1 | 70 | 786 | 0.27 |
| fp8 | 4 | 111 | 1975 | 0.11 |
| fp8 | 16 | 323 | 2711 | 0.077 |
| fp8 | 64 | 1164 | **3010** | **0.069** |
| bf16 | 1 | 114 | 482 | 0.43 |
| bf16 | 64 | 1800 | 1947 | 0.107 |

- **H1 被证伪一半**：decode 时每条 session 只有自己的 96 行 query（b=1），attention
  是**带宽/占用绑定**（786 TF/s，4 TB/s），不是算力绑定。每层 0.27 µs/ktok → 219k ctx
  每 token 每 session **1.4 ms**（24 层）。所以 decode attention 的成本模型要按
  0.27 µs/ktok/层算，不是 209 kFLOP / 1.5 PF；全集 decode attention ≈ 55 GPU·h
  （而非 28）。**DCP 分的是这条带宽**——每 rank 总字节守恒，只改延迟与均衡。
- **共享 KV 的 query 批起来便宜 4×**（b=64 时 0.069 µs/ktok·行）：投机解码的 k+1
  行、共享前缀的 sibling 子 agent（trace 里 82% 块共享）都该批到同一次扫描里。
- FP8 GEMM 在这些形状上打到 **3.0 PF/s**（bf16 1.95），比早先假设的 1.5 PF 高——
  extend 侧（大 M）的 FLOP 模型用 3 PF fp8 / 1.9 PF bf16。

### B2：decode 的 DCP 交换税（`dcp.cu`，`b2-smoke.txt`、`b2b3-matrix.txt`）

24 层，每层每 rank：q 广播到 W−1 个 peer（peer D2D 拷贝 + doorbell）→ 为 W 个
owner 各扫本地条带（spin = S × (C/W) × 0.27 µs/ktok）→ partial 回写 + doorbell →
等 W−1 个 return → 真实 LSE merge → `T_other` 300 µs。C = 219k。

| 场景 | W=1 | W=2 | W=4 |
|---|---:|---:|---:|
| 纯交换（算时=0），S=4 / 16 / 64 | — | 0.46 / 0.67 / 1.51 ms | **1.12 / 1.59 / 4.14 ms** |
| 带算时，S=16 | 29.9 ms | 30.6 (+2%) | 31.5 (**+5%**) |
| 带算时，S=4 | 12.9 ms | 13.3 (+3%) | 14.0 (**+8%**) |
| 不平衡：rank0 的 session 630k，其余 219k，S=16 | **72.6 ms**（锁步等最慢） | — | **42.1 ms**（均摊 + 税） |

- **H2 成立但没那么便宜**：DCP4 的税是每层 ~50–70 µs（约 40 µs 固定 = 串行服务
  4 个 owner 的 wait/copy/bell 链 + 每 MB ~1.5 µs），每步 1.1–1.6 ms，占 13–30 ms
  的步 5–8%。W=2 只要 W=4 的 40%。
- **对称满载时 DCP 是纯税**：每 rank 扫的字节守恒（自己的 S 条 × C/W × W 个 owner），
  步时不降。DCP 赚的是（a）**不平衡**——一个 rank 上有 3× 长的 session 时锁步步时
  从 72.6 降到 42.1 ms（−42%）；（b）单 session 超过一张卡的余量；（c）whale chunk
  的前缀扫描分摊。**结论：DCP 按 session 选择性开（长 ctx / 不平衡时），不是全体
  默认；tray 内 W=2 的性价比高于 W=4。**
- 交换的字节：S=16、W=4 每步 q 0.5 GB + partial 0.9 GB（4 个 rank 合计），有效
  ~600 GB/s；merge 本身可忽略。

### B3：extend chunk 的 DCP 流量（`dcp.cu --chunk`，`job1-b3-b3p-b6.txt`、`job3-tray14-*.txt`；tray14）

S=16 条 decode session + rank0 上一条 c 行的 extend（prefix P=219k），partial 按修正后的
**128-dim**（早先一版按 512-dim，数字虚高 4×，已弃）。每 extend token 每层每 helper：
q 37 KB + partial 49 KB = **86 KB**（decode 一行只有 8 KB）。

纯交换（算时=0），每步：

| c | W=2 | W=4 | W=4 字节/步（4 rank 合计） |
|---:|---:|---:|---:|
| 512 | 3.1 ms | 5.3 ms | 4.6 GB（870 GB/s） |
| 2048 | 11.1 ms | 18.1 ms | 14.2 GB（780 GB/s） |
| 8192 | 42.2 ms | 69.6 ms | 52.4 GB（750 GB/s） |

带算时（extend 0.02 µs/(token·ktok)/层 = 3 PF fp8，decode 0.27 µs/ktok/层）：

| c | W=1 | W=2 | W=4 | W=4 加速 / 并行效率 |
|---:|---:|---:|---:|---|
| 512 | 83.8 ms | 59.4 | **48.3**（算 43.4 + 税 4.9） | 1.7× / 43% |
| 2048 | 245 ms | 146 | **98.5**（算 83.7 + 税 14.8） | 2.5× / 62% |

- **H3 成立**：交换随 c 与 W−1 线性（W=2→4 ×1.65），每 extend token 每步
  W=2 4.7 µs / W=4 7.2 µs（= 6.2 MB/token/步 ÷ ~860 GB/s）。按 5 ms filler
  预算、c≈100 token/步：税 0.5–0.7 ms/步（10–15%）——**要把通信项写进 chunk 公式**：
  `t_step(c) = c·P/W·t_ext + c·(W−1)·24·86 KB / 860 GB/s`。降法：q 用 fp8、partial 用 bf16
  各减半（→43 KB/token/helper）。
- 但 **extend 的 DCP 与 decode 不同，是真赚**：decode 每 rank 字节守恒（B2），extend 的
  FLOP 被 W 分摊（c=2048：245 → 98.5 ms）。效率 43–62%，输给的是税 + 未分摊的 T_other
  与 decode 底。
- 单 rank 上 2k chunk 对 220k 前缀一步 245 ms、W=4 也要 98 ms——再次确认 whale 只能
  "多走小步"（c≈100/步，每步 ≤5 ms），不能"走大步"。

### B3p：远端 KV 条带直接读 vs ship q（`kvread.cu`；tray14）

问题：extend 的 owner 是把 q 送到 helper（B3 的做法），还是直接读 peer 上的条带？

SM 流式读 151 MB（262k token × 576 B），plain `ld.global`：

| grid（block×256） | 本地 HBM | peer（同 tray，NVLink） | 差 |
|---:|---:|---:|---:|
| 32 | 293 GB/s | 63 GB/s | 4.6×（延迟绑定） |
| 128 | 1140 | 246 | 4.6× |
| 512 | 3958 | 710 | 5.6× |
| 2048 | **5817** | **739** | **7.9×** |
| `cudaMemcpyPeer` 搬到本地 | — | 805 GB/s | |

- **decode（b=1，带宽绑定 ~4 TB/s）永远在 KV 所在处算**：远端读慢 8×，没有讨论余地。
- extend 的字节交叉点：ship = (W−1)·c·86 KB，拉条带 = (W−1)·(P/W)·576 B
  → **c* = P/(149·W)**（P=219k、W=4 ≈ 370 行）。但拉条带 = owner 独自算全部 FLOP
  （W=1 那列），c 大时 ship + helper 算才有 B3 的 2.5×；c 小时 ship 的字节本来就少。
  **所以拉远端条带在任何 c 下都不赢**——远端读只用于迁移/staging（805 GB/s，与跨 tray
  fabric 710 GB/s 同量级）。attention 的 partial+merge 是唯一路径，模型更简单了。
- **事故**：让 cuBLAS/cublasLt 直接对 peer 显存做 GEMM（`mla_gemm 0 1`，bf16 和 fp8 都
  用 TMA 路径）把 tray13 的两张卡打进 **"GPU requires reset"**（`nvidia-smi -q`：Access
  Timeout Recovery / Route Unhealthy），进程变 defunct，CUDA 只剩 2 张卡可见——job1 里
  W=4 的 "invalid device ordinal" 就是这个原因，不是代码 bug。plain load/store 的自写
  kernel（B2/B3/kvread、fabric barrier）全程没事。~~规则：TMA/bulk-copy 类 kernel 不能
  指向 peer/fabric 映射~~ **09-02 隔离后修正：只有 multicast TMA 挂卡（cuBLAS sm100 用
  它），单播 TMA、1-D bulk copy、bulk reduce、sys 原子跨 tray 全部可用**，见
  `moe-comm-survey.md` §5。（tray13 待有权限的人 reset。）

### B4：EP dispatch 的 fan-out（`a2a.cu`，8 rank = tray14 + tray17 跨 tray，`a2a-run-170417/log0`）

每 rank 向 W−1 个 peer 各推 chunk 字节 + release flag，等 W−1 个 flag，一步：

| chunk/peer | 0 B | 16 KB | 256 KB | 1 MB | 4 MB |
|---|---:|---:|---:|---:|---:|
| W=8 跨 tray，µs/步 | **8.6** | 10.4 | 21.9 | 58.2 | 203.5 |
| egress/rank | — | 11 GB/s | 84 | 126 | **144 GB/s** |

- 固定项 8.6 µs（7 个 flag 的 fan-out + 等待，与 4.8 µs 的 2-rank fabric barrier 一致
  地按 log/线性长），decode 尺寸消息（≤256 KB/peer）**≤ 22 µs/步**。
- 外推 EP36：decode 一步每 rank 的 dispatch 载荷 = S×topk×7 KB / 36 每 peer（S=64、
  topk 8：~100 KB/peer）→ 单次 ~15 µs@W=8，按 fan-out 线性外推 W=36 ~30–40 µs；
  dispatch+combine × ~90 MoE 层 ≈ **每步 5–7 ms**。**H4 只成立一半**：绝对值不大，但
  是 30 ms 步的 15–20%，与 DCP4 税同量级，且随 cell 变大慢慢涨——EP36 vs EP16 的差
  别在这一项上约 1.5×。缓解是 mega 式把 dispatch 融进 GEMM（本来就是 EP36 要
  `mega@36` 验证的原因），或每层双缓冲让 combine 与下一层 dense 重叠。
- 单 rank egress 144 GB/s 是 kernel push 的上限（fabric 裸拷 710 GB/s）；dispatch 的
  字节远达不到，瓶颈是同步不是带宽。

### B6：session 睡/醒的 C2C 流量 vs decode 算（`mla_gemm` + `hostnuma`；tray14）

同一张卡：fp8 GEMM（n=131k）单独 vs 并发 `hostnuma local 0 0`（1 GiB 循环
kernel-read/write + memcpy H2D/D2H，~200 GB/s C2C 打满 40 s，GEMM 2.6 s 全程重叠）：

| | 单独 | 并发 C2C | 差 |
|---|---:|---:|---:|
| b=1（带宽绑定） | 34 µs | 36 µs | +6% |
| b=16（算力绑定） | 174 µs | 179 µs | +3% |

- **H5 成立**：C2C（NVLink-C2C 到 Grace）与 HBM 是两条路，睡/醒 6 GB 的搬运对 decode
  步时影响 ≤6%，不需要排到空隙里。唤醒可以是每步固定的后台流。

## 3. 结论（对 2×EP36 形态）

1. **DCP 的角色分两种**：decode 上是税（W=4 每步 5–8%，字节守恒），只在不平衡/超容量时
   按 session 开；extend 上是真并行（W=4 得 1.7–2.5×），是 whale 摊 FLOP 的机制。
   两者是同一个 partial+merge op（B2/B3 同一骨架），调度器按 span 类型决定 w。
2. **通信永远是 ship q / 回 partial，不读远端 KV**（B3p）：decode 远端读慢 8×；extend
   在任何 c 下 ship + helper 算都优于拉条带。远端读只做迁移。
3. **chunk 公式加通信项**：每 extend token 每步 W=4 约 7 µs（可减半），c≈100/步时
   10–15%；W=2 是 W=4 的 60%。tray 内 W=2/4，不跨 tray（跨 tray 只多 ~1 µs/barrier，
   但每 token 6 MB 的交换走 fabric 会和 EP dispatch 抢 egress）。
4. **EP36 的每步同步税估 5–7 ms**（B4 外推），与 DCP 税同量级、比它更普遍（每步每 rank）；
   这是 EP36 比 EP16 唯一实测到的代价，决策仍取决于 `mega@36` 能否把 dispatch 融掉。
   下一步要在 9 tray 上真跑 W=36 的 `a2a` 而不是外推。
5. **decode attention 是带宽绑定**（B1 b=1 786 TF/s、4 TB/s）：0.27 µs/ktok/层 → 219k
   ctx 每 token 1.4 ms；共享 KV 的行批起来便宜 4×（spec 的 k+1 行、子 agent sibling）。
   全集 decode attention 55 GPU·h（成本模型已按此改）。
6. **C2C 与 HBM 互不干扰**（B6），session 睡/醒不需要调度配合。
7. **硬约束**（09-02 修正）：peer/fabric 映射的显存禁止 multicast TMA（cuBLAS 类 GEMM），
   其余含单播 TMA 与 1-D bulk copy 全部可用，见 `moe-comm-survey.md` §5。

## 4. 未做 / 下一步

- [ ] W=36 真实 fan-out（9 tray `a2a`），替换 B4 的外推。
- [ ] mega@36 的 dispatch 融合是否成立（pegainfer 侧）。
- [ ] B3 的降字节版（q fp8 + partial bf16）与 helper 侧 pre-merge。
- [ ] 真 FlashMLA 核在 b=1 的实测（B1 是 cuBLAS 代理，K/V 分读 1088 B/token，真核 576 B，
  带宽绑定下真核可能快 ~1.8×）。
