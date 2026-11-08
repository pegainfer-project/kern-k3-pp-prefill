# EP dispatch/combine 的候选实现：源码级对比（NVL72 视角）

> Moved from kern's `docs/` on 2026-11-08. Paths in backticks (`tools/`, `docs/`,
> `crates/`, `examples/`) are kern's as they were when this was written.

状态：2026-09-02；同日拍板：**主路径 MegaMoE**，one-sided fallback 暂不立项。对象 = DeepEP v1/v2、DeepGEMM MegaMoE、TRT-LLM one-sided /
two-sided / CuTe MegaMoE、FlashInfer（含 Helix DCP all-to-all）。源码 clone 在
`/tmp/moe-comm/{deepep,deepgemm,trtllm,flashinfer}`（HEAD 浅克隆）。问题：K3@EP36
（或 EP72）decode superstep 里 MoE 层的通信该用哪一套，以及 kern 的 `export`/`peer`
ABI 够不够。

## 1. 一句话结论

**通信原语层面各家没有本质差别**——NVL72 内全是"mapped VA + 普通 st / 1-D bulk copy +
release/acquire flag"，带宽都能到 700+ GB/s。差别在**协议的同步点数**和**有没有 kernel
边界**。decode 尺寸（每 peer 几十–几百 KB）下时间由同步点和 kernel 边界决定，不由带宽
决定，所以 **MegaMoE（DeepGEMM 版）是当前最优**：每层 2–3 个 rank 级 barrier、dispatch
与 FC1 重叠、一层一个 launch。它的代价是模型/架构专属（sm100、K3 形状）和"pull 要先等
count"。**独立的 dispatch/combine 里 TRT one-sided 是最精简的协议**（2 个 barrier、count
融进 dispatch 尾部、纯 128-bit ld/st、有超时），作为 dense/小模型的垫底和 MegaMoE 不适用
时的备选；不需要自写。DeepEP v1 在 72 rank 上直接不可用（`NUM_MAX_NVL_PEERS 8`）；v2
每层 5 个 barrier + NCCL window ABI，优点（4–6 SM、CPU 弹性缓冲）是 prefill/训练的，
不是 decode 的。（注意：pegainfer 树里 vendor 的 DeepEP 是 `d4f41e4`，比本文分析的
HEAD `01dc3aa` 旧：`dispatch_impl` 17 参数 vs 18，barrier 原子 scope 无条件 `sys` vs 按
情况 `gpu`/`sys`。若从它复用 barrier 代码，钉 revision。）

## 2. 对比表

| | DeepEP v1 intranode | DeepEP v2 elastic | DeepGEMM MegaMoE | TRT one-sided | TRT two-sided | TRT CuTe MegaMoE | Helix DCP a2a |
|---|---|---|---|---|---|---|---|
| peer 内存 | cudaIpc / fabric handle 指针数组 | NCCL window（flat VA，4 GB 固定 stride） | 对称 slab：base + offset[72] + rank（`SymBuffer` by value） | VMM fabric，base + rank×stride | 同左 | 同 DeepGEMM（期望 NVSHMEM heap） | MNNVL workspace，base + rank×stride |
| 碰 peer 的指令 | 普通 `st.global` | **1-D `cp.async.bulk` S2G 直写 peer** + `red.add.release.sys` | **1-D `cp.async.bulk` G2S 从 peer 拉**；combine 普通 16 B `st.global`；`atom.sys` | 纯 128-bit ld/st（dispatch push、combine pull） | register `int4` store 到 peer FIFO；TMA 只在本地 | 1-D bulk 拉 + bulk store 推 + **`cp.reduce.async.bulk` 远端归约** | 普通 `int4` store；TMA 变体存在但从未调用 |
| tensor-map TMA 指向 peer | 无 | 无 | **无**（描述符全本地） | 无（CFT 路径 `fabric.try_put` 走 LE，不走 VA） | 无 | 无 | 无 |
| 每层 rank 级同步 | 2 + host 回读 count | **5**（dispatch 3 含 count 往返；combine 2） | **2–3**（count 推送+atom → barrier → 拉；combine push → barrier → 本地 reduce） | **2**（count 融进 dispatch 尾；flag epoch） | 2 + 独立 prepare 阶段 | 同 DeepGEMM | 1（LL128 flag 内嵌 + credit） |
| 与 GEMM 重叠 | 无 | 无（PDL epilogue 只重叠 permute） | **有**：per-BLOCK_M 到达计数，FC1 TMA warp `ld.acquire` 等 | 无 | 无 | 有 | n/a |
| 每层 kernel 数 | 2 + GEMM 链 | 4 + GEMM 链 | **1** | 2 + GEMM 链（+sanitize） | 3 + GEMM 链 | 1 | 1 |
| 接收布局 | rank-major | rank-major → epilogue 转 expert-major 对齐（GEMM 直读） | expert-major 环形池，按 BLOCK_M 填充，源 rank 轮转交错 | rank-major 无 topk 重复；GEMM 靠 indirection 表 | FIFO → 重排 | expert-major 本地池 | 转置 |
| grid 形状 | 静态 | 静态（cooperative + cluster 2） | **静态 = SM 数**（persistent） | = token 数 | 静态 | 静态 | 静态 |
| 图可捕获 | normal 否（host 回读） | cached_mode 是 | 是（pegainfer EP1 已捕；EP4 尚 eager） | 是 | 是 | 是 | 是 |
| SM 占用 | 24 | 4–6 | 全部（融合） | 全部（按 token） | ~半 | 全部 | 全部 |
| 超时 | trap | trap | **无** | `clock64` 预算到期 trap + host watchdog | 无 | 无 | 无 |
| rank 上限 | **8** | 无硬限 | 模板参数，`kNumMaxRanks 72` | **64**（`kMaxRanks`） | — | — | — |
| 运行时依赖 | NVSHMEM 全局符号（internode） | `ncclDevComm`/`ncclWindow` by value，无全局符号 | **无**：指针 + 计数器 + 模板常量 | 无：by-value 指针表 `recv_buffers[128][4]` | 无 | 无（offset 表） | workspace 内持久 head/tail |
| 已知数字 | H800 EP8 153 GB/s；LL EP64 173/314 µs | SM100 EP8 726/740 GB/s@64 SM；无 µs | 无公开；pegainfer GB300 EP1：mega vs 链 −5…−8%/层 | GB200 EP64 bs≤64：**dispatch 20–29 / combine 33–38 µs** | — | — | — |

## 3. 为什么 MegaMoE 在 decode 上赢

decode 尺寸每层的通信时间 ≈ 同步点 × barrier 延迟 + kernel 边界。TRT one-sided 在
GB200 EP64 测得每层 55–65 µs（bs 1–64 几乎不随 bs 变，即纯同步开销），K3 ~92 层 → 每步
5–6 ms，与我们 `dcp-bench.md` B4 从 a2a 外推的 5–7 ms 一致。这 55 µs 里数据搬运不到
5 µs，其余是两次 rank 级 flag 等待和 dispatch → GEMM → combine 之间的 kernel 边界
（launch 间隙、tail、drain/fill）。MegaMoE 把边界全部去掉，barrier 数持平，dispatch 拉取
与 FC1 的 tile 计算按 BLOCK_M 粒度流水。它的 count-first（pull 前必须收齐所有 peer 的
count）在 decode 上不亏：数据本来就小，等 count 与等 flag 是同一次 barrier。

DeepEP v2 的 5 个 barrier 是硬伤（entry/exit barrier 各一对 + count 往返）；它省的是
SM（一个 elected lane 发 bulk store，无接收方 block），这在 prefill 大批量下有价值，在
decode 下没有。

## 4. 对 kern 的含义

1. **kernel 来源定为 DeepGEMM MegaMoE（pegainfer 的 vendored/AOT fork，
   `csrc/k3/k3_mega_moe_sm100.cu`）**，作为 manifest 的一个 op。runtime 侧不需要新东西：
   slab 是一个 `export` 的 workspace buffer，`peer` 数组给出各 rank 的 base，rank 来自
   `{"rank"}`。kernel 要的 `SymBuffer{base, offset[72], rank}` 由 host 在 load 时从 peer
   数组算出（`offset[r] = peer[r] − peer[me]`），作为 by-value 参数；或者在 fork 里把
   `SymBuffer` 改成读指针数组（一行改动，我们有源码）。**要求每 rank 的 slab 是一次分配**
   （offset 对整块成立）。
2. **超时要自己补**：MegaMoE 的 `spin_wait` / NVLink barrier / grid sync 全是无限循环。
   fork 里加 `globaltimer` 预算 + 错误码写 carry buffer（符合 multi-gpu.md 的故障模型）。
3. **fallback = TRT one-sided 的两个 kernel**（`moeA2ADispatchKernel` / `moeA2ACombineKernel`）：
   无运行时依赖、by-value 指针表、自带超时；`kMaxRanks` 64 → 72 要改常量重编。给 dense
   小模型或 MegaMoE 没覆盖的 dtype 用。**不自写。**
4. **DCP 的 (O, LSE) 回传可以照抄 Helix**：flag 内嵌在 128 B 载荷里（LL128）省掉一次
   barrier，credit 流控；但它一次只做一个方向、占全部 SM，我们 B2 骨架的 doorbell 版
   已经是同样的形状，只需借 LL128 的 flag-in-payload。
5. **硬约束已隔离清楚**（§5）：挂卡的只有 multicast TMA；MegaMoE 用的 1-D bulk 拉/推、
   bulk reduce、sys 原子跨 tray 全部可用，非 multicast 的 tensor-map TMA 也可用。
   MegaMoE 跨 tray 没有障碍。

## 5. peer/fabric 显存上的 async-proxy 实验（2026-09-02，tray18；tray17 作跨 tray 对端）

程序见本地存档 `bench-results/2026-09-02-peer-bulk/peerbulk.cu`（不在仓库），日志 `p1..p5-*.log`。发起方
GPU 对 256 MiB 的目标区域做每项操作并逐位校验；三种映射：同 tray peer access
（`cudaDeviceEnablePeerAccess` + `cudaMalloc`）、同进程 `cuMem` FABRIC 导入、跨 tray
FABRIC 导入。

| 操作 | 本地 | 同 tray peer | 同 tray fabric | 跨 tray fabric |
|---|---:|---:|---:|---:|
| 普通 ld / st 128 MiB | 3.1 TB/s | 731 / 695 GB/s | 687 / 561 | 734 / 696 |
| 1-D `cp.async.bulk` G2S 拉 / S2G 推 | 3.0 TB/s | 746 / 688 | 700 / 556 | **742 / 689** |
| `cp.reduce.async.bulk add.f32` | PASS | PASS | PASS | PASS |
| `red.release.sys` / `atom.sys` / `ld.acquire.sys` | PASS | PASS | PASS | PASS |
| tensor-map TMA 2D load / store（单 tile） | PASS | PASS | PASS | PASS |
| tensor-map TMA 1024 CTA 满并发 1.3 GiB | 2.6 TB/s | **747 GB/s PASS** | — | — |
| **tensor-map TMA `.multicast::cluster`（cluster 2）** | PASS | **挂：发起方 GPU requires reset** | — | — |
| 昨天的 `mla_gemm 0 1`（cuBLAS，K/V 与 3 GB 中间张量在 peer） | — | **挂：发起方 GPU requires reset** | — | — |

- 两次挂卡都是**发起方**的卡进入 "Access Timeout Recovery / Route Unhealthy: GPU
  requires reset"，进程 `<defunct>` 占着显存，CUDA 枚举少一张卡（编号前移，别再用
  `CUDA_VISIBLE_DEVICES` 按 nvidia-smi 编号裁）。被访问的卡无事。
- 所以昨天 "TMA 打 peer 会挂" 的归因是错的：**单播 TMA 满并发也没事，只有 multicast TMA
  挂**。cuBLAS sm100 GEMM（2-CTA UMMA + multicast load）正是这条路。
- 结论对设计：MegaMoE 的全部远端原语（1-D bulk 拉、bulk store / `cp.reduce.async.bulk`
  推、`atom.sys`）跨 tray 可用，带宽与普通 ld/st 持平（~740 GB/s，NVLink5 单向 900）。
  verifier 只需拒绝带 `.MULTICAST` 的 UTMALDG/UTMASTG，以及禁止 extern op 接 peer buffer。
- 代价：tray18 bus 0008（GPU0）与 0018（GPU2）需要特权 reset；tray13 两张卡同样。

## 6. 待做

- [ ] MegaMoE W=36 的 barrier 成本实测（每层 2 个 × 92 层；W=8 跨 tray 单次 8.6 µs）。
- [ ] `SymBuffer` 改指针数组 + spin 超时的 fork 补丁；EP4 下图捕获（pegainfer 目前 eager）。
- [ ] verifier 的 `.MULTICAST` SASS 拼写在 sm_103 cuobjdump 上核对。
