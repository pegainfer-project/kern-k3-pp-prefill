// K3 `k3_kda_spec` — the two small kernels around SGLang's KDA verify and
// ReplaySSM fold (source/sglang-kda-spec) in a D speculative round.
//
//   extern "C" __global__ void kern_k3_kda_spec_gate(
//       const f32* partial,    // [B, LDS]  q | k | v | g, then beta's HEADS columns and f_a's 128
//       const bf16* w_f_b,     // [INNER, 128]
//       f32* gate);            // [B, INNER]
//   grid (B, HEADS, 1), block 128
//
// The forget path's second projection, landed as k3_kda_core lands it:
// gate[b, c] = f32(bf16(sum_j f32(bf16(f_a[b, j])) * f32(w_f_b[c, j]))).
// The verify adds dt_bias and runs the gate itself.
//
//   extern "C" __global__ void kern_k3_kda_spec_conv(
//       const bf16* steps,     // [seqs, STEPS, 3 stream, 3 tap, INNER]  the verify's window after each step
//       void* kda_base, const int* line_index, long long line_bytes,
//       const int* nacc);      // [seqs]  steps the round keeps
//   grid (seqs, 1, 1), block 256
//
// Copies sequence n's window after its last kept step into its line, where
// k3_kda_core keeps it (after rec, [stream][tap][c]). nacc 0 keeps the line.
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 12
#endif
#define KD 128
#define INNER (HEADS * KD)
#ifndef LDS
#define LDS (4 * INNER + 256)
#endif
#define FA (4 * INNER + HEADS)
#define STEPS 8
#define WIN (3 * 3 * INNER)

typedef __nv_bfloat16 bf16;

extern "C" __global__ __launch_bounds__(128) void kern_k3_kda_spec_gate(const float* __restrict__ partial,
                                                                       const bf16* __restrict__ w_f_b,
                                                                       float* __restrict__ gate) {
  __shared__ float fa[KD];
  const int b = blockIdx.x, c = blockIdx.y * KD + threadIdx.x;
  fa[threadIdx.x] = __bfloat162float(__float2bfloat16_rn(partial[(size_t)b * LDS + FA + threadIdx.x]));
  __syncthreads();
  const bf16* w = w_f_b + (size_t)c * KD;
  float s = 0.0f;
#pragma unroll 8
  for (int j = 0; j < KD; ++j) s = fmaf(fa[j], __bfloat162float(w[j]), s);
  gate[(size_t)b * INNER + c] = __bfloat162float(__float2bfloat16_rn(s));
}

extern "C" __global__ __launch_bounds__(256) void kern_k3_kda_spec_conv(const bf16* __restrict__ steps,
                                                                       void* kda_base,
                                                                       const int* __restrict__ line_index,
                                                                       long long line_bytes,
                                                                       const int* __restrict__ nacc) {
  const int n = blockIdx.x, keep = nacc[n];
  if (keep <= 0) return;
  const uint4* src = reinterpret_cast<const uint4*>(steps + ((size_t)n * STEPS + keep - 1) * WIN);
  uint4* dst = reinterpret_cast<uint4*>(static_cast<char*>(kda_base) + (long long)line_index[n] * line_bytes +
                                        (long long)HEADS * KD * KD * 4);
  for (int i = threadIdx.x; i < WIN * 2 / 16; i += blockDim.x) dst[i] = src[i];
}
