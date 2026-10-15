// K3 MLA decode, absorb on tensor cores: the same contract as k3_mla_absorb.cu
// (docs/k3-kernel-abi.md K5a), for a decode batch of at most 64 rows.
//
//   extern "C" __global__ void kern_k3_mla_absorb_mma(
//       const f32*  q_partial,   // [B, HEADS*192]  f32 partial, nope 128 | rope 64
//       const bf16* w_kv_b,      // [HEADS*256, 512]  W_UK = rows h*256+0..128
//       bf16*       q_abs,       // [B, HEADS, 576]  latent 512 | rope 64 (e4m3 bytes under Q_FP8)
//       int B);                  // B <= 64
//
//   grid  (1, HEADS, LAT / CS = 4)   block (2 * CS = 256, 1, 1)   smem 0 dynamic (52 KB static)
//
// Block (h, cs) owns columns cs*CS .. +CS of head h for every row: W_UK's
// 128 x CS slice is read once (the scalar kernel re-read it per 8-row group,
// and its 16-way smem reduction made a 16-row step ~13 us for 12 MB),
// staged in shared memory with the batch's bf16 query, and each of the CS / 16
// warps multiplies its 16 columns with wmma 16x16x16 (bf16 in, f32
// accumulate) over ceil(B/16) row tiles. Rows past B are zero in shared
// memory and never stored. Each block also copies its eighth of the rope
// columns (2 threads x 4 columns per row, loaded with everything else).
//
//   q_h     = bf16(q_partial[b, h*192 .. +192])
//   q_abs   = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) for j<512 | q_h[128..192] ]   f32 acc
//
// The f32 sum runs in the tensor core's order, not the scalar kernel's: the
// same products, rounded once to bf16 at the end. Every block's order is
// fixed, so a result never depends on the schedule.
//
// -DQ_FP8: q_abs is the fp8 query of SGLang's fp8 cache, one e4m3 byte an
// element: each bf16 above goes on through f32 to e4m3 by cvt.rn without
// saturation, the aten cast SGLang applies to its bf16 query.
//
//   nvcc -cubin -arch=sm_103a -O3 source/k3_mla_absorb_mma.cu
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

#ifdef Q_FP8
#include <cuda_fp8.h>
typedef unsigned char q_t;
__device__ __forceinline__ void put2(q_t* p, __nv_bfloat162 v) {
  *reinterpret_cast<unsigned short*>(p) = __nv_cvt_float2_to_fp8x2(__bfloat1622float2(v), __NV_NOSAT, __NV_E4M3);
}
#define PUT2(p, i, v) put2((p) + 2 * (i), v)
#else
typedef __nv_bfloat16 q_t;
#define PUT2(p, i, v) (reinterpret_cast<__nv_bfloat162*>(p)[i] = (v))
#endif

#define HEADS 96
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define BMAX 64
#ifndef CS
#define CS 128     // columns per block
#endif
#define THREADS (2 * CS)  // a warp per 16 columns
#define QP (NOPE + 8)  // padded smem rows keep the wmma loads off one bank
#define WP (CS + 8)

extern "C" __global__ void __launch_bounds__(THREADS) kern_k3_mla_absorb_mma(const float* __restrict__ q_partial,
                                                                        const __nv_bfloat16* __restrict__ w_kv_b,
                                                                        q_t* __restrict__ q_abs, int B) {
  __shared__ __align__(32) __nv_bfloat16 qs[BMAX][QP];
  __shared__ __align__(32) __nv_bfloat16 ws[NOPE][WP];
  __shared__ __align__(32) float out[CS / 16][16][16];
  const int h = blockIdx.y, c0 = blockIdx.z * CS, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int tiles = (B + 15) / 16;
  // W slice: 128 rows x CS columns = 16 * CS sixteen-byte vectors, 8 per thread, all in flight.
  const __nv_bfloat16* __restrict__ w = w_kv_b + (size_t)h * 256 * LAT + c0;
  uint4 wv[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
    wv[k] = *reinterpret_cast<const uint4*>(w + (size_t)d * LAT + v * 8);
  }
  // This block's CS / 8 rope columns of every row (ROPE / gridDim.z), a float4 per thread, copied at the end.
  const int rr = t / (CS / 32), rc = blockIdx.z * (CS / 8) + t % (CS / 32) * 4;
  const float4 rv = rr < B ? *reinterpret_cast<const float4*>(q_partial + (size_t)rr * (HEADS * QW) + (size_t)h * QW + NOPE + rc)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
  // The query: up to 64 rows x 128 f32 = 16 float4 per thread, every load in
  // flight before the first store (one after another they cost a round trip each).
  float4 qv[BMAX * NOPE / 4 / THREADS];
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d4 = i % (NOPE / 4);
    qv[k] = r < B ? *reinterpret_cast<const float4*>(q_partial + (size_t)r * (HEADS * QW) + (size_t)h * QW + d4 * 4)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
  }
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d4 = i % (NOPE / 4);
    if (r < tiles * 16) {
      reinterpret_cast<__nv_bfloat162*>(&qs[r][d4 * 4])[0] = __floats2bfloat162_rn(qv[k].x, qv[k].y);
      reinterpret_cast<__nv_bfloat162*>(&qs[r][d4 * 4])[1] = __floats2bfloat162_rn(qv[k].z, qv[k].w);
    }
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
    *reinterpret_cast<uint4*>(&ws[d][v * 8]) = wv[k];
  }
  __syncthreads();
  // The warp's W columns are the same for every row tile: their fragments stay in registers.
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> wf[NOPE / 16];
#pragma unroll
  for (int k = 0; k < NOPE / 16; ++k) wmma::load_matrix_sync(wf[k], &ws[k * 16][warp * 16], WP);
  for (int m = 0; m < tiles; ++m) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#pragma unroll
    for (int k = 0; k < NOPE / 16; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::load_matrix_sync(a, &qs[m * 16][k * 16], QP);
      wmma::mma_sync(acc, a, wf[k], acc);
    }
    wmma::store_matrix_sync(&out[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    // 16 rows x 8 column pairs per warp, 4 per lane
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = lane + k * 32, r = i / 8, j2 = i % 8, b = m * 16 + r;
      if (b < B) {
        q_t* o = q_abs + ((size_t)b * HEADS + h) * ROW + c0 + warp * 16;
        PUT2(o, j2, __floats2bfloat162_rn(out[warp][r][2 * j2], out[warp][r][2 * j2 + 1]));
      }
    }
    __syncwarp();
  }
  if (rr < B) {
    q_t* o = q_abs + ((size_t)rr * HEADS + h) * ROW + LAT + rc;
    PUT2(o, 0, __floats2bfloat162_rn(rv.x, rv.y));
    PUT2(o, 1, __floats2bfloat162_rn(rv.z, rv.w));
  }
}
