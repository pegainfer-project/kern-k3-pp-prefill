// kda_spec.cu: a D speculative round's KDA (source/k3_kda_spec.cu around the
// vendored SGLang verify and fold, source/sglang-kda-spec) against eight
// k3_kda_core steps, the TP8 D build, on one GPU.
//
// N sequences each feed 8 random partial rows to lines that are a random
// permutation of the slots. The reference runs k3_kda_core once per step and
// keeps every line after each. The round runs the gate, the verify over all
// 8N rows, then, each sequence keeping a random 1..8 of its steps, the fold
// and the conv commit. Checked:
//   - the verify leaves every line as it was (it only reads them);
//   - its 8 outputs per sequence against k3_kda_core's steps (relative RMS);
//   - each committed line's state against the reference after its kept steps
//     (relative RMS), its conv window bit for bit, the spare slot untouched.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a kda_spec.cu -o kda_spec -lcuda
//   ./kda_spec --core <k3_kda_core+HEADS=12+KDA_CONV=1+LDS=6400+...cubin> \
//              --verify <k3_kda_verify+HEADS=12.cubin> --fold <k3_kda_fold+HEADS=12.cubin> \
//              --glue <k3_kda_spec+HEADS=12.cubin> --N 32
//
// Exit 0 PASS, 1 FAIL, 2 harness or driver error.
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

typedef uint16_t bf16;

static const int HEADS = 12, KD = 128, INNER = HEADS * KD, LDS = 4 * INNER + 256, STEPS = 8, RING = 8;
static const long long REC = (long long)HEADS * KD * KD * 4, WIN = 3LL * 3 * INNER * 2, LINE = REC + WIN;
static const long long RAWV = (long long)HEADS * RING * KD * 2, RAWK = RAWV, RG = (long long)HEADS * RING * KD * 4,
                       RBETA = (long long)HEADS * RING * 4, RROW = RAWV + RAWK + RG + RBETA;
static const int VERIFY_BLOCK = 256, VERIFY_SMEM = 95008, FOLD_SMEM = 128;
static const double OUT_TOL = 2e-2, REC_TOL = 1e-2;

static void cu(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  std::fprintf(stderr, "%s: %s\n", what, s ? s : "?");
  std::exit(2);
}
#define CU(x) cu((x), #x)

static CUdeviceptr put(const void* h, size_t n) {
  CUdeviceptr d;
  CU(cuMemAlloc(&d, n));
  CU(cuMemcpyHtoD(d, h, n));
  return d;
}

static std::vector<uint8_t> get(CUdeviceptr d, size_t n) {
  std::vector<uint8_t> h(n);
  CU(cuMemcpyDtoH(h.data(), d, n));
  return h;
}

static bf16 f2b(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float b2f(bf16 b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static CUfunction fn(const std::string& cubin, const char* name) {
  CUmodule m;
  CU(cuModuleLoad(&m, cubin.c_str()));
  CUfunction f;
  if (name) {
    CU(cuModuleGetFunction(&f, m, name));
    return f;
  }
  unsigned n = 0;
  CU(cuModuleGetFunctionCount(&n, m));
  if (n != 1) { std::fprintf(stderr, "%s: %u entries, want one\n", cubin.c_str(), n); std::exit(2); }
  CU(cuModuleEnumerateFunctions(&f, 1, m));
  return f;
}

static double rel_rms(const float* a, const float* b, size_t n) {
  double d = 0, r = 0;
  for (size_t i = 0; i < n; ++i) {
    d += (double)(a[i] - b[i]) * (a[i] - b[i]);
    r += (double)b[i] * b[i];
  }
  return std::sqrt(d / std::max(r, 1e-30));
}

int main(int argc, char** argv) {
  std::string core, verify, fold, glue;
  int N = 32;
  unsigned seed = 1234;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--core") core = argv[i + 1];
    else if (a == "--verify") verify = argv[i + 1];
    else if (a == "--fold") fold = argv[i + 1];
    else if (a == "--glue") glue = argv[i + 1];
    else if (a == "--N") N = std::atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)std::atoi(argv[i + 1]);
    else { std::fprintf(stderr, "unknown option `%s`\n", argv[i]); return 2; }
  }
  if (core.empty() || verify.empty() || fold.empty() || glue.empty() || N < 1) {
    std::fprintf(stderr, "usage: kda_spec --core <cubin> --verify <cubin> --fold <cubin> --glue <cubin> [--N n]\n");
    return 2;
  }
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CUfunction f_core = fn(core, "kern_k3_kda_core"), f_verify = fn(verify, nullptr);
  CUfunction f_fold = fn(fold, "kda_replayssm_exact_fold_kernel");
  CUfunction f_gate = fn(glue, "kern_k3_kda_spec_gate"), f_conv = fn(glue, "kern_k3_kda_spec_conv");
  CU(cuFuncSetAttribute(f_verify, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, VERIFY_SMEM));

  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto f32 = [&](size_t n, float mean, float sd) {
    std::vector<float> v(n);
    for (auto& x : v) x = mean + sd * n01(rng);
    return v;
  };
  const int T = N * STEPS, slots = N + 1;
  std::vector<int> perm(slots);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int> line_index(perm.begin(), perm.begin() + N);
  const int spare = perm[N];
  std::vector<int> nacc(N);
  for (int n = 0; n < N; ++n) nacc[n] = n == 0 ? 1 : n == 1 ? STEPS : 1 + (int)(rng() % STEPS);

  auto partial = f32((size_t)T * LDS, 0.f, 1.f), cw = f32(3 * 4 * (size_t)INNER, 0.f, 0.5f);
  std::vector<bf16> w_f_b((size_t)INNER * KD);
  for (auto& x : w_f_b) x = f2b(0.044f * n01(rng));
  auto dt_bias = f32(INNER, 0.f, 1.f), a_log = f32(HEADS, 0.f, 0.5f), gamma_o = f32(KD, 1.f, 0.1f);
  std::vector<uint8_t> lines((size_t)slots * LINE);
  for (int s = 0; s < slots; ++s) {
    float* r = (float*)&lines[(size_t)s * LINE];
    for (long long i = 0; i < REC / 4; ++i) r[i] = 0.1f * n01(rng);
    bf16* w = (bf16*)&lines[(size_t)s * LINE + REC];
    for (long long i = 0; i < WIN / 2; ++i) w[i] = f2b(n01(rng));
  }

  CUdeviceptr dp = put(partial.data(), partial.size() * 4), dcw = put(cw.data(), cw.size() * 4);
  CUdeviceptr dwfb = put(w_f_b.data(), w_f_b.size() * 2), ddt = put(dt_bias.data(), dt_bias.size() * 4);
  CUdeviceptr dal = put(a_log.data(), a_log.size() * 4), dgo = put(gamma_o.data(), gamma_o.size() * 4);
  CUdeviceptr didx = put(line_index.data(), N * 4), dnacc = put(nacc.data(), N * 4);
  int zero = 0, span = 0;
  CUdeviceptr dat = put(&zero, 4);
  long long lb = LINE;

  // Reference: eight decode steps, step t's rows gathered from the round's.
  CUdeviceptr dref = put(lines.data(), lines.size()), dstep, dout;
  CU(cuMemAlloc(&dstep, (size_t)N * LDS * 4));
  CU(cuMemAlloc(&dout, (size_t)N * INNER * 2));
  std::vector<std::vector<uint8_t>> after(STEPS);
  std::vector<float> ref_out((size_t)T * INNER);
  for (int t = 0; t < STEPS; ++t) {
    for (int n = 0; n < N; ++n)
      CU(cuMemcpyDtoD(dstep + (size_t)n * LDS * 4, dp + ((size_t)n * STEPS + t) * LDS * 4, (size_t)LDS * 4));
    CUdeviceptr dwsm = dstep + (CUdeviceptr)4 * INNER * 4;
    void* a[] = {&dstep, &dcw, &dwsm, &dwfb, &ddt, &dal, &dgo, &dref, &didx, &lb, &dout, &N, &dat, &span};
    CU(cuLaunchKernel(f_core, N, HEADS, 1, 128, 1, 1, 0, 0, a, nullptr));
    CU(cuCtxSynchronize());
    after[t] = get(dref, lines.size());
    auto o = get(dout, (size_t)N * INNER * 2);
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < INNER; ++c)
        ref_out[((size_t)n * STEPS + t) * INNER + c] = b2f(((const bf16*)o.data())[(size_t)n * INNER + c]);
  }

  // The round.
  CUdeviceptr dlines = put(lines.data(), lines.size()), dgate, dvout, dsteps, dring;
  CU(cuMemAlloc(&dgate, (size_t)T * INNER * 4));
  CU(cuMemAlloc(&dvout, (size_t)T * INNER * 2));
  CU(cuMemAlloc(&dsteps, (size_t)N * STEPS * WIN));
  CU(cuMemAlloc(&dring, (size_t)N * RROW));
  {
    void* a[] = {&dp, &dwfb, &dgate};
    CU(cuLaunchKernel(f_gate, T, HEADS, 1, 128, 1, 1, 0, 0, a, nullptr));
  }
  {
    CUdeviceptr h0 = dlines, xq = dp, xk = dp + INNER * 4, xv = dp + 2 * INNER * 4;
    CUdeviceptr wq = dcw, wk = dcw + 4 * INNER * 4, wv = dcw + 8 * INNER * 4;
    CUdeviceptr csq = dlines + REC, csk = csq + 3 * INNER * 2, csv = csq + 6 * INNER * 2;
    CUdeviceptr beta = dp + 4 * INNER * 4, onorm = dp + 3 * INNER * 4;
    CUdeviceptr sq = dsteps, sk = dsteps + 3 * INNER * 2, sv = dsteps + 6 * INNER * 2;
    CUdeviceptr rv = dring, rk = dring + RAWV, rg = rk + RAWK, rb = rg + RG;
    void* a[] = {&h0, &xq, &xk, &xv, &wq, &wk, &wv, &csq, &csk, &csv, &dal, &dgate, &ddt, &beta, &dvout,
                 &sq, &sk, &sv, &rv, &rk, &rg, &rb, &onorm, &dgo, &didx};
    CU(cuLaunchKernel(f_verify, HEADS, N, 1, VERIFY_BLOCK, 1, 1, VERIFY_SMEM, 0, a, nullptr));
  }
  CU(cuCtxSynchronize());
  const bool kept = get(dlines, lines.size()) == lines;
  {
    CUdeviceptr rv = dring, rk = dring + RAWV, rg = rk + RAWK, rb = rg + RG, null = 0;
    void* a[] = {&dlines, &rv, &rk, &rg, &rb, &didx, &dnacc, &didx, &dnacc, &null, &null};
    CU(cuLaunchKernel(f_fold, KD / 32, N, HEADS, 32, 1, 1, FOLD_SMEM, 0, a, nullptr));
    void* c[] = {&dsteps, &dlines, &didx, &lb, &dnacc};
    CU(cuLaunchKernel(f_conv, N, 1, 1, 256, 1, 1, 0, 0, c, nullptr));
  }
  CU(cuCtxSynchronize());

  auto vo = get(dvout, (size_t)T * INNER * 2);
  std::vector<float> spec_out((size_t)T * INNER);
  for (size_t i = 0; i < spec_out.size(); ++i) spec_out[i] = b2f(((const bf16*)vo.data())[i]);
  double out_max = 0;
  for (int r = 0; r < T; ++r)
    out_max = std::max(out_max, rel_rms(&spec_out[(size_t)r * INNER], &ref_out[(size_t)r * INNER], INNER));

  auto got = get(dlines, lines.size());
  double rec_max = 0;
  long bad_win = 0;
  for (int n = 0; n < N; ++n) {
    const size_t at = (size_t)line_index[n] * LINE;
    const auto& want = after[nacc[n] - 1];
    rec_max = std::max(rec_max, rel_rms((const float*)&got[at], (const float*)&want[at], REC / 4));
    bad_win += std::memcmp(&got[at + REC], &want[at + REC], WIN) != 0;
  }
  const bool spare_kept = !std::memcmp(&got[(size_t)spare * LINE], &lines[(size_t)spare * LINE], LINE);
  const bool ok = kept && spare_kept && !bad_win && out_max < OUT_TOL && rec_max < REC_TOL;
  std::printf("RESULT\tkda_spec\tN=%d\tverify_kept_lines=%d\tout_relrms_max=%.3g\trec_relrms_max=%.3g\t"
              "win_diff=%ld\tspare_kept=%d\t%s\n",
              N, kept, out_max, rec_max, bad_win, spare_kept, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
