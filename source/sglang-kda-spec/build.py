#!/usr/bin/env python3
"""Compile the vendored SGLang KDA verify and ReplaySSM fold for one D member into cubins.

    python3 build.py --tp 8 --out <dir>

writes `<dir>/k3_kda_verify+HEADS=<hl>.cubin` and
`<dir>/k3_kda_fold+HEADS=<hl>.cubin` and prints each kernel's entry, parameters,
block and shared memory as JSON. Needs nvidia-cutlass-dsl 4.8.0 and Triton
3.8.0 (SGLang's image); no GPU.

Every operand is a fixed view of a buffer the D manifest owns, so every layout
is static and the kernels take bare pointers (PROVENANCE.md, "ABI"):

- the KDA line: a member's `hl` heads of recurrent state f32 [hl][128 v][128 k],
  then the conv window bf16 [3 stream q|k|v][3 tap][hl*128];
- the qkvg partial: f32 rows of q | k | v | g, then beta's hl columns and f_a's;
- the verify scratch, a row per sequence: the per-step conv windows
  [8 step][3][3][hl*128] bf16 and the ReplaySSM ring.
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
HEADS, D = 96, 128
STEPS = 8                      # an anchor and seven drafts
RING = STEPS                   # the ring holds one verify's steps
LOWER_BOUND = -5.0
ONORM_EPS = 1e-5
BLOCK_THREADS = 256


def geometry(tp):
    hl = HEADS // tp
    inner = hl * D
    rec = hl * D * D * 4
    line = rec + 3 * 3 * inner * 2
    lds = 4 * inner + 256
    return dict(hl=hl, inner=inner, rec=rec, line=line, lds=lds,
                conv_row=3 * 3 * inner,
                ring=dict(rawv=hl * RING * D * 2, rawk=hl * RING * D * 2, g=hl * RING * D * 4, beta=hl * RING * 4))


def ring_offsets(g):
    r = g["ring"]
    return {"rawv": 0, "rawk": r["rawv"], "g": r["rawv"] + r["rawk"], "beta": r["rawv"] + r["rawk"] + r["g"],
            "row": sum(r.values())}


def build_verify(tp, out):
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_103a")
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_ptr
    import cuda.bindings.driver as cuda
    sys.path.insert(0, str(HERE))
    import kda_decode_mtp as m

    g = geometry(tp)
    hl, inner, lds = g["hl"], g["inner"], g["lds"]
    # An extent bounds nothing (no checks): the cubin is the same at every batch. Not 1, which CuTe folds away.
    seqs = 1 << 16
    T = seqs * STEPS
    slots = 1 << 20
    line_f32, line_bf16 = g["line"] // 4, g["line"] // 2
    conv_off = g["rec"] // 2
    ring = ring_offsets(g)
    L = cute.make_layout

    @cute.jit
    def verify(kda: cute.Pointer, partial: cute.Pointer, cw: cute.Pointer, gate: cute.Pointer,
               dt_bias: cute.Pointer, a_log: cute.Pointer, gamma_o: cute.Pointer, out_: cute.Pointer,
               line_index: cute.Pointer, conv_steps: cute.Pointer, ring_: cute.Pointer, stream: cuda.CUstream):
        f32 = lambda p, off: cute.make_ptr(cutlass.Float32, p.toint() + off * 4, cute.AddressSpace.gmem, assumed_align=16)
        b16 = lambda p, off: cute.make_ptr(cutlass.BFloat16, p.toint() + off * 2, cute.AddressSpace.gmem, assumed_align=16)
        T_ = cute.make_tensor
        x = lambda s: T_(f32(partial, s * inner), L((1, T, hl, D), stride=(0, lds, D, 1)))
        w = lambda s: T_(f32(cw, s * 4 * inner), L((inner, 4), stride=(1, inner)))
        cs = lambda s: T_(b16(kda, conv_off + s * 3 * inner), L((slots, inner, 3), stride=(line_bf16, 1, inner)))
        steps = lambda s: T_(b16(conv_steps, s * 3 * inner), L((seqs, STEPS, inner, 3),
                                                                 stride=(STEPS * g["conv_row"], g["conv_row"], 1, inner)))
        rrow = ring["row"]
        m._run_kda_decode_mtp_dspark(
            T_(f32(kda, 0), L((slots, hl, D, D), stride=(line_f32, D * D, D, 1))),
            x(0), x(1), x(2), w(0), w(1), w(2), cs(0), cs(1), cs(2),
            T_(f32(a_log, 0), L((hl,))),
            T_(f32(gate, 0), L((1, T, hl, D), stride=(0, inner, D, 1))),
            T_(f32(dt_bias, 0), L((inner,))),
            T_(f32(partial, 4 * inner), L((1, T, hl), stride=(0, lds, 1))),
            T_(b16(out_, 0), L((1, T, hl, D), stride=(0, inner, D, 1))),
            steps(0), steps(1), steps(2),
            T_(cute.make_ptr(cutlass.Int32, line_index.toint(), cute.AddressSpace.gmem, assumed_align=4), L((seqs,))),
            T_(b16(ring_, ring["rawv"] // 2), L((seqs, hl, RING, D), stride=(rrow // 2, RING * D, D, 1))),
            T_(b16(ring_, ring["rawk"] // 2), L((seqs, hl, RING, D), stride=(rrow // 2, RING * D, D, 1))),
            T_(f32(ring_, ring["g"] // 4), L((seqs, hl, RING, D), stride=(rrow // 4, RING * D, D, 1))),
            T_(f32(ring_, ring["beta"] // 4), L((seqs, hl, RING), stride=(rrow // 4, RING, 1))),
            T_(f32(partial, 3 * inner), L((1, T, hl, D), stride=(0, lds, D, 1))),
            T_(f32(gamma_o, 0), L((D,))),
            scale=D ** -0.5, H=hl, N=seqs, NUM_SPEC=STEPS - 1, BLOCK_THREADS=BLOCK_THREADS,
            lower_bound=LOWER_BOUND, CACHE_RING=True, APPLY_ONORM=True, onorm_eps=ONORM_EPS, stream=stream)

    ptr = lambda dt: make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)
    dump = pathlib.Path(out) / f"cute-{tp}"
    dump.mkdir(parents=True, exist_ok=True)
    cute.compile(verify, *(ptr(dt) for dt in (cutlass.Float32,) * 7 + (cutlass.BFloat16, cutlass.Int32,
                                                                         cutlass.BFloat16, cutlass.Int8)),
                 cuda.CUstream(0), options=f"--keep-cubin --dump-dir={dump}")
    (cubin,) = dump.glob("*.cubin")
    name = f"k3_kda_verify+HEADS={hl}"
    cubin.rename(pathlib.Path(out) / f"{name}.cubin")
    dump.rmdir()
    t = STEPS
    smem = (3 * t * D + t + t * D) * 4 + 2 * 64 * (D + 8) * 4 + t * D * 4 + (t + 1) * D * 4 + 256
    return {"module": name, "block": [BLOCK_THREADS, 1, 1], "smem": smem, "grid": [hl, "seqs", 1]}


def build_fold(tp, out):
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"  # no source paths in the cubin
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    sys.path.insert(0, str(HERE))
    import kda_replayssm_spec_decode as f

    g = geometry(tp)
    ring = ring_offsets(g)
    hl = g["hl"]
    const = dict(stride_state_slot=g["line"] // 4, stride_rawv_slot=ring["row"] // 2, stride_rawk_slot=ring["row"] // 2,
                 stride_gk_slot=ring["row"] // 4, stride_beta_slot=ring["row"] // 4,
                 stride_state_layer=0, stride_rawv_layer=0, stride_rawk_layer=0, stride_gk_layer=0,
                 stride_beta_layer=0, stride_indices=1, stride_accept=1, stride_track=0, stride_steps=0,
                 H=hl, HV=hl, K=D, V=D, BK=D, BV=32, MAX_CACHE_LEN=RING, USE_QK_L2NORM_IN_KERNEL=True,
                 NULL_BLOCK_ID=-1, HAS_TRACK=False)
    sig = {"h0": "*fp32", "rawv_cache": "*bf16", "rawk_cache": "*bf16", "gk_cache": "*fp32", "beta_cache": "*fp32",
           "ssm_state_indices": "*i32", "accept_lens": "*i32", "mamba_track_indices": "*i32",
           "mamba_steps_to_track": "*i32", **{k: "constexpr" for k in const}}
    src = ASTSource(fn=f.kda_replayssm_exact_fold_kernel, signature=sig, constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 103, 32), options={"num_warps": 1, "num_stages": 3})
    name = f"k3_kda_fold+HEADS={hl}"
    (pathlib.Path(out) / f"{name}.cubin").write_bytes(k.asm["cubin"])
    return {"module": name, "entry": k.metadata.name, "block": [32 * k.metadata.num_warps, 1, 1],
            "smem": k.metadata.shared, "grid": [D // 32, "seqs", hl]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["verify", "fold"])
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.only != "fold":
        print(json.dumps(build_verify(a.tp, a.out)))
    if a.only != "verify":
        print(json.dumps(build_fold(a.tp, a.out)))


if __name__ == "__main__":
    main()
