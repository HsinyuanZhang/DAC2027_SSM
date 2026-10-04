# C2: Continuous Stateful Carry and Streaming Evaluation

Date: 2026-10-04. Roadmap item C2 (docs/exploration_roadmap_20261004.md, section 5.2). Status: done. Code lives in `ssm_decode/mamba3_streaming.py`. Tests live in `tests/test_mamba3_streaming.py`. Numbers below come from `equivalence.json`, produced by `measure.py` in this directory.

## What was built

1. `forward_with_states(decoder, x, states)` runs an `OfficialMamba3Decoder` chunk with kernel-level state carry. `initial_states(decoder, batch)` returns zero states. `stream_recording(decoder, x, chunk_lens)` runs a long recording as consecutive carried chunks.
2. `core_step` / `decoder_step` are pure-PyTorch single-bin streaming steps. They share the carry-state layout with the kernel path, so kernel chunks and single bins can be interleaved on one state tuple.
3. `tests/test_mamba3_streaming.py` holds the acceptance gate. Ten tests pass. Existing tests still pass (`tests/test_mamba3_official.py`, `tests/test_mamba3_cpu.py`, `tests/test_mamba3_cpu_lora.py`, 19 passed).

## Verified state-carry contract

Read from the pinned kernel source at commit e9594ce (`.tools/mamba_official`), files `mamba3_siso_combined.py`, `mamba3_siso_fwd.py`, `angle_dt.py`, `mamba3_siso_step.py`:

1. `Input_States` is a 4-tuple `(angle, ssm, k, v)` for the SISO kernel. All four entries appear together or none do.
   - `angle`: `(B, H, A)`, `A = d_state // 4` rope angles. It is the running `mod 2*pi` cumsum of `tanh(Angles) * pi * DT` at the start of the chunk. `angle_dt_fwd` consumes it to build `Angles_Cumsum`.
   - `ssm`: `(B, H, headdim_v=64, headdim_qk=d_state)`, fp32 accumulation buffer.
   - `k`: `(B, H, d_state)`. Rotated `K + K_bias` of the last token of the previous chunk, before the `scale = gamma + shifted_gamma` multiply.
   - `v`: `(B, H, 64)`. `V = x` of that same last token.
2. `return_final_states=True` returns `(out, final_angle, final_ssm, final_k, final_v)`. `final_k` is the same pre-scale rotated `K + bias` of the last token, read at index `(seqlen - 1) % chunk_size`. `final_v` is `V[:, -1]`.
3. On resume the kernel adds one trapezoid correction `ssm += v * k * dt[0] * (1 - sigmoid(trap[0]))`, with `dt[0]` and `trap[0]` from the first token of the new chunk. This restores the `shifted_gamma` term that the previous call masked to zero at its sequence end. This is why `final_k` is stored pre-scale: the correction must apply its own coefficient, so a post-scale k would double-count `gamma`.
4. Chunk lengths need not align with the kernel-internal `chunk_size=64`. Masks handle short tails. Each boundary re-factors the boundary token in bf16, which shows up as noise (numbers below).
5. The official `Mamba3.forward` never feeds states into the kernel, only writes cache out. So the carry drives the kernel directly for the core scan, mirroring `ssm_decode/input_adaptation.py`. This follows the fallback the roadmap allows.

## Angle carry precision choice

The public wrapper `mamba3_siso_combined` casts `Angles` to bf16 before the cumsum, so its returned angle state is bf16 and loses about `2^-8` relative angle precision per boundary. `forward_with_states` defaults to `fp32_angles=True`: it calls the same autograd Function (`_Mamba3Function.apply`) that the public wrapper calls, applies the wrapper's bf16 casts to Q, K, V, Trap, and Z itself, and keeps Angles fp32. The carried angle state is then fp32-exact. `fp32_angles=False` selects plain public-wrapper carry for comparison.

## Equivalence numbers

Model: Mamba-3 128x2, state 128, chunk T=2048, batch 2, random init, `measure.py`. Max abs diffs:

1. Chunked carry vs one-shot, same entry, fp32 angle carry:
   - 16 chunks of 128: 4.63e-3.
   - 8 chunks of 256: 4.39e-3.
   - Irregular [200, 512, 331, 1005]: 7.56e-3.
2. Chunked carry vs one-shot, bf16 angle carry (public wrapper): 7.72e-3, 8.78e-3, 8.74e-3 for the same three splits.
3. Carry one-shot vs the untouched module default forward: 7.38e-3 (fp32 angles, caused by the bf16 angle-input cast inside the module path) and 3.67e-3 (bf16 angles, pure same-math noise floor from a different autotune key and return-states path).
4. Angle continuity: chunked final angle vs one-shot final angle 2.86e-6. Both within 5.9e-5 of an exact fp32 `cumsum` reference over 2048 bins. The 5.9e-5 is the accumulated `tanh.approx` PTX error.
5. Step reference vs kernel on GPU, T=256: 1.96e-5 with 0.02-scaled cores. 6.62e-3 at default init. Mixed run (kernel chunk of 192 bins then 64 step bins) vs one-shot kernel: 2.07e-5 scaled, 7.55e-3 default init.
6. Step reference vs `CPUDecoder` on CPU, T=128: 3.10e-6 scaled, 5.06e-3 default init (this seed). See the caveat below.

## Tolerance decision and BF16 caveats

1. The roadmap asked for 1e-3 in float32 terms. The fused kernel is a bf16 kernel. Its own reproducibility floor sits near 4e-3: two calls of the same math through different entry paths differ by that much (item 3 above). Chunked carry adds re-factoring noise on top. The acceptance gate therefore asserts 2e-2 max abs diff for the carry tests and reports the measured value in every failure message. The measured carry numbers are 4.4e-3 to 7.6e-3, that is below one bf16 ulp of the output scale (output max about 2.1, ulp 0.0156).
2. Single-bin stepping against the chunked kernel at default init is dominated by bf16 factoring, not by math error. The init makes `q . k` dots about 60, so one bf16 ulp moves outputs by about 0.1 where sums cancel. Measured drift is seed-dependent, from 5e-3 to 0.30 over 128 bins. The step math itself is exact: with 0.02-scaled cores (the same trick as `tests/test_mamba3_cpu.py`) the step matches `CPUDecoder` to 3.1e-6 and the GPU kernel to 1.96e-5. Tight gates use scaled cores. A loose default-init test pins the measured noise band.
3. `B_bias` and `C_bias` initialize to all ones. A batch-equals-heads test setup can hide per-head bias bugs. The width-128 GPU tests (4 heads, batch 2) exposed this during development. The fixed code indexes the per-head bias exactly as the kernel does.

## Training use

The carry path is differentiable. Gradients flow through the kernel's autograd Function into the carried states. A two-chunk backward smoke test returns finite gradients for all 38 parameter tensors and nonzero gradients at the initial state roots (max 3.74e-3). Continuous chunked training as in C2 step 2 can therefore back-propagate across chunk boundaries through the state tuple.

## Reproduce

```bash
cd /home/xinyuan/Work_host/SSM
export PY=/home/xinyuan/miniconda3/envs/spint/bin/python
PYTHONNOUSERSITE=1 \
PYTHONPATH="/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM" \
TRITON_CACHE_DIR=/tmp/triton_w1 \
CUDA_VISIBLE_DEVICES=1 \
$PY -m pytest tests/test_mamba3_streaming.py -q

PYTHONNOUSERSITE=1 \
PYTHONPATH="/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM" \
TRITON_CACHE_DIR=/tmp/triton_w1 \
CUDA_VISIBLE_DEVICES=1 \
$PY results/analysis/c2_streaming_20261004/measure.py
```

The first run compiles Triton kernels and takes a few minutes. Later runs reuse `TRITON_CACHE_DIR`. GPU tests skip without CUDA. Kernel tests skip without the isolated `mamba_deps` PYTHONPATH.

## Files

1. `ssm_decode/mamba3_streaming.py`: carry API, step reference, contract docstring.
2. `tests/test_mamba3_streaming.py`: CPU unit tests and the GPU acceptance gate.
3. `results/analysis/c2_streaming_20261004/measure.py`: reproduces every number.
4. `results/analysis/c2_streaming_20261004/equivalence.json`: raw numbers.
