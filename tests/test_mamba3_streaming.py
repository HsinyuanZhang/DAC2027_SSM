"""C2 streaming tests: kernel state carry and the pure-PyTorch step reference.

CPU tests check the step reference against the fixed-window CPU port in
ssm_decode/mamba3_cpu.py.  GPU tests (run with CUDA_VISIBLE_DEVICES set to a
free GPU) are the C2 acceptance gate: chunked-with-carry must match the
one-shot kernel forward, the carried angle state must stay continuous, and
the step reference must match the kernel on short sequences.

Tolerances: the fused kernel materializes Q/K/V/Trap/Z and its output in
BF16, so two valid evaluations of the same math differ at BF16 scale.  The
measured noise floor appears in results/analysis/c2_streaming_20261004.
"""
import math

import pytest
import torch

from ssm_decode.mamba3_streaming import (
    decoder_step,
    forward_with_states,
    initial_states,
    stream_recording,
    streaming_reference,
)

TWO_PI = 2.0 * math.pi
# BF16 kernel noise floor: same-shape calls of the same kernel path differ by
# ~4e-3; chunk boundaries add re-factoring noise.  Single-bin bf16 stepping
# against the chunked kernel at default init scale diverges at ~0.15 over 64+
# bins (K_bias init makes q.k dots ~60, so one bf16 ulp moves outputs ~0.1).
# Scaled cores (0.02, like tests/test_mamba3_cpu.py) isolate the math.
# See results/analysis/c2_streaming_20261004 for measured values.
TOL_CARRY = 2e-2
TOL_STEP_SCALED = 1e-3
TOL_STEP_DEFAULT_INIT = 5e-1


def _scale_cores(decoder, factor=0.02):
    with torch.no_grad():
        for block in decoder.blocks:
            block.ssm.in_proj.weight.mul_(factor)
    return decoder


def _maybe_decoder(*args, **kwargs):
    from ssm_decode.mamba3_official import build_mamba3_official
    try:
        return build_mamba3_official(*args, **kwargs)
    except RuntimeError as exc:  # missing clone or non-isolated triton
        pytest.skip(f"official Mamba-3 runtime unavailable: {exc}")


def _cuda_decoder():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the streaming carry gate")
    return _maybe_decoder(8, 2, 128, 2, 128).cuda().eval()


def test_cpu_step_matches_fixed_window_port():
    decoder = _scale_cores(_maybe_decoder(3, 2, 64, 1, 32).eval())
    from ssm_decode.mamba3_cpu import CPUDecoder
    cpu = CPUDecoder.from_state_dict(decoder.state_dict())
    torch.manual_seed(11)
    x = torch.randn(2, 128, 3)
    for length in (1, 7, 64, 100, 128):
        streamed = streaming_reference(decoder, x[:, :length])
        windowed = cpu.forward(x[:, :length])
        diff = float((streamed - windowed).abs().max())
        assert diff < TOL_STEP_SCALED, f"T={length}: max abs diff {diff}"
        assert torch.isfinite(streamed).all()


def test_cpu_step_default_init_documents_bf16_stepping_noise():
    # With default init the fused path quantizes large q/k dots (~60) in bf16,
    # so a single-bin factoring drifts from the fixed-window factoring.  This
    # pins the measured drift so regressions in the step math itself (which
    # would diverge) are still caught.  Scaled-core agreement is the math gate.
    decoder = _maybe_decoder(3, 2, 64, 1, 32).eval()
    from ssm_decode.mamba3_cpu import CPUDecoder
    cpu = CPUDecoder.from_state_dict(decoder.state_dict())
    torch.manual_seed(11)
    x = torch.randn(2, 128, 3)
    diff = float((streaming_reference(decoder, x) - cpu.forward(x)).abs().max())
    assert diff < TOL_STEP_DEFAULT_INIT, f"default-init bf16 stepping noise {diff}"


def test_cpu_step_states_match_kernel_state_layout():
    decoder = _maybe_decoder(3, 2, 64, 1, 32).eval()
    states = initial_states(decoder, 2)
    assert len(states) == 1
    angle, ssm, k, v = states[0]
    assert angle.shape == (2, 2, 8) and ssm.shape == (2, 2, 64, 32)
    assert k.shape == (2, 2, 32) and v.shape == (2, 2, 64)
    x = torch.randn(2, 5, 3)
    _, new_states = decoder_step(decoder, x[:, 0], states)
    for before, after in zip(states[0], new_states[0]):
        assert before.dtype == after.dtype and before.device == after.device
        assert after.shape == before.shape


@pytest.mark.parametrize("chunk_lens", [
    [128] * 16,
    [256] * 8,
    [200, 512, 331, 1005],
], ids=["c128", "c256", "irregular"])
def test_gpu_chunked_carry_matches_oneshot(chunk_lens):
    decoder = _cuda_decoder()
    torch.manual_seed(5)
    x = torch.randn(2, 2048, 8, device="cuda")
    with torch.no_grad():
        oneshot, _ = forward_with_states(decoder, x, initial_states(decoder, 2, device="cuda"))
        carried = stream_recording(decoder, x, chunk_lens)
        module_default = decoder(x)
    boundary = chunk_lens[0]
    diff = float((carried[:, boundary:] - oneshot[:, boundary:]).abs().max())
    prefix = float((carried[:, :boundary] - oneshot[:, :boundary]).abs().max())
    vs_module = float((carried - module_default).abs().max())
    assert diff < TOL_CARRY, f"carry diff {diff} (prefix {prefix}, vs module {vs_module})"


def test_gpu_chunked_carry_angle_continuity():
    # One layer so the fp32 reference uses the exact core input (the root
    # projection); deeper blocks would need their own carried residual.
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the streaming carry gate")
    decoder = _maybe_decoder(8, 2, 128, 1, 128).cuda().eval()
    torch.manual_seed(6)
    x = torch.randn(2, 2048, 8, device="cuda")
    with torch.no_grad():
        # The 1e-5 angle gate belongs to the fp32-angle mode.  The default
        # (bf16, matching the training forward) keeps a looser gate.
        _, oneshot_final = forward_with_states(decoder, x, initial_states(decoder, 2, device="cuda"),
                                               fp32_angles=True)
        states = initial_states(decoder, 2, device="cuda")
        position = 0
        for length in (200, 512, 331, 1005):
            _, states = forward_with_states(decoder, x[:, position:position + length], states,
                                            fp32_angles=True)
            position += length
    # (a) chunked final angle equals the one-shot final angle per block
    for index, (chunked, oneshot) in enumerate(zip(states, oneshot_final)):
        diff = float((chunked[0] - oneshot[0]).abs().max())
        assert diff < 1e-5, f"block {index} final angle diff {diff}"
    # (a-bf16) the default carry (public-wrapper semantics) stays within the
    # documented bf16 angle quantization of about 2^-8 relative.
    with torch.no_grad():
        _, oneshot_bf16 = forward_with_states(decoder, x, initial_states(decoder, 2, device="cuda"))
        states_bf16 = initial_states(decoder, 2, device="cuda")
        position = 0
        for length in (200, 512, 331, 1005):
            _, states_bf16 = forward_with_states(decoder, x[:, position:position + length], states_bf16)
            position += length
    for index, (chunked, oneshot) in enumerate(zip(states_bf16, oneshot_bf16)):
        diff = float((chunked[0] - oneshot[0]).abs().max())
        assert diff < 0.05, f"block {index} bf16 final angle diff {diff}"
    # (b) both match the exact fp32 cumsum reference mod 2*pi
    from ssm_decode.mamba3_streaming import _core_prepare
    block = decoder.blocks[0]
    _, _, _, _, _, dt, _, angles = _core_prepare(block.ssm, block.norm1(decoder.in_proj(x)))
    increments = torch.tanh(angles.float()) * math.pi * dt.float().transpose(1, 2)[..., None]
    reference = torch.remainder(torch.cumsum(increments, dim=1)[:, -1], TWO_PI)
    for label, state in (("chunked", states[0]), ("oneshot", oneshot_final[0])):
        delta = torch.remainder(state[0] - reference + math.pi, TWO_PI) - math.pi
        diff = float(delta.abs().max())
        assert diff < 1e-3, f"block 0 {label} angle vs fp32 reference: {diff}"


def test_gpu_step_reference_matches_kernel():
    decoder = _scale_cores(_cuda_decoder())
    torch.manual_seed(7)
    x = torch.randn(2, 256, 8, device="cuda")
    with torch.no_grad():
        kernel = decoder(x)
        stepped = streaming_reference(decoder, x)
    diff = float((stepped - kernel).abs().max())
    assert diff < TOL_STEP_SCALED, f"step reference diff {diff}"


def test_gpu_mixed_kernel_chunks_and_step_bins():
    decoder = _scale_cores(_cuda_decoder())
    torch.manual_seed(9)
    x = torch.randn(2, 256, 8, device="cuda")
    with torch.no_grad():
        states = initial_states(decoder, 2, device="cuda")
        out, states = forward_with_states(decoder, x[:, :192], states)
        rows = [out]
        for position in range(192, 256):
            y, states = decoder_step(decoder, x[:, position], states)
            rows.append(y)
        mixed = torch.cat([out, torch.stack(rows[1:], dim=1)], dim=1)
        kernel = decoder(x)
    diff = float((mixed[:, 192:] - kernel[:, 192:]).abs().max())
    assert diff < TOL_STEP_SCALED, f"mixed carry/step diff {diff}"


def test_gpu_carry_backward_is_finite():
    decoder = _cuda_decoder()
    torch.manual_seed(12)
    x = torch.randn(2, 128, 8, device="cuda")
    states = initial_states(decoder, 2, device="cuda")
    y1, states = forward_with_states(decoder, x[:, :64], [tuple(s.detach().requires_grad_(True) for s in block) for block in states])
    y2, _ = forward_with_states(decoder, x[:, 64:], states)
    (y1.square().mean() + y2.square().mean()).backward()
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
