"""C2 streaming equivalence measurements (roadmap item C2, 2026-10-04).

Reproduces every number quoted in README.md.  Run on a free GPU:

  cd <repo root>
  PYTHONNOUSERSITE=1 PYTHONPATH=.tools/mamba_deps:. TRITON_CACHE_DIR=/tmp/triton_c2 \
  CUDA_VISIBLE_DEVICES=1 results/analysis/c2_streaming_20261004/measure.py
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ssm_decode.mamba3_official import build_mamba3_official  # noqa: E402
from ssm_decode.mamba3_streaming import (  # noqa: E402
    _core_prepare,
    decoder_step,
    forward_with_states,
    initial_states,
    stream_recording,
    streaming_reference,
)

TWO_PI = 2.0 * math.pi
REPORT = {}


def scale_cores(decoder, factor=0.02):
    with torch.no_grad():
        for block in decoder.blocks:
            block.ssm.in_proj.weight.mul_(factor)
    return decoder


def carry_experiment(device="cuda"):
    decoder = build_mamba3_official(8, 2, 128, 2, 128).to(device).eval()
    torch.manual_seed(5)
    x = torch.randn(2, 2048, 8, device=device)
    out = {}
    with torch.no_grad():
        module_default = decoder(x)
        for fp32 in (True, False):
            mode = "fp32_angle_carry" if fp32 else "bf16_angle_carry"
            oneshot, _ = forward_with_states(decoder, x, initial_states(decoder, 2, device=device), fp32)
            out[mode] = {"oneshot_vs_module_default": float((oneshot - module_default).abs().max())}
            for name, lens in (("c128", [128] * 16), ("c256", [256] * 8),
                               ("irregular", [200, 512, 331, 1005])):
                carried = stream_recording(decoder, x, lens, fp32)
                out[mode][name] = {
                    "chunked_vs_oneshot_after_first_boundary": float((carried[:, lens[0]:] - oneshot[:, lens[0]:]).abs().max()),
                    "chunked_vs_module_default": float((carried - module_default).abs().max()),
                }
    # noise floor: the module's own entry vs the public wrapper on identical inputs
    REPORT["carry"] = out


def angle_experiment(device="cuda"):
    decoder = build_mamba3_official(8, 2, 128, 1, 128).to(device).eval()
    torch.manual_seed(6)
    x = torch.randn(2, 2048, 8, device=device)
    with torch.no_grad():
        _, oneshot_final = forward_with_states(decoder, x, initial_states(decoder, 2, device=device))
        states = initial_states(decoder, 2, device=device)
        position = 0
        for length in (200, 512, 331, 1005):
            _, states = forward_with_states(decoder, x[:, position:position + length], states)
            position += length
        block = decoder.blocks[0]
        _, _, _, _, _, dt, _, angles = _core_prepare(block.ssm, block.norm1(decoder.in_proj(x)))
        increments = torch.tanh(angles.float()) * math.pi * dt.float().transpose(1, 2)[..., None]
        reference = torch.remainder(torch.cumsum(increments, dim=1)[:, -1], TWO_PI)
        wrapped = lambda state: float((torch.remainder(state[0] - reference + math.pi, TWO_PI) - math.pi).abs().max())
        REPORT["angle"] = {
            "chunked_vs_oneshot_final_angle": float((states[0][0] - oneshot_final[0][0]).abs().max()),
            "chunked_vs_fp32_reference": wrapped(states[0]),
            "oneshot_vs_fp32_reference": wrapped(oneshot_final[0]),
        }


def step_experiment(device="cuda"):
    torch.manual_seed(7)
    x = torch.randn(2, 256, 8, device=device)
    out = {}
    for tag, scale in (("scaled_core_0.02", True), ("default_init", False)):
        decoder = build_mamba3_official(8, 2, 128, 2, 128).to(device).eval()
        if scale:
            scale_cores(decoder)
        with torch.no_grad():
            kernel = decoder(x)
            stepped = streaming_reference(decoder, x)
            states = initial_states(decoder, 2, device=device)
            first, states = forward_with_states(decoder, x[:, :192], states)
            rows = []
            for position in range(192, 256):
                y, states = decoder_step(decoder, x[:, position], states)
                rows.append(y)
            mixed = torch.cat([first, torch.stack(rows, dim=1)], dim=1)
        out[tag] = {
            "step_vs_kernel_T256": float((stepped - kernel).abs().max()),
            "kernel_output_abs_max": float(kernel.abs().max()),
            "mixed_chunk192_plus_64_steps_vs_kernel": float((mixed[:, 192:] - kernel[:, 192:]).abs().max()),
        }
    REPORT["step_gpu"] = out


def cpu_experiment():
    torch.manual_seed(11)
    decoder = build_mamba3_official(3, 2, 64, 1, 32).eval()
    x = torch.randn(2, 128, 3)
    from ssm_decode.mamba3_cpu import CPUDecoder
    unscaled = float((streaming_reference(decoder, x) - CPUDecoder.from_state_dict(decoder.state_dict()).forward(x)).abs().max())
    scale_cores(decoder)
    cpu = CPUDecoder.from_state_dict(decoder.state_dict())
    scaled = {f"T{length}": float((streaming_reference(decoder, x[:, :length]) - cpu.forward(x[:, :length])).abs().max())
              for length in (1, 7, 64, 100, 128)}
    REPORT["step_cpu"] = {"default_init_T128": unscaled, "scaled_core_0.02": scaled}


def backward_experiment(device="cuda"):
    decoder = build_mamba3_official(8, 2, 128, 2, 128).to(device).eval()
    torch.manual_seed(12)
    x = torch.randn(2, 128, 8, device=device)
    roots = [tuple(s.detach().requires_grad_(True) for s in block)
             for block in initial_states(decoder, 2, device=device)]
    y1, states = forward_with_states(decoder, x[:, :64], roots)
    y2, _ = forward_with_states(decoder, x[:, 64:], states)
    (y1.square().mean() + y2.square().mean()).backward()
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    state_grads = [t.grad for block in roots for t in block if t.grad is not None]
    REPORT["carry_backward"] = {
        "parameter_grad_tensors": len(grads),
        "all_parameter_grads_finite": bool(all(torch.isfinite(g).all() for g in grads)),
        "carried_state_grads_returned": len(state_grads),
        "state_grad_abs_max": float(max(g.abs().max() for g in state_grads)) if state_grads else None,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carry_experiment(device)
    angle_experiment(device)
    step_experiment(device)
    cpu_experiment()
    backward_experiment(device)
    REPORT["schema"] = "c2_streaming_equivalence_v1"
    REPORT["device"] = device
    REPORT["torch"] = torch.__version__
    target = Path(__file__).resolve().parent / "equivalence.json"
    target.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(json.dumps(REPORT, indent=2))
