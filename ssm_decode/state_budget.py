"""C10 recurrent-state budget accounting (revision 4.4.1).

The state layout follows the verified Mamba-3 SISO carry contract in
``ssm_decode/mamba3_streaming.py``.  Per layer and head:

- angle cumsum: ``d_state / 4`` values (``num_rope_angles``),
- SSM state: ``headdim * d_state`` = ``64 * d_state`` values,
- K carry: ``d_state`` values,
- V carry: ``headdim`` = 64 values.

The decoder wraps ``layers`` such cores, ``nheads = 2 * width / 64`` heads per
core (the official module uses ``expand = 2``), and the budget assumes INT24
state storage at 3 bytes per value.

Anchor check (tests/test_state_budget.py): width 256, 4 layers, d_state 32
gives 201.9 KB, matching the roadmap figure of approximately 202 KB.
"""
from __future__ import annotations

INT24_BYTES = 3
HEADDIM = 64
BUDGET_BYTES = 128 * 1024


def state_values(d_state: int) -> int:
    """Carried values per layer per head at ``d_state``."""
    if d_state < 4:
        raise ValueError("d_state must be a multiple of 4 >= 4")
    return HEADDIM * d_state + d_state // 4 + d_state + HEADDIM


def state_bytes(width: int, layers: int, d_state: int) -> int:
    """Total recurrent-state bytes of one Mamba-3 decoder."""
    nheads = (2 * width) // HEADDIM
    return state_values(d_state) * nheads * layers * INT24_BYTES


def state_report(width: int, layers: int, d_state: int, *, ffn_mult: int = 2) -> dict:
    total = state_bytes(width, layers, d_state)
    return {"width": width, "layers": layers, "d_state": d_state, "ffn_mult": ffn_mult,
            "state_kb": round(total / 1024, 1),
            "budget_kb": round(BUDGET_BYTES / 1024, 1),
            "within_budget": total <= BUDGET_BYTES,
            "nheads_per_layer": (2 * width) // HEADDIM}


# C10 candidate matrix: capacity moves into stateless parts, state stays small.
C10_VARIANTS = {
    "small_ref": {"width": 128, "layers": 2, "d_state": 32, "ffn_mult": 2},   # 50.4 KB
    "wide_ref": {"width": 256, "layers": 4, "d_state": 32, "ffn_mult": 2},    # 201.9 KB, over budget
    "c10_wide_n8": {"width": 256, "layers": 4, "d_state": 8, "ffn_mult": 2},  # 54.9 KB
    "c10_w2_l2_n16": {"width": 256, "layers": 2, "d_state": 16, "ffn_mult": 2},  # 51.9 KB
    "c10_deep_n8": {"width": 128, "layers": 4, "d_state": 8, "ffn_mult": 2},  # 27.5 KB
    "c10_ffn4": {"width": 128, "layers": 2, "d_state": 32, "ffn_mult": 4},    # 50.4 KB, wider FFN
}
