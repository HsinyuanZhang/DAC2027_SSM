"""H4 live-residual tests. L0 (rebased_feedback_loop) is not modified."""
import torch
from torch import nn

from ssm_decode import stream_f7 as sf
from ssm_decode.live_residual import (
    calibration_pieces,
    fir_increment,
    live_rebased,
    roll_ring,
)


class _Id(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Identity()

    def forward(self, z):
        return self.out_proj(z)


def _memoryless(_z, state):
    return _z, state


def _stateful(z, state):
    """y_t = z_t + s, s <- 0.5 s + z_t. A reset at a trial start is observable."""
    if state is None:
        state = torch.zeros(z.shape[0], z.shape[-1], dtype=z.dtype, device=z.device)
    outs = []
    s = state
    for t in range(z.shape[1]):
        y = z[:, t, :] + s
        s = 0.5 * s + z[:, t, :]
        outs.append(y)
    return torch.stack(outs, 1), s


def _prepare_factory(x, spans, prior):
    """Prefix-only normalisation. Does not read a trial it has not been asked for."""

    def prepare(k, closed):
        nprefix = 1 if (not closed and k == 0) else (k + 1 if closed else k)
        parts = [x[a:b] for a, b in spans[:nprefix]]
        chunk = torch.cat(parts, 0)
        mean = chunk.mean(0)
        std = torch.ones_like(mean)
        table = torch.stack([mean, 0.25 * mean], 0)
        if not closed and k == 0:
            return mean, std, prior
        return mean, std, table

    return prepare


def _pack(n=3, d=2, k=2, gaps=True, b=4, cold=5):
    torch.manual_seed(0)
    spans = []
    cursor = cold
    length = 6
    for i in range(b):
        spans.append((cursor, cursor + length))
        cursor += length
        if gaps and i + 1 < b:
            cursor += 3
    t = cursor
    x = torch.randn(t, n)
    y = torch.randn(t, d)
    valid = torch.ones(t)
    prior = torch.randn(k, n)
    jbar = torch.randn(4, d, k) * 0.1
    return x, y, valid, spans, prior, jbar


def test_pieces_cover_cold_trials_and_gaps_and_stop_at_the_last_trial():
    spans = [(5, 9), (12, 15), (15, 18)]
    pieces = calibration_pieces(spans, 2, 40)
    assert pieces == [(0, 5, "cold", None), (5, 9, "trial", 0), (9, 12, "gap", 0), (12, 15, "trial", 1)]
    assert calibration_pieces([(0, 4), (4, 7)], 2, 7)[0] == (0, 4, "trial", 0)


def test_fir_increment_matches_zero_pad_and_a_carried_ring():
    torch.manual_seed(1)
    xz = torch.randn(7, 3)
    r = torch.randn(7, 2)
    jbar = torch.randn(4, 2, 5)
    ring0 = xz.new_zeros(3, 3)
    u, new_ring = fir_increment(xz, r, jbar, ring0)
    assert torch.allclose(u, sf.fir_data_term(xz, r, jbar), atol=1e-6)
    assert torch.allclose(new_ring, xz[-3:])
    hist = torch.randn(3, 3)
    u_h, _ = fir_increment(xz, r, jbar, hist)
    cat_x = torch.cat([hist, xz], 0)
    cat_r = torch.cat([torch.zeros(3, 2), r], 0)
    assert torch.allclose(u_h, sf.fir_data_term(cat_x, cat_r, jbar), atol=1e-6)
    # A zero residual adds nothing, but the ring still advances.
    u0, rolled = fir_increment(xz, torch.zeros_like(r), jbar, hist)
    assert torch.allclose(u0, torch.zeros_like(u0))
    assert torch.allclose(rolled, roll_ring(xz, hist))


def test_l2_memoryless_matches_l0_with_cold_prefix_and_gaps():
    x, y, valid, spans, prior, jbar = _pack()
    prepare = _prepare_factory(x, spans, prior)
    base = _Id()
    raw = [(x[a:b], y[a:b], valid[a:b]) for a, b in spans]
    l0 = sf.rebased_feedback_loop(base, raw, prior, 0.4, 0.1, prepare, mode="rebase",
                                  kind="fir", jbar=jbar)
    live = live_rebased(base, x, y, valid, spans, prior, 0.4, 0.1, prepare, jbar, len(spans),
                        ring_mode="reset", forward_fn=_memoryless)
    assert torch.allclose(live["table"], l0["table"], atol=1e-5)
    assert torch.allclose(live["bias"], l0["bias"], atol=1e-5)
    assert torch.allclose(live["delta"], l0["delta"], atol=1e-5)


def test_l1_ring_sees_the_gap_and_l2_does_not():
    x, y, valid, spans, prior, jbar = _pack(gaps=True)
    # Nonzero lag so a pre-trial input can enter U. Zero the within-trial part of the
    # comparison by using the same labels; only the ring differs.
    prepare = _prepare_factory(x, spans, prior)
    base = _Id()
    l1 = live_rebased(base, x, y, valid, spans, prior, 1.0, 0.0, prepare, jbar, len(spans),
                      ring_mode="carry", forward_fn=_memoryless)
    l2 = live_rebased(base, x, y, valid, spans, prior, 1.0, 0.0, prepare, jbar, len(spans),
                      ring_mode="reset", forward_fn=_memoryless)
    assert not torch.allclose(l1["delta"], l2["delta"], atol=1e-6)

    x_gap = x.clone()
    # Gap between trial 0 and trial 1.
    g0, g1 = spans[0][1], spans[1][0]
    assert g1 > g0
    x_gap[g0:g1] += 3.0
    y_gap = y.clone()
    y_gap[g0:g1] += 5.0
    l1_x = live_rebased(base, x_gap, y, valid, spans, prior, 1.0, 0.0, prepare, jbar, len(spans),
                        ring_mode="carry", forward_fn=_memoryless)
    l1_y = live_rebased(base, x, y_gap, valid, spans, prior, 1.0, 0.0, prepare, jbar, len(spans),
                        ring_mode="carry", forward_fn=_memoryless)
    l2_x = live_rebased(base, x_gap, y, valid, spans, prior, 1.0, 0.0, prepare, jbar, len(spans),
                        ring_mode="reset", forward_fn=_memoryless)
    # Gap labels are not residuals. Gap inputs move L1 and not L2.
    assert torch.allclose(l1_y["table"], l1["table"], atol=1e-6)
    assert not torch.allclose(l1_x["table"], l1["table"], atol=1e-5)
    assert torch.allclose(l2_x["table"], l2["table"], atol=1e-6)


def test_eta0_rho0_is_prefix_p0_exactly():
    x, y, valid, spans, prior, jbar = _pack(b=3)
    prepare = _prepare_factory(x, spans, prior)
    ans = live_rebased(_Id(), x, y, valid, spans, prior, 0.0, 0.0, prepare, jbar, len(spans),
                       ring_mode="carry", forward_fn=_stateful)
    _mean, _std, p0 = prepare(len(spans) - 1, True)
    assert torch.allclose(ans["table"], p0, atol=1e-6)
    assert torch.allclose(ans["delta"], torch.zeros_like(ans["delta"]), atol=1e-6)
    assert torch.allclose(ans["bias"], torch.zeros_like(ans["bias"]), atol=1e-6)
    # Every commit is that prefix's P0, which is what the contiguous P0 decode installs.
    for item in ans["history"]:
        _m, _s, p0_k = prepare(item["k"], True)
        assert torch.allclose(item["table"], p0_k, atol=1e-6)


def test_state_is_carried_across_trial_starts():
    x, y, valid, spans, prior, jbar = _pack(b=3, cold=8)
    prepare = _prepare_factory(x, spans, prior)
    ans = live_rebased(_Id(), x, y, valid, spans, prior, 0.0, 0.0, prepare, jbar, len(spans),
                       ring_mode="carry", forward_fn=_stateful, keep_yhat=True)
    assert ans["yhat"].shape[0] == spans[-1][1]
    a, b = spans[1]
    mean, std, p0 = prepare(1, False)
    xz = (x[a:b] - mean) / std
    z = (xz @ p0.T).unsqueeze(0)
    y_reset, _ = _stateful(z, None)
    # Fails if the live forward restarted the trunk at this trial.
    assert not torch.allclose(ans["yhat"][a:b], y_reset[0], atol=1e-5)
    assert (ans["yhat"][a:b] - y_reset[0]).abs().max() > 1e-3


def test_l1_causality_at_b8_with_positive_eta():
    x, y, valid, spans, prior, jbar = _pack(b=8, cold=4, gaps=True)
    prepare = _prepare_factory(x, spans, prior)
    kwargs = dict(ring_mode="carry", forward_fn=_stateful, keep_yhat=True)
    base = live_rebased(_Id(), x, y, valid, spans, prior, 1.5, 0.2, prepare, jbar, 8, **kwargs)
    t = spans[4][0]
    assert t >= 8
    xp, yp = x.clone(), y.clone()
    xp[t:] += torch.randn_like(xp[t:])
    yp[t:] += torch.randn_like(yp[t:])
    # prepare for prefixes that end before t must not see the perturbation.
    prepare_p = _prepare_factory(xp, spans, prior)
    pert = live_rebased(_Id(), xp, yp, valid, spans, prior, 1.5, 0.2, prepare_p, jbar, 8, **kwargs)
    assert torch.allclose(base["yhat"][:t], pert["yhat"][:t], atol=1e-6)
    for left, right in zip(base["history"][:4], pert["history"][:4]):
        assert left["k"] == right["k"]
        assert spans[left["k"]][1] <= t
        assert torch.allclose(left["table"], right["table"], atol=1e-6)
        assert torch.allclose(left["delta"], right["delta"], atol=1e-6)
    # Positive control: the same perturbation moves a later commit.
    assert not torch.allclose(base["history"][-1]["table"], pert["history"][-1]["table"], atol=1e-5)


def test_gpu_mamba_eta0_matches_oneshot_and_state_is_carried():
    """Real carry path: eta=0 table is P0, yhat matches a one-shot forward, state is not reset."""
    if not torch.cuda.is_available():
        return
    from ssm_decode.mamba3_official import build_mamba3_official
    from ssm_decode.mamba3_streaming import forward_with_states

    torch.manual_seed(2)
    k, d, n = 8, 2, 4
    decoder = build_mamba3_official(k, d, width=64, layers=1, state_size=16).cuda().eval()
    spans = [(10, 50), (60, 100), (100, 140)]
    x = torch.randn(140, n, device="cuda")
    y = torch.randn(140, d, device="cuda")
    valid = torch.ones(140, device="cuda")
    prior = torch.randn(k, n, device="cuda")
    jbar = torch.randn(5, d, k, device="cuda") * 0.05
    prepare = _prepare_factory(x, spans, prior)
    # _prepare_factory builds a K=2 table. Rebuild a K-wide prefix table.
    def prepare_k(k_idx, closed):
        mean, std, _ = prepare(k_idx, closed)
        nprefix = 1 if (not closed and k_idx == 0) else (k_idx + 1 if closed else k_idx)
        chunk = torch.cat([x[a:b] for a, b in spans[:nprefix]], 0)
        mean = chunk.mean(0)
        std = torch.ones_like(mean)
        base = mean.unsqueeze(0).expand(k, -1).contiguous()
        scale = torch.linspace(0.2, 1.0, k, device=x.device).unsqueeze(1)
        table = base * scale
        return mean, std, table

    ans = live_rebased(decoder, x, y, valid, spans, prior, 0.0, 0.0, prepare_k, jbar, 3,
                       ring_mode="carry", keep_yhat=True)
    _m, _s, p0 = prepare_k(2, True)
    assert torch.allclose(ans["table"], p0, atol=1e-5)
    # One-shot z under the P0 schedule (cold prior, then each commit).
    pieces = []
    cold = x[:spans[0][0]] @ prior.T
    pieces.append(cold)
    for idx, (a, b) in enumerate(spans):
        mean, std, table = prepare_k(idx, False) if idx else prepare_k(0, False)
        if idx == 0:
            table = prior
            mean, std, _ = prepare_k(0, False)
        else:
            mean, std, table = prepare_k(idx, False)
        xz = (x[a:b] - mean) / std
        pieces.append(xz @ table.T)
        if idx + 1 < len(spans) and spans[idx + 1][0] > b:
            gap = (x[b:spans[idx + 1][0]] - mean) / std
            pieces.append(gap @ table.T)
    z = torch.cat(pieces, 0).unsqueeze(0)
    with torch.no_grad():
        oneshot, _ = forward_with_states(decoder, z, None)
    assert oneshot.shape[1] == ans["yhat"].shape[0]
    err = float((oneshot[0] - ans["yhat"]).abs().max())
    assert err < 2e-2, err
    a, b = spans[1]
    mean, std, table = prepare_k(1, False)
    z1 = ((x[a:b] - mean) / std @ table.T).unsqueeze(0)
    with torch.no_grad():
        reset, _ = forward_with_states(decoder, z1, None)
    assert float((ans["yhat"][a:b] - reset[0]).abs().max()) > 1e-2
