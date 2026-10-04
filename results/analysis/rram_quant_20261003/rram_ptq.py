"""Eval-only 2T2R RRAM weight-quantization study for the Mamba-3 128x2 source checkpoints.

Post-training quantization only: no retraining, no target labels except final scoring.
RRAM-resident tensors (per block): ssm.in_proj z/x rows, ssm.in_proj B/C rows,
ssm.out_proj, ffn.0, ffn.2.  Everything else (root in/out projections, dt/A/trap/angle
rows, norms, biases, recurrent state) is treated as digital and left in floating point.

2T2R encoding: one differential pair stores one signed digit in {-1, 0, +1}.
  * 'tern'  : 1 pair/weight, TWN threshold (0.7*mean|w|) and per-output-row scale.
  * k bits  : symmetric levels [-(2^(k-1)-1), +(2^(k-1)-1)], per-row MSE-searched
              clipping, sign-magnitude bit slices -> (k-1) pairs/weight, shift-add in digital.
Device model (optional): LRS relative variation sigma on every 'on' cell, finite on/off
ratio; the systematic (1 - 1/ratio) gain is assumed absorbed by per-row digital scale.
Activations at RRAM-layer inputs can be fake-quantized to INT8 with a static per-tensor
scale calibrated on target support windows (neural data only).
ADC/partial-sum quantization is NOT modeled.
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from ssm_decode import debug_experiment as d
from ssm_decode.data import source_target_plan, load_recording
from ssm_decode.mamba3_official import build_mamba3_official

OUT = Path('results/analysis/rram_quant_20261003')
CKPT = 'results/debug_round2/latest/{task}/cross_session_mamba3_official_w128_l2_n16_ctx128_seed0'


def rram_groups(model):
    groups = {}
    for i, b in enumerate(model.blocks):
        core = b.ssm; inner = core.d_inner; bc = core.d_state * core.num_bc_heads * core.mimo_rank
        groups[f'L{i}.in_zx'] = (core.in_proj, slice(0, 2 * inner))
        groups[f'L{i}.in_BC'] = (core.in_proj, slice(2 * inner, 2 * inner + 2 * bc))
        groups[f'L{i}.out'] = (core.out_proj, slice(None))
        groups[f'L{i}.ffn1'] = (b.ffn[0], slice(None))
        groups[f'L{i}.ffn2'] = (b.ffn[2], slice(None))
    return groups


def pairs_per_weight(bits):
    return 1 if bits == 'tern' else int(bits) - 1


def _digits(q, n):
    """Sign-magnitude binary slices of integer tensor q -> list of n tensors in {-1,0,1}."""
    s, m = torch.sign(q), q.abs().long()
    return [s * ((m >> k) & 1).to(q.dtype) for k in range(n)]


def quantize_block(w, bits, sigma=0.0, ratio=20.0, gen=None):
    w = w.double()
    if bits == 'tern':
        delta = 0.7 * w.abs().mean(1, keepdim=True)
        mask = (w.abs() > delta).double()
        scale = (w.abs() * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        digits, weights = [torch.sign(w) * mask], [1.0]
    else:
        L = 2 ** (int(bits) - 1) - 1
        amax = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
        best, best_err = None, None
        for c in (1.0, .9, .8, .7, .6, .5, .4, .3):
            sc = amax * c / L
            err = ((torch.round(w / sc).clamp(-L, L) * sc - w) ** 2).sum(1, keepdim=True)
            if best is None:
                best, best_err = sc.clone(), err
            else:
                upd = err < best_err; best = torch.where(upd, sc, best); best_err = torch.where(upd, err, best_err)
        scale = best
        q = torch.round(w / scale).clamp(-L, L)
        n = int(bits) - 1
        digits, weights = _digits(q, n), [2.0 ** k for k in range(n)]
    if sigma > 0:
        out = torch.zeros_like(w)
        gain = 1 - 1 / ratio  # nominal differential gain, absorbed by per-row digital scale
        for dk, wk in zip(digits, weights):
            e1 = torch.randn(w.shape, generator=gen, dtype=w.dtype)
            e2 = torch.randn(w.shape, generator=gen, dtype=w.dtype)
            on = dk.abs()
            # d=+-1: one LRS and one HRS cell; d=0: two HRS cells (HRS = LRS/ratio).
            eff = (dk * ((1 + sigma * e1) - (1 + sigma * e2) / ratio)
                   + (1 - on) * (sigma / ratio) * (e1 - e2)) / gain
            out += wk * eff
        return (out * scale).float()
    return (sum(wk * dk for dk, wk in zip(digits, weights)) * scale).float()


class ActQuant:
    """Static per-tensor INT8 fake quantization at RRAM-layer inputs."""
    def __init__(self, modules):
        self.modules = list(dict.fromkeys(modules)); self.amax = {}; self.mode = 'off'; self.handles = []
        for m in self.modules:
            self.handles.append(m.register_forward_pre_hook(self._hook))
    def _hook(self, module, inputs):
        x = inputs[0]
        if self.mode == 'calibrate':
            self.amax[module] = max(self.amax.get(module, 0.0), float(x.detach().abs().amax()))
            return None
        if self.mode == 'quant':
            s = self.amax[module] / 127.0
            return (torch.round(x / s).clamp(-127, 127) * s,)
        return None


def evaluate(model, rec, stats, dev):
    cache = d._target_predictions(model, rec, stats, dev, 128)
    z = d._score_target(cache, stats, 'zero'); r = d._score_target(cache, stats, 'ridge')
    return {'zero_allvalid': z['all_valid_query_trial_r2_variance_weighted'],
            'ridge_allvalid': r['all_valid_query_trial_r2_variance_weighted'],
            'zero_legacy': z['r2_variance_weighted']}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['sensitivity', 'mixed'], required=True)
    p.add_argument('--tasks', nargs='+', default=['m1', 'm2']); a = p.parse_args()
    torch.set_num_threads(1); dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True); results = {}
    for task in a.tasks:
        plan = source_target_plan(task); t = load_recording(task, 'held_in', plan['cross_session_local_dev']['target_session'])
        ck_dir = CKPT.format(task=task); z = np.load(ck_dir + '/normalizer.npz')
        stats = tuple(z[k].astype('float32') for k in ('x_mean', 'x_std', 'y_mean', 'y_std'))
        u = [b for b in t.trial_bounds if b[1] - b[0] >= 50][:33]
        xs = np.concatenate([t.neural[s:e] for s, e in u]).astype(np.float64); tm, ts = xs.mean(0), xs.std(0); ts[ts < 1e-6] = 1.
        rec = replace(t, neural=((t.neural - tm) / ts * stats[1] + stats[0]).astype('float32'))  # support z-score (P0)
        ck = torch.load(ck_dir + '/best.pt', map_location='cpu', weights_only=False); cfg = ck['args']
        model = build_mamba3_official(t.neural.shape[1], t.behavior.shape[1], width=cfg['width'], layers=cfg['layers'],
                                      state_size=cfg['state_size'], dropout=cfg['dropout']).to(dev)
        base = {k: v.clone() for k, v in ck['state_dict'].items()}
        model.load_state_dict(base); model.eval()
        groups = rram_groups(model); names = list(groups)
        numel = {g: m.weight[sl].numel() for g, (m, sl) in groups.items()}
        aq = ActQuant([m for m, _ in groups.values()])
        # calibrate activation scales on support windows only (neural data, no labels)
        aq.mode = 'calibrate'
        sup = np.unique((d.split_support_query(rec, support_trials=33, query_start_trials=33, sequence=50)[0][:, None] + np.arange(50)).ravel())
        x_norm = ((rec.neural - stats[0]) / stats[1]).astype('float32')
        d._predict_endpoints(model, x_norm, rec.trial_bounds, sup, dev, 128); aq.mode = 'off'

        def run(name, alloc, act=True, sigma=0.0, seed=0):
            model.load_state_dict(base); gen = torch.Generator(device='cpu').manual_seed(seed)
            with torch.no_grad():
                for g, bits in alloc.items():
                    m, sl = groups[g]
                    if bits == 'fp':
                        continue
                    w = m.weight.data[sl].detach().cpu()
                    m.weight.data[sl] = quantize_block(w, bits, sigma=sigma, gen=gen).to(m.weight.device, m.weight.dtype)
            aq.mode = 'quant' if act else 'off'
            started = time.time(); res = evaluate(model, rec, stats, dev); aq.mode = 'off'
            pairs = sum(numel[g] * pairs_per_weight(b) for g, b in alloc.items() if b != 'fp')
            res.update(alloc={g: str(b) for g, b in alloc.items()}, act_int8=act, sigma=sigma, seed=seed,
                       rram_2t2r_pairs=pairs, rram_weights=sum(numel.values()), seconds=time.time() - started)
            key = f'{task}/{name}'; results[key] = res
            print(f"{key:40s} zero={res['zero_allvalid']:.4f} ridge={res['ridge_allvalid']:.4f} "
                  f"pairs={pairs/2**20:.3f}Mi ({time.time()-started:.0f}s)", flush=True)

        allbits = lambda b: {g: b for g in names}
        if a.phase == 'sensitivity':
            print(f'{task}: RRAM weights per group', numel, 'total', sum(numel.values()), flush=True)
            run('fp', allbits('fp'), act=False)
            run('fp_actint8', allbits('fp'), act=True)
            for b in (8, 6, 4, 3, 'tern'):
                run(f'uniform_{b}', allbits(b))
            for g in names:
                alloc = allbits(8); alloc[g] = 'tern'; run(f'tern_only_{g}', alloc)
            for g in names:
                alloc = allbits('tern'); alloc[g] = 8; run(f'all_tern_except_{g}_8b', alloc)
        else:
            spec = json.loads(Path(OUT / 'mixed_configs.json').read_text())
            for name, rule in spec.items():
                alloc = {g: next((b for pat, b in rule.items() if pat in g), rule.get('default', 8)) for g in names}
                alloc = {g: (b if b in ('tern', 'fp') else int(b)) for g, b in alloc.items()}
                run(name, alloc)
                for sigma in (0.05, 0.10):
                    for seed in (0, 1, 2):
                        run(f'{name}_sigma{sigma}_chip{seed}', alloc, sigma=sigma, seed=seed)
    path = OUT / f'{a.phase}.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(results); path.write_text(json.dumps(old, indent=2) + '\n')


if __name__ == '__main__':
    main()
