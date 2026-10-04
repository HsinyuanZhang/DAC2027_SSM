"""Sanity checks for the surprisingly small ternary-PTQ loss (eval-only).

1. Weight-level relative error of each quantizer on the RRAM groups.
2. Prediction-level change vs FP (relative RMS, correlation) on target all-valid bins.
3. Branch ablations: skip whole blocks, zero SSM branch, zero FFN branch.
4. Perturbation controls: random noise with the same per-row norm as the ternary error;
   random weights with the same per-row norm as the original ("destroy").
5. In-distribution check: source-validation R² (source normalizer), where cross-session
   shift cannot mask model error.
"""
from __future__ import annotations
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rram_ptq import rram_groups, quantize_block, ActQuant, CKPT, OUT  # noqa: E402
from ssm_decode import debug_experiment as d
from ssm_decode.data import source_target_plan, load_recording
from ssm_decode.mamba3_official import build_mamba3_official


def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    torch.set_num_threads(1); dev = torch.device('cuda:0'); torch.cuda.set_device(dev); results = {}
    for task in ['m1', 'm2']:
        plan = source_target_plan(task); t = load_recording(task, 'held_in', plan['cross_session_local_dev']['target_session'])
        ck_dir = CKPT.format(task=task); z = np.load(ck_dir + '/normalizer.npz')
        stats = tuple(z[k].astype('float32') for k in ('x_mean', 'x_std', 'y_mean', 'y_std'))
        u = [b for b in t.trial_bounds if b[1] - b[0] >= 50][:33]
        xs = np.concatenate([t.neural[s:e] for s, e in u]).astype(np.float64); tm, ts = xs.mean(0), xs.std(0); ts[ts < 1e-6] = 1.
        rec = replace(t, neural=((t.neural - tm) / ts * stats[1] + stats[0]).astype('float32'))
        src = [load_recording(task, 'held_in', s) for s in plan['source_held_in_sessions']]
        valid = d._windows(src, [d._trial_split(r)[1] for r in src], stats, dev)
        sel, _ = d._validation_selection(valid, 128, 1024)
        ck = torch.load(ck_dir + '/best.pt', map_location='cpu', weights_only=False); cfg = ck['args']; base = ck['state_dict']

        def fresh():
            m = build_mamba3_official(t.neural.shape[1], t.behavior.shape[1], width=cfg['width'], layers=cfg['layers'],
                                      state_size=cfg['state_size'], dropout=cfg['dropout']).to(dev)
            m.load_state_dict(base); return m.eval()

        def score(m):
            cache = d._target_predictions(m, rec, stats, dev, 128)
            zz = d._score_target(cache, stats, 'zero'); rr = d._score_target(cache, stats, 'ridge')
            return cache['all_valid_prediction'], {'target_zero': zz['all_valid_query_trial_r2_variance_weighted'],
                                                   'target_ridge': rr['all_valid_query_trial_r2_variance_weighted'],
                                                   'source_val': d._val(m, valid, 128, sel, 256, stats)}

        m = fresh(); p_fp, s_fp = score(m)
        def record(name, m, act=None):
            if act is not None: act.mode = 'quant'
            p, s = score(m)
            if act is not None: act.mode = 'off'
            s['pred_rel_rms_vs_fp'] = float(np.sqrt(((p - p_fp) ** 2).sum() / ((p_fp - p_fp.mean(0)) ** 2).sum()))
            s['pred_corr_vs_fp'] = float(np.corrcoef(p.ravel(), p_fp.ravel())[0, 1])
            results[f'{task}/{name}'] = s
            print(f"{task}/{name:28s} tgt={s['target_zero']:.4f} tgt+ridge={s['target_ridge']:.4f} "
                  f"srcval={s['source_val']:.4f} predRelRMS={s['pred_rel_rms_vs_fp']:.3f} corr={s['pred_corr_vs_fp']:.4f}", flush=True)
        results[f'{task}/fp'] = dict(s_fp, pred_rel_rms_vs_fp=0.0, pred_corr_vs_fp=1.0)
        print(f"{task}/{'fp':28s} tgt={s_fp['target_zero']:.4f} tgt+ridge={s_fp['target_ridge']:.4f} srcval={s_fp['source_val']:.4f}", flush=True)

        # 1. weight-level error
        m = fresh(); groups = rram_groups(m); werr = {}
        for bits in ('tern', 3, 4, 8):
            num = den = 0.
            for g, (mod, sl) in groups.items():
                w = mod.weight.data[sl].detach().cpu(); q = quantize_block(w, bits)
                num += float(((q - w) ** 2).sum()); den += float((w ** 2).sum())
            werr[str(bits)] = (num / den) ** .5
        results[f'{task}/weight_rel_error'] = werr; print(task, 'weight relative Frobenius error', {k: round(v, 3) for k, v in werr.items()}, flush=True)

        # quantized references (with INT8 activations, as in rram_ptq)
        def quantized(bits_fn):
            m = fresh(); groups = rram_groups(m); act = ActQuant([mod for mod, _ in groups.values()])
            act.mode = 'calibrate'
            sup = np.unique((d.split_support_query(rec, support_trials=33, query_start_trials=33, sequence=50)[0][:, None] + np.arange(50)).ravel())
            d._predict_endpoints(m, ((rec.neural - stats[0]) / stats[1]).astype('float32'), rec.trial_bounds, sup, dev, 128); act.mode = 'off'
            gen = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for g, (mod, sl) in groups.items():
                    w = mod.weight.data[sl].detach().cpu()
                    mod.weight.data[sl] = bits_fn(w, gen).to(mod.weight.device, mod.weight.dtype)
            return m, act
        for bits in (8, 4, 'tern'):
            m, act = quantized(lambda w, g, b=bits: quantize_block(w, b)); record(f'uniform_{bits}', m, act)
        # 4. perturbation controls
        def same_norm_noise(w, gen):
            e = quantize_block(w, 'tern') - w; r = torch.randn(w.shape, generator=gen)
            return w + r * (e.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True))
        m, act = quantized(same_norm_noise); record('random_noise_same_norm_as_tern', m, act)
        def destroy(w, gen):
            r = torch.randn(w.shape, generator=gen); return r * (w.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True))
        m, act = quantized(destroy); record('random_weights_same_row_norm', m, act)
        # 3. branch ablations
        m = fresh()
        for b in m.blocks: b.forward = (lambda x: x)
        record('skip_all_blocks', m)
        m = fresh()
        for b in m.blocks: b.ssm.forward = (lambda u, **kw: torch.zeros_like(u))
        record('zero_ssm_branch', m)
        m = fresh()
        for b in m.blocks: b.ffn.forward = (lambda x: torch.zeros_like(x))
        record('zero_ffn_branch', m)
    (OUT / 'sanity_checks.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
