"""Eval-only check: does the FA mean offset (fit centered, decode uncentered) explain the
Track A FA-method failure on M2?  Also decodes the whole recording in one pass (no 8192 resets).
Uses the existing post-fix AP checkpoints. No training."""
import json, sys
from pathlib import Path
import numpy as np, torch
from ssm_decode import association as assoc
from ssm_decode import track_a_experiment as ta

ROOT = Path('results/ap_pretraining_m2_20261004')
out = {}
dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
for target in sys.argv[1:]:
    model, blob = ta.load_frozen_ap_backbone(str(ROOT / target / 'best.pt'), device=dev)
    for p in model.parameters(): p.requires_grad_(False)
    k = model.k
    stats = tuple(np.asarray(s, dtype=np.float64) for s in blob['source_statistics'][0][2:4])
    rec = ta.cached_recording('m2', 'held_in', target, root=Path(ta.DEFAULT_ROOT), cache_root=Path(ta.DEFAULT_CACHE))
    trials = ta.usable_trials(rec); b_src = model.head.weight.detach().cpu().numpy().astype(np.float64)
    mean_table = np.mean([t.detach().cpu().numpy().astype(np.float64) for t in model.tables], axis=0)
    def decode(xz, table):  # one contiguous pass
        with torch.inference_mode():
            z = torch.from_numpy(xz.astype(np.float32)).to(dev).unsqueeze(0) @ torch.from_numpy(table.astype(np.float32)).to(dev).T
            return model.base(z).squeeze(0).cpu().numpy()
    def r2(pred, bins):
        p = pred[bins] * stats[1] + stats[0]; y = rec.behavior[bins].astype(np.float64)
        return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    for budget in (8, 33):
        mean, std = ta.support_stats(rec, budget, trials)
        xz = (rec.neural.astype(np.float64) - mean) / std
        l_bins = ta.labeled_bins(rec, trials, budget); q_bins = ta.query_bins(rec, trials, budget)
        y_l = (rec.behavior[l_bins].astype(np.float64) - stats[0]) / stats[1]
        y_q = (rec.behavior[q_bins].astype(np.float64) - stats[0]) / stats[1]
        res = {'mean_table': r2(decode(xz, mean_table), q_bins)}
        for name, (xl, yl) in {'am1': (xz[l_bins], y_l), 'oracle_frame': (xz[q_bins], y_q)}.items():
            a = assoc.am1_table(xl, xz, yl, b_src, k=k, lags=2, lam=1.0, head_lam=1.0, em_iters=300, seed=0)
            m = a['fa']['mean']; beta = a['beta']
            zl = (xz - m) @ beta.T
            res[f'{name}_uncentered(as_repo)'] = r2(decode(xz, a['table']), q_bins)
            res[f'{name}_centered'] = r2(decode(xz - m, a['table']), q_bins)
            res[f'{name}_latent_offset_over_std'] = float(np.linalg.norm(beta @ m) / np.sqrt((zl.var(0)).sum()))
        out[f'{target}/b{budget}'] = res
        print(target, budget, {k2: round(v, 3) for k2, v in res.items()}, flush=True)
Path('results/analysis/analysis_20261005/fa_centering_check.json').write_text(json.dumps(out, indent=2) + '\n')
