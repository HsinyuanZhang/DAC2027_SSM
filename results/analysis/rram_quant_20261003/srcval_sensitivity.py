"""Per-group / mixed 2T2R quantization measured in-distribution (source validation R²) and by
prediction fidelity vs FP on source-validation endpoints. Eval-only, INT8 activations at RRAM inputs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rram_ptq import rram_groups, quantize_block, ActQuant, CKPT, OUT, pairs_per_weight  # noqa: E402
from ssm_decode import debug_experiment as d
from ssm_decode.data import source_target_plan, load_recording
from ssm_decode.mamba3_official import build_mamba3_official

def main():
    torch.set_num_threads(1); dev = torch.device('cuda:0'); torch.cuda.set_device(dev); res = {}
    for task in ['m1', 'm2']:
        plan = source_target_plan(task); src = [load_recording(task, 'held_in', s) for s in plan['source_held_in_sessions']]
        ck_dir = CKPT.format(task=task); z = np.load(ck_dir + '/normalizer.npz')
        stats = tuple(z[k].astype('float32') for k in ('x_mean', 'x_std', 'y_mean', 'y_std'))
        tr = d._windows(src, [d._trial_split(r)[0] for r in src], stats, dev)
        va = d._windows(src, [d._trial_split(r)[1] for r in src], stats, dev); sel, _ = d._validation_selection(va, 128, 1024)
        calib, _ = d._validation_selection(tr, 128, 2048)  # activation scales from source TRAIN endpoints only
        ck = torch.load(ck_dir + '/best.pt', map_location='cpu', weights_only=False); cfg = ck['args']
        m = build_mamba3_official(src[0].neural.shape[1], src[0].behavior.shape[1], width=cfg['width'], layers=cfg['layers'],
                                  state_size=cfg['state_size'], dropout=cfg['dropout']).to(dev)
        m.load_state_dict(ck['state_dict']); m.eval(); groups = rram_groups(m); names = list(groups)
        numel = {g: mod.weight[sl].numel() for g, (mod, sl) in groups.items()}
        act = ActQuant([mod for mod, _ in groups.values()])
        def preds(selection):
            out = []
            with torch.inference_mode():
                for lo in range(0, len(selection), 256):
                    xx, _, _ = d._right_aligned(va, selection[lo:lo + 256], 128); out.append(m(xx)[:, -1].float().cpu())
            return torch.cat(out).numpy()
        act.mode = 'calibrate'
        with torch.inference_mode():
            for lo in range(0, len(calib), 256):
                xx, _, _ = d._right_aligned(tr, calib[lo:lo + 256], 128); m(xx)
        act.mode = 'off'; p_fp = preds(sel); fp = d._val(m, va, 128, sel, 256, stats)
        def run(name, alloc):
            m.load_state_dict(ck['state_dict'])
            with torch.no_grad():
                for g, b in alloc.items():
                    if b == 'fp': continue
                    mod, sl = groups[g]; w = mod.weight.data[sl].detach().cpu()
                    mod.weight.data[sl] = quantize_block(w, b).to(mod.weight.device, mod.weight.dtype)
            act.mode = 'quant'; v = d._val(m, va, 128, sel, 256, stats); p = preds(sel); act.mode = 'off'
            rel = float(np.sqrt(((p - p_fp) ** 2).sum() / ((p_fp - p_fp.mean(0)) ** 2).sum()))
            pairs = sum(numel[g] * pairs_per_weight(b) for g, b in alloc.items() if b != 'fp')
            res[f'{task}/{name}'] = {'source_val_r2': v, 'delta_vs_fp': v - fp, 'pred_rel_rms_vs_fp': rel, 'rram_2t2r_pairs': pairs}
            print(f'{task}/{name:30s} srcval={v:.4f} (Δ{v-fp:+.4f}) predRelRMS={rel:.3f} pairs={pairs/2**20:.2f}Mi', flush=True)
        res[f'{task}/fp'] = {'source_val_r2': fp}; print(f'{task}/fp srcval={fp:.4f}', flush=True)
        run('act_int8_only', {g: 'fp' for g in names})
        for b in (8, 4, 3, 'tern'): run(f'uniform_{b}', {g: b for g in names})
        for g in names:
            alloc = {k: 8 for k in names}; alloc[g] = 'tern'; run(f'tern_only_{g}', alloc)
        mixed = {'in_zx_4b_rest_tern': lambda g: 4 if 'in_zx' in g else 'tern',
                 'in_zx_8b_rest_tern': lambda g: 8 if 'in_zx' in g else 'tern',
                 'in_zx+out_4b_rest_tern': lambda g: 4 if ('in_zx' in g or g.endswith('.out')) else 'tern',
                 'ffn+BC_tern_rest_4b': lambda g: 'tern' if ('ffn' in g or 'in_BC' in g) else 4,
                 'ffn+BC_tern_rest_8b': lambda g: 'tern' if ('ffn' in g or 'in_BC' in g) else 8}
        for name, fn in mixed.items(): run(name, {g: fn(g) for g in names})
    (OUT / 'srcval_sensitivity.json').write_text(json.dumps(res, indent=2) + '\n')

if __name__ == '__main__':
    main()
