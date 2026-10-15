"""AP-style backbone training and target-side rows for the NinaPro DB6 protocol D-A (v2: contiguous streaming).

Thin wrapper around ``APBackbone`` (per-session K x 14 read-in tables + shared
Mamba-3 trunk + linear head) for per-bin classification of 8 classes
(rest + 7 grasps) with cross-entropy.

Streaming contract (v2)
-----------------------
* A session is ONE continuous stream of 50 Hz bins.  The model, the front-end
  integrator and the ridge EMA are run once, causally, over the whole stream and their
  state is carried across trial boundaries.  Trial boundaries (derived from the labels)
  are used only to choose which bins are scored / trained on, never to reset state.
  (v1 reset the state at every trial start; the reset position leaks the label timing
  because each trial ends with its movement, and it is not implementable on a stream.)
* Training windows are cut from the continuous session stream at random offsets
  (``WIN_MIN``..``WIN_MAX`` bins, never aligned to trial boundaries); the first ``BURN``
  bins of a window provide state context only and are excluded from the loss.
* Source split: repetitions in ``SOURCE_VAL_REPS`` of every grasp are held out.  Their
  labels are never used in training (loss mask); their inputs appear only as unlabeled
  context inside the continuous windows.  Source validation decodes the whole session
  stream in one pass and scores the held-out trials' bins.
* Target: the B calibration trials supply z-score statistics, labels and fitting data.
  Table / ridge fitting never uses neural data outside the B trials: maximal runs of
  adjacent calibration trials are fitted as contiguous segments from zero state.  The
  query is scored on the bins of the non-calibration trials of the one-pass stream decode;
  calibration bins are decoded (state context) but never scored.
* Input z-score: per-channel mean/std over the bins of the given trials; the std is floored
  at ``floor * median std over channels`` (near-dead channels would otherwise be amplified to
  unit variance noise).
* Metrics: per-bin balanced accuracy and macro-F1 over the 8 classes, and the
  per-trial majority-vote grasp accuracy (majority over bins predicted non-rest).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ap_backbone import APBackbone
from .db6_data import DEFAULT_CACHE, DEFAULT_TAU_MS, N_CH, N_CLASS, load_session
from .session_frontend import drift_augment

SOURCE_VAL_REPS = (4, 8, 12)
IGNORE = -100
WIN_MIN, WIN_MAX = 512, 1024     # training window length (bins)
BURN = 128                       # leading bins of a window excluded from the loss
P_START = 0.05                   # fraction of windows that start at bin 0 (no burn-in)
TRAIN_STD_FLOOR = 1e-3           # relative std floor used for source-session statistics (never binds on source)


# -------------------------------------------------------------- streams ------

class Stream:
    """One session as a continuous bin stream: x [T,N] envelope, y [T] labels, trial table."""

    def __init__(self, key, blob):
        self.key = tuple(key)
        self.x = blob["x"].astype(np.float32)
        self.y = blob["y"].astype(np.int64)
        self.tstart = blob["trial_start"].astype(np.int64)
        self.tend = blob["trial_end"].astype(np.int64)
        self.grasp = blob["trial_grasp"].astype(np.int64)
        self.rep = blob["trial_rep"].astype(np.int64)
        self.T = self.x.shape[0]
        self.lens = self.tend - self.tstart

    @property
    def n(self):
        return len(self.tstart)

    def bin_mask(self, trials) -> np.ndarray:
        """Boolean [T] mask of the bins that belong to the given trials."""
        m = np.zeros(self.T, bool)
        for i in trials:
            m[self.tstart[i]:self.tend[i]] = True
        return m


def load_stream(key, tau_ms=DEFAULT_TAU_MS, cache_dir=DEFAULT_CACHE) -> Stream:
    return Stream(key, load_session(*key, tau_ms=tau_ms, cache_dir=cache_dir))


def input_stats(stream: Stream, trials, floor: float = TRAIN_STD_FLOOR):
    """Per-channel mean / std over the bins of ``trials``; std floored at floor * median(std)."""
    flat = stream.x[stream.bin_mask(trials)].astype(np.float64)
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.maximum(sd, floor * max(float(np.median(sd)), 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore(x, mu, sd):
    """z-score a whole stream [T,N] (constants only, so causal)."""
    return ((x - mu) / sd).astype(np.float32)


def draw_calibration(b: int, rng, grasp, rep):
    """Class-balanced draw of b trials: round-robin over a random grasp order, random unused repetition.

    Rest is present in every trial.  Returns sorted trial indices; the grasp coverage is
    ``np.bincount(grasp[idx], minlength=8)[1:]`` (counts differ by at most 1 across grasps).
    """
    order = list(rng.permutation(7) + 1)
    chosen, used = [], {g: set() for g in order}
    i = 0
    while len(chosen) < b:
        g = int(order[i % 7]); i += 1
        free = [t for t in np.where(grasp == g)[0] if t not in used[g]]
        if not free:
            continue
        t = int(rng.choice(free)); used[g].add(t); chosen.append(t)
    return sorted(chosen)


def coverage(idx, grasp):
    """Per-grasp counts (7,) of a calibration set."""
    return np.bincount(grasp[np.asarray(idx)], minlength=N_CLASS)[1:].tolist()


def cal_segments(stream: Stream, cal):
    """Maximal runs of calibration trials that are adjacent in the stream -> list of (start, end) bins.

    Only these bins may be used for fitting: the neural data outside the B trials is not available
    (H4), so a run of adjacent trials is the longest legitimate contiguous context.
    """
    idx = sorted(int(i) for i in cal)
    segs = []
    for i in idx:
        a, b = int(stream.tstart[i]), int(stream.tend[i])
        if segs and segs[-1][1] == a:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    return segs


# ----------------------------------------------------------------- metrics ---

def confusion(truth, pred, n=N_CLASS):
    return np.bincount(truth * n + pred, minlength=n * n).reshape(n, n)


def class_scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Balanced accuracy (mean per-class recall) and macro-F1 over the classes present in truth."""
    cm = confusion(truth, pred)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    present = support > 0
    recall = tp[present] / support[present]
    prec = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
    rec_all = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    f1 = np.where(prec + rec_all > 0, 2 * prec * rec_all / np.maximum(prec + rec_all, 1e-12), 0.0)
    return {"bal_acc": float(recall.mean()), "macro_f1": float(f1[present].mean()),
            "acc": float(tp.sum() / cm.sum())}


def vote_grasp(pred_trial: np.ndarray) -> int:
    """Majority class among bins predicted non-rest; 0 (rest) if no bin is predicted non-rest."""
    nz = pred_trial[pred_trial > 0]
    return int(np.bincount(nz, minlength=N_CLASS).argmax()) if nz.size else 0


def score_stream(logits: np.ndarray, stream: Stream, trials) -> dict:
    """logits [T,8] of the one-pass stream decode, scored on the bins of ``trials`` only."""
    trials = [int(t) for t in trials]
    pred = logits.argmax(-1)
    m = stream.bin_mask(trials)
    out = class_scores(stream.y[m], pred[m])
    votes = np.array([vote_grasp(pred[stream.tstart[i]:stream.tend[i]]) for i in trials])
    out["vote_acc"] = float((votes == stream.grasp[trials]).mean())
    return out


# ---------------------------------------------------------------- sources ---

class SourceSet:
    """Source sessions (days 1-2): continuous streams, per-session train/val trial split, z-scored streams."""

    def __init__(self, keys, tau_ms=DEFAULT_TAU_MS, cache_dir=DEFAULT_CACHE, val_reps=SOURCE_VAL_REPS,
                 floor=TRAIN_STD_FLOOR):
        self.keys = [tuple(k) for k in keys]
        self.tau_ms = tau_ms
        self.floor = floor
        self.sess = [load_stream(k, tau_ms, cache_dir) for k in self.keys]
        self.train_idx = [np.where(~np.isin(s.rep, val_reps))[0] for s in self.sess]
        self.val_idx = [np.where(np.isin(s.rep, val_reps))[0] for s in self.sess]
        self.stats = [input_stats(s, i, floor) for s, i in zip(self.sess, self.train_idx)]
        self.xz = [zscore(s.x, mu, sd) for s, (mu, sd) in zip(self.sess, self.stats)]
        # label tensors for the loss: IGNORE everywhere except the bins of training trials
        self.y_loss = []
        for s, i in zip(self.sess, self.train_idx):
            y = np.full(s.T, IGNORE, np.int64)
            m = s.bin_mask(i)
            y[m] = s.y[m]
            self.y_loss.append(y)
        # fixed order-shuffled validation streams (trial order permuted; removes the 12-repetition grasp blocks)
        vr = np.random.default_rng(20261016)
        self.val_perm = [vr.permutation(s.n) for s in self.sess]
        self.sess_shuf = [reorder_stream(s, p) for s, p in zip(self.sess, self.val_perm)]
        self.val_idx_shuf = [np.where(np.isin(s.rep, val_reps))[0] for s in self.sess_shuf]
        self.xz_shuf = [zscore(s.x, mu, sd) for s, (mu, sd) in zip(self.sess_shuf, self.stats)]

    def shuffled_training_streams(self, rng, device):
        """One random trial-order permutation per session: (xz [T',N], y_loss [T']) GPU tensors, trial bins only.

        Training windows cut from these streams never see the 12-repetition grasp blocks of the recording
        (a stateful decoder could otherwise exploit 'same grasp as the previous trial').
        """
        xs, ys = [], []
        for s, S in enumerate(self.sess):
            order = rng.permutation(S.n)
            sl = [slice(int(S.tstart[i]), int(S.tend[i])) for i in order]
            xs.append(torch.from_numpy(np.concatenate([self.xz[s][a] for a in sl])).to(device))
            ys.append(torch.from_numpy(np.concatenate([self.y_loss[s][a] for a in sl])).to(device))
        return xs, ys

    def subject_of(self, i):
        return self.keys[i][0]


# ------------------------------------------------------------------ model ---

def build_model(n_sessions, *, k, width, layers, state_size, dropout, device):
    return APBackbone([N_CH] * n_sessions, N_CLASS, k=k, width=width, layers=layers,
                      state_size=state_size, dropout=dropout).to(device)


def trunk_hash(model) -> str:
    h = hashlib.sha256()
    for p in model.base.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def decode_z(base, z: torch.Tensor, max_tokens: int = 200_000) -> np.ndarray:
    """Causal one-pass decode of latent streams z [M,T,K] -> logits [M,T,8] (numpy, no grad).

    Every row is an independent stream run from zero state over its full length; there is no
    trial segmentation.  Rows are processed in chunks of ``max_tokens`` tokens.
    """
    m, t = z.shape[:2]
    step = max(1, max_tokens // t)
    outs = []
    with torch.inference_mode():
        for lo in range(0, m, step):
            outs.append(base(z[lo:lo + step]).float().cpu())
    return torch.cat(outs).numpy()


def decode_stream(base, xz: torch.Tensor, table: torch.Tensor, max_tokens: int = 200_000) -> np.ndarray:
    """xz [T,N] or [M,T,N] z-scored streams, table [K,N] or [M,K,N] -> logits [M,T,8]."""
    if xz.ndim == 2:
        xz = xz.unsqueeze(0)
    if table.ndim == 2:
        table = table.unsqueeze(0).expand(xz.shape[0], -1, -1)
    z = torch.einsum("mtn,mkn->mtk", xz, table)
    return decode_z(base, z, max_tokens)


def masked_ce(logits, y):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=IGNORE)


@torch.inference_mode()
def validate(model, src: SourceSet, xz_t, device, batch_sessions=6, xz_shuf_t=None):
    """Mean over source sessions of balanced accuracy / macro-F1 on the held-out repetitions,
    each session decoded in ONE contiguous causal pass (state carried across trials).

    ``xz_t`` are the original-order streams; ``xz_shuf_t`` (optional) the fixed order-shuffled streams
    (trial bins only), reported as ``shuf_bal_acc`` (the block-free metric).
    """
    model.eval()
    out = {}
    for tag, streams, sessions, vidx in (("", xz_t, src.sess, src.val_idx),
                                         ("shuf_", xz_shuf_t, src.sess_shuf, src.val_idx_shuf)):
        if streams is None:
            continue
        n = len(src.keys)
        tmax = max(s.T for s in sessions)
        bal, f1, head_bal = [None] * n, [None] * n, [None] * n
        for lo in range(0, n, batch_sessions):
            ids = list(range(lo, min(n, lo + batch_sessions)))
            x = torch.zeros(len(ids), tmax, N_CH, device=device)
            for j, s in enumerate(ids):
                x[j, :sessions[s].T] = streams[s]
            tab = torch.stack([model.tables[s] for s in ids])
            z = torch.einsum("mtn,mkn->mtk", x, tab)
            logits = model.base(z).float().cpu().numpy()
            head = (z @ model.head.weight.T + model.head.bias).float().cpu().numpy() if tag == "" else None
            for j, s in enumerate(ids):
                S, vi = sessions[s], vidx[s]
                sc = score_stream(logits[j, :S.T], S, vi)
                bal[s], f1[s] = sc["bal_acc"], sc["macro_f1"]
                if head is not None:
                    head_bal[s] = score_stream(head[j, :S.T], S, vi)["bal_acc"]
        out[tag + "bal_acc"] = float(np.mean(bal))
        out[tag + "macro_f1"] = float(np.mean(f1))
        out[tag + "per_session"] = [float(v) for v in bal]
        if tag == "":
            out["head_bal_acc"] = float(np.mean(head_bal))
    out["val_bal_acc"], out["val_macro_f1"], out["per_session"] = out["bal_acc"], out["macro_f1"], out["per_session"]
    return out


def train_backbone(src: SourceSet, out_dir, *, k=8, width=128, layers=2, state_size=32, dropout=0.1,
                   lr=1e-3, weight_decay=1e-2, batch=16, max_steps=15000, val_every=250, patience=8,
                   lam_head=0.5, lam_white=0.1, gain_sigma=0.3, offset_sigma=0.1, channel_dropout=0.05,
                   win_min=WIN_MIN, win_max=WIN_MAX, burn=BURN, p_start=P_START,
                   shuffle_trials=True, reshuffle_every=1000, select_on="shuf",
                   table_init="supervised", seed=0, device="cuda:0", log=lambda m: print(m, flush=True)):
    """Train on random-offset windows of the continuous source streams (loss on training-trial bins only
    after a burn-in prefix); early-stop on contiguous source-validation balanced accuracy.

    shuffle_trials: every ``reshuffle_every`` steps each session stream is rebuilt with a fresh random trial
    order, so windows never see the 12-repetition grasp blocks of the recording (no 'same grasp as before'
    shortcut).  select_on: 'shuf' = early stopping on the order-shuffled validation streams, 'orig' = on the
    original order.  Both metrics are always logged."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, lr=lr,
               weight_decay=weight_decay, batch=batch, max_steps=max_steps, val_every=val_every,
               patience=patience, lam_head=lam_head, lam_white=lam_white, gain_sigma=gain_sigma,
               offset_sigma=offset_sigma, channel_dropout=channel_dropout, table_init=table_init,
               win_min=win_min, win_max=win_max, burn=burn, p_start=p_start, seed=seed, tau_ms=src.tau_ms,
               decoding="contiguous", std_floor_train=src.floor, shuffle_trials=shuffle_trials,
               reshuffle_every=reshuffle_every, select_on=select_on)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    n = len(src.keys)
    model = build_model(n, k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, device=dev)
    xz_t = [torch.from_numpy(x).to(dev) for x in src.xz]
    y_t = [torch.from_numpy(y).to(dev) for y in src.y_loss]
    xz_shuf_t = [torch.from_numpy(x).to(dev) for x in src.xz_shuf]
    t_len = [s.T for s in src.sess]
    if table_init == "supervised":
        wins, labs = [], []
        for s in range(n):
            m = src.sess[s].bin_mask(src.train_idx[s])
            f = src.xz[s][m]
            lab = np.eye(N_CLASS, dtype=np.float32)[src.sess[s].y[m]]
            wins.append(torch.from_numpy(f)); labs.append(torch.from_numpy(lab - lab.mean(0)))
        model.init_from_supervised(wins, labs)
    elif table_init != "random":
        raise ValueError(table_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_step, bad, history = -np.inf, 0, 0, []
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        model.train()
        if shuffle_trials and (step - 1) % reshuffle_every == 0:
            tr_x, tr_y = src.shuffled_training_streams(rng, dev)
            t_len = [x.shape[0] for x in tr_x]
        elif not shuffle_trials:
            tr_x, tr_y = xz_t, y_t
        w = int(rng.integers(win_min, win_max + 1))
        sess = rng.integers(0, n, size=batch)
        xs, ys = [], []
        for s in sess:
            if rng.random() < p_start:
                o, bi = 0, 0
            else:
                o, bi = int(rng.integers(0, t_len[s] - w + 1)), burn
            xs.append(tr_x[s][o:o + w])
            y = tr_y[s][o:o + w].clone()
            y[:bi] = IGNORE
            ys.append(y)
        xs, ys = torch.stack(xs), torch.stack(ys)
        sess = torch.as_tensor(sess, device=dev)
        mask = ys != IGNORE
        xs = drift_augment(xs, gain_sigma, offset_sigma, channel_dropout)
        tables = torch.stack(list(model.tables))                          # [S, K, N]
        z = torch.einsum("btn,bkn->btk", xs, tables[sess])
        dec, head = model.base(z), model.head(z)
        loss_dec = masked_ce(dec, ys)
        loss = loss_dec + lam_head * masked_ce(head, ys)
        loss = loss + lam_white * model.whitening_penalty(z, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % val_every == 0 or step == 1:
            m = validate(model, src, xz_t, dev, xz_shuf_t=xz_shuf_t)
            ev = {"step": step, "loss": float(loss_dec), "elapsed": time.perf_counter() - t0,
                  "val_bal_acc": m["bal_acc"], "val_macro_f1": m["macro_f1"], "head_bal_acc": m["head_bal_acc"],
                  "val_shuf_bal_acc": m["shuf_bal_acc"], "val_shuf_macro_f1": m["shuf_macro_f1"]}
            history.append(ev)
            log(json.dumps(ev))
            score = m["shuf_bal_acc"] if select_on == "shuf" else m["bal_acc"]
            if score > best:
                best, best_step, bad = score, step, 0
                torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_step": step,
                            "val_bal_acc": m["bal_acc"], "val_macro_f1": m["macro_f1"],
                            "val_shuf_bal_acc": m["shuf_bal_acc"], "val_shuf_macro_f1": m["shuf_macro_f1"],
                            "val_per_session": m["per_session"], "val_shuf_per_session": m["shuf_per_session"],
                            "keys": src.keys, "stats": src.stats}, out / "best.pt")
            else:
                bad += 1
                if bad >= patience and step > 1:
                    break
    summary = {"cfg": cfg, "best_step": best_step, "best_select_score": best,
               "best_val_bal_acc": history[[h["step"] for h in history].index(best_step)]["val_bal_acc"],
               "best_val_shuf_bal_acc": history[[h["step"] for h in history].index(best_step)]["val_shuf_bal_acc"],
               "steps_run": step,
               "stopped_early": bool(bad >= patience), "budget_rule_ok": bool(best_step <= 0.8 * max_steps),
               "seconds": time.perf_counter() - t0,
               "n_params_trunk": int(sum(p.numel() for p in model.base.parameters())),
               "n_params_tables": int(sum(t.numel() for t in model.tables))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "history.json").write_text(json.dumps(history))
    return summary


def load_backbone(path, device="cuda:0"):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    model = build_model(len(blob["keys"]), k=c["k"], width=c["width"], layers=c["layers"],
                        state_size=c["state_size"], dropout=0.0, device=torch.device(device))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob


# ---------------------------------------------------- target-side routines --

def pack_segments(segs, ys, probs, max_tokens=80_000):
    """Group (x [L,N], y [L], problem id) segments into padded chunks of at most ``max_tokens`` tokens.

    Segments are sorted by length so that padding stays small.  Returns a list of (X [c,Lc,N],
    Y [c,Lc] with IGNORE padding, P [c]) tensors on the device of the inputs.
    """
    order = sorted(range(len(segs)), key=lambda i: segs[i].shape[0])
    chunks, cur = [], []
    for i in order:
        longest = segs[i].shape[0]
        if cur and longest * (len(cur) + 1) > max_tokens:
            chunks.append(cur); cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    out = []
    for ch in chunks:
        lc = max(segs[i].shape[0] for i in ch)
        x = segs[0].new_zeros(len(ch), lc, segs[0].shape[1])
        y = ys[0].new_full((len(ch), lc), IGNORE)
        for j, i in enumerate(ch):
            x[j, :segs[i].shape[0]] = segs[i]
            y[j, :segs[i].shape[0]] = ys[i]
        out.append((x, y, torch.as_tensor([probs[i] for i in ch], device=segs[0].device)))
    return out


def fit_tables(model, segs, ys, probs, init_tables, *, steps=300, lr=3e-3, max_tokens=80_000):
    """Gradient fit of read-in tables only, through the frozen trunk (ui_prior style).

    segs: list of z-scored calibration segments [L,N] (each a run of adjacent calibration trials, decoded
    from zero state), ys: list of [L] labels, probs: list of problem ids in 0..D-1, init_tables [D,K,N].
    The D problems are independent (separate Adam elements; the loss is a sum over problems, each the
    mean CE over its own bins).  Returns tables [D,K,N] and the final per-problem loss.
    """
    for p in model.base.parameters():
        if p.requires_grad:
            raise RuntimeError("trunk must be frozen")
    table = torch.nn.Parameter(init_tables.clone())
    opt = torch.optim.Adam([table], lr=lr)
    d = table.shape[0]
    chunks = pack_segments(segs, ys, probs, max_tokens)
    count = torch.zeros(d, device=table.device)
    for _, y, p in chunks:
        count.index_add_(0, p, (y != IGNORE).sum(1).float())
    last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_each = torch.zeros(d, device=table.device)
        for x, y, p in chunks:
            z = torch.einsum("cln,ckn->clk", x, table[p])
            dec = model.base(z)
            ce = torch.nn.functional.cross_entropy(dec.reshape(-1, dec.shape[-1]), y.reshape(-1),
                                                   ignore_index=IGNORE, reduction="none").reshape(y.shape).sum(1)
            part = torch.zeros(d, device=table.device).index_add_(0, p, ce) / count
            part.sum().backward()
            loss_each += part.detach()
        opt.step()
        last = loss_each
    return table.detach(), last.cpu().numpy()


def causal_ema(x: np.ndarray, beta: float) -> np.ndarray:
    """Causal exponential smoothing along axis -2 (time): y[t] = beta*y[t-1] + (1-beta)*x[t]."""
    if beta <= 0:
        return x
    from scipy.signal import lfilter
    return lfilter([1 - beta], [1.0, -beta], x, axis=-2).astype(np.float32)


def onehot(y):
    return np.eye(N_CLASS, dtype=np.float64)[y]


def ridge_fit(xz_segs, y_segs, lam, ema=0.0):
    """Closed-form ridge (N -> 8 one-hot) on causally smoothed features of the calibration segments.

    xz_segs: list of z-scored [L,N] calibration segments, y_segs: list of [L] labels.  The EMA runs from
    zero state inside each segment (no data outside the calibration trials).  Returns W [N,8], b [8].
    A linear least-squares classifier on one-hot targets (LDA-style, closed form).
    """
    f = np.concatenate([causal_ema(s, ema) for s in xz_segs]).astype(np.float64)
    t = onehot(np.concatenate(y_segs))
    fm, tm = f.mean(0), t.mean(0)
    fc = f - fm
    a = fc.T @ fc + lam * len(fc) / 1000.0 * np.eye(fc.shape[1])
    w = np.linalg.solve(a, fc.T @ (t - tm))
    return w, tm - fm @ w


def ridge_logits(xz, w, b, ema=0.0):
    """Ridge scores for a whole stream xz [T,N]: one causal EMA pass over the stream, then the linear map."""
    return causal_ema(xz, ema) @ w + b


def segment_arrays(stream: Stream, xz: np.ndarray, cal):
    """Calibration segments of a z-scored stream: (list of [L,N] features, list of [L] labels)."""
    segs = cal_segments(stream, cal)
    return [xz[a:b] for a, b in segs], [stream.y[a:b] for a, b in segs]


# ---------------------------------------------------------- protocol keys ---

# S2_D2_T2 is a truncated recording (103 s, a single grasp, 12 trials): excluded from the source set.
EXCLUDED_SOURCE = ((2, 2, 2),)


def source_keys(subjects=range(1, 11), days=(1, 2)):
    return [(s, d, t) for s in subjects for d in days for t in (1, 2) if (s, d, t) not in EXCLUDED_SOURCE]


def target_keys(subjects=range(1, 11), days=(3, 4, 5)):
    return [(s, d, t) for s in subjects for d in days for t in (1, 2)]


def latest_source_index(keys, subject):
    """Index of the subject's chronologically latest source session in ``keys``."""
    idx = [i for i, k in enumerate(keys) if k[0] == subject]
    return max(idx, key=lambda i: (keys[i][1], keys[i][2]))


# ----------------------------------------------------------- diagnostics ---

def reorder_stream(stream: Stream, order) -> Stream:
    """New stream made only of the trials of ``order`` (trial indices) concatenated in that order.

    Used for the order-shuffle diagnostic: DB6 records each grasp in one block of 12 consecutive
    repetitions, so a stateful decoder could exploit "same grasp as the previous trial".  Shuffling the trial
    order removes that block structure (the join between non-adjacent trials is a jump in the envelope).
    """
    order = [int(i) for i in order]
    x = np.concatenate([stream.x[stream.tstart[i]:stream.tend[i]] for i in order])
    y = np.concatenate([stream.y[stream.tstart[i]:stream.tend[i]] for i in order])
    lens = stream.lens[order]
    ends = np.cumsum(lens)
    blob = {"x": x, "y": y, "trial_start": ends - lens, "trial_end": ends,
            "trial_grasp": stream.grasp[order], "trial_rep": stream.rep[order]}
    return Stream(stream.key, blob)
