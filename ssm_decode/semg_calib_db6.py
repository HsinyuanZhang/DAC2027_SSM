"""DB6 (NinaPro D-A, v2 contiguous streaming) episodic environment for the sEMG calibrators.

The session is ONE continuous causal stream; the trunk state is carried across trial boundaries
(trial boundaries only choose which bins are scored / which trials are summed for the descriptors).
The v2 backbones are trained on trial-order-shuffled streams (the 12-repetition grasp blocks of the
recordings would otherwise be exploitable by a stateful decoder), so the calibrator episodes and the
source validation use trial-order-shuffled copies of the source streams as well.
Episode = a random-offset window of a SOURCE session stream; B calibration trials (training
repetitions, class-balanced) give the descriptors and the z-score statistics; the loss is the CE on the
bins of the held-out (validation-repetition) trials inside the window, after a burn-in prefix.
Target sessions (days 3-5) are never loaded by the training code.

Priors: ``mean`` = channel-aligned mean of the source tables of the OTHER subjects; ``own`` = a
strictly earlier session of the same subject (random earlier session in training, the latest source
session at test time); ``free`` = identity withheld.
"""
from __future__ import annotations

import numpy as np
import torch

from . import db6_backbone as db6b
from .semg_calib import SemgSpec, SemgStats, TrialBank, mean_prior
from .semg_calib_env import forward_table, table_rms

B_TRAIN_MAX = 20
WIN_MIN, WIN_MAX, BURN = 512, 1024, 128


class StreamStore:
    """GPU copy of one session stream + lazily banked per-trial sums (calibration trials only)."""

    def __init__(self, stream, spec: SemgSpec, device):
        self.stream = stream
        self.x = torch.from_numpy(stream.x).to(device)
        self.y = torch.from_numpy(stream.y).to(device)
        self.spec = spec
        self.bank = TrialBank(spec)
        self.device = device

    def stats(self, idx) -> SemgStats:
        for i in idx:
            if not self.bank.has(i):
                a, b = int(self.stream.tstart[i]), int(self.stream.tend[i])
                yi = torch.nn.functional.one_hot(self.y[a:b], db6b.N_CLASS).double()
                self.bank.add(i, self.x[a:b].double(), yi, torch.ones(b - a, dtype=torch.bool, device=self.device))
        return SemgStats.from_bank(self.bank, idx)


class Db6StreamEnv:
    kind = "db6"
    metric = "bal_acc"

    def __init__(self, ckpt, device="cuda:0", std_floor=None, n_copies=6):
        """``std_floor``: relative floor (x median channel std) of the B-trial z-score statistics; default = the
        backbone's training floor.  The same value is used in calibrator training episodes and at the target.
        ``device='cpu'`` uses the CPU port of the frozen trunk (slower; only for GPU-less training)."""
        if device == "cpu":
            return self._init_cpu(ckpt, std_floor, n_copies)
        self.dev = torch.device(device)
        self.model, blob = db6b.load_backbone(ckpt, device=device)
        self.blob = blob
        self.keys = [tuple(k) for k in blob["keys"]]
        assert self.keys == db6b.source_keys()
        self.tau = blob["cfg"]["tau_ms"]
        floor = float(std_floor) if std_floor is not None else blob["cfg"].get("std_floor_train", db6b.TRAIN_STD_FLOOR)
        self.src = db6b.SourceSet(self.keys, tau_ms=self.tau)
        self.tables = torch.stack([t.detach() for t in self.model.tables]).to(self.dev)
        self.k = self.model.k
        self._finish_init(floor, n_copies)

    def _finish_init(self, floor, n_copies):
        self.spec = SemgSpec("db6", 14, self.k, db6b.N_CLASS, (0,), std_floor_ref="median", std_floor=floor)
        self.out_scale = table_rms(self.tables)
        self.sess = [StreamStore(s, self.spec, self.dev) for s in self.src.sess]      # per-trial sums (any order)
        # trial-order-shuffled copies of the source streams for episodes (R fixed permutations per session)
        self.n_copies = n_copies
        prng = np.random.default_rng(20261017)
        self.copies = []
        for j, S in enumerate(self.src.sess):
            row = []
            for _ in range(n_copies):
                perm = prng.permutation(S.n)
                C = db6b.reorder_stream(S, perm)
                vm = np.where(np.isin(C.rep, db6b.SOURCE_VAL_REPS))[0]
                row.append({"x": torch.from_numpy(C.x).to(self.dev), "y": torch.from_numpy(C.y).to(self.dev),
                            "q": torch.from_numpy(C.bin_mask(vm)).to(self.dev), "T": C.T})
            self.copies.append(row)
        self.subject = [k[0] for k in self.keys]
        self.val_sess = sorted({db6b.latest_source_index(self.keys, s) for s in set(self.subject)})

    def _init_cpu(self, ckpt, std_floor, n_copies):
        """Training environment without a GPU (CPU port of the frozen trunk; weights are constants)."""
        import types
        self.dev = torch.device("cpu")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.blob = blob
        self.keys = [tuple(k) for k in blob["keys"]]
        assert self.keys == db6b.source_keys()
        self.tau = blob["cfg"]["tau_ms"]
        floor = float(std_floor) if std_floor is not None else blob["cfg"].get("std_floor_train", db6b.TRAIN_STD_FLOOR)
        sd = blob["state_dict"]
        self.tables = torch.stack([sd[f"tables.{i}"].detach().cpu() for i in range(len(self.keys))])
        self.k = int(blob["cfg"]["k"])
        self.model = types.SimpleNamespace(base=TrainableCpuTrunk(sd), parameters=lambda: [], k=self.k)
        self.cpu_decode = _LongCpuDecoder(sd)
        self.src = db6b.SourceSet(self.keys, tau_ms=self.tau)
        self._finish_init(floor, n_copies)

    def decode(self, z):
        """z [M,T,K] -> logits [M,T,8] (numpy), one contiguous causal pass per row."""
        if getattr(self, "cpu_decode", None) is not None:
            with torch.no_grad():
                outs, step = [], max(1, 200_000 // z.shape[1])
                for lo in range(0, z.shape[0], step):
                    outs.append(self.cpu_decode(z[lo:lo + step]))
            return torch.cat(outs).numpy()
        return db6b.decode_z(self.model.base, z)

    def trunk_hash(self):
        return db6b.trunk_hash(self.model)

    # -- priors -----------------------------------------------------------------
    def earlier(self, j):
        kj = self.keys[j]
        return [i for i, k in enumerate(self.keys) if k[0] == kj[0] and (k[1], k[2]) < (kj[1], kj[2])]

    def prior_for(self, source, j, rng=None, latest=False, exclude_val=True):
        if source == "free":
            return None
        if source == "mean":
            ex = [i for i in range(len(self.keys)) if self.subject[i] == self.subject[j]]
            if exclude_val:
                ex = ex + list(self.val_sess)
            return mean_prior(self.tables, ex)
        if source == "own":
            prev = self.earlier(j)
            if not prev:
                raise ValueError("no earlier session")
            if latest or rng is None:
                i = max(prev, key=lambda i: (self.keys[i][1], self.keys[i][2]))
            else:
                i = int(rng.choice(prev))
            return self.tables[i].clone()
        raise ValueError(source)

    def train_sessions(self, source):
        out = [j for j in range(len(self.keys)) if j not in self.val_sess]
        if source == "own":
            out = [j for j in out if self.earlier(j)]
        return out

    # -- episodes -----------------------------------------------------------------
    def _window(self, rng, j, r, length):
        C = self.copies[j][r]
        for _ in range(30):
            o = int(rng.integers(0, C["T"] - length + 1))
            if int(C["q"][o + BURN:o + length].sum()) >= 64:
                return o
        return o

    def sample_episode(self, rng, train_sess, source, b_max=B_TRAIN_MAX):
        j = int(rng.choice(train_sess))
        b = int(min(b_max, np.floor(np.exp(rng.uniform(0, np.log(b_max + 1))))))
        S, tr = self.src.sess[j], self.src.train_idx[j]
        loc = db6b.draw_calibration(b, rng, S.grasp[tr], S.rep[tr])
        length = int(rng.integers(WIN_MIN, WIN_MAX + 1))
        r = int(rng.integers(self.n_copies))
        return {"s": j, "r": r, "cal": [int(tr[i]) for i in loc], "o": self._window(rng, j, r, length), "L": length,
                "prior": self.prior_for(source, j, rng)}

    def val_episodes(self, source, budgets=(1, 2, 4, 8, 16), draws=2):
        eps = []
        for j in self.val_sess:
            if source == "own" and not self.earlier(j):
                continue
            S, tr = self.src.sess[j], self.src.train_idx[j]
            for b in budgets:
                for d in range(draws):
                    rng = np.random.default_rng([556, j, b, d])
                    loc = db6b.draw_calibration(b, rng, S.grasp[tr], S.rep[tr])
                    eps.append({"s": j, "cal": [int(tr[i]) for i in loc],
                                "prior": self.prior_for(source, j, None, latest=True), "b": b})
        return eps

    def val_x(self, j):
        if not hasattr(self, "_val_x"):
            self._val_x = {}
        if j not in self._val_x:
            self._val_x[j] = torch.from_numpy(self.src.sess_shuf[j].x).to(self.dev)
        return self._val_x[j]

    # -- loss / metric --------------------------------------------------------------
    def loss(self, calib, eps):
        zs, ys = [], []
        top = max(e["L"] for e in eps)
        for e in eps:
            st = self.sess[e["s"]].stats(e["cal"])
            table, mean, std = forward_table(calib, st, e["prior"], grad=True)
            o, L = e["o"], e["L"]
            C = self.copies[e["s"]][e["r"]]
            xw = C["x"][o:o + L]
            z = ((xw - mean) / std) @ table.T
            y = torch.full((top,), db6b.IGNORE, dtype=torch.long, device=self.dev)
            m = C["q"][o:o + L].clone()
            m[:min(BURN, L) if o > 0 else 0] = False
            y[:L] = torch.where(m, C["y"][o:o + L], torch.full_like(C["y"][o:o + L], db6b.IGNORE))
            zs.append(torch.nn.functional.pad(z, (0, 0, 0, top - L))); ys.append(y)
        dec = self.model.base(torch.stack(zs))
        return db6b.masked_ce(dec, torch.stack(ys))

    @torch.no_grad()
    def val_score(self, calib, eps):
        by_sess = {}
        for e in eps:
            by_sess.setdefault(e["s"], []).append(e)
        scores = []
        for j, group in by_sess.items():
            S = self.src.sess_shuf[j]                      # fixed trial-order-shuffled validation stream
            xs = self.val_x(j)
            zs = []
            for e in group:
                st = self.sess[j].stats(e["cal"])          # sums of the calibration trials (original indices)
                table, mean, std = forward_table(calib, st, e["prior"], grad=False)
                zs.append(((xs - mean) / std) @ table.T)
            logits = self.decode(torch.stack(zs))
            for i in range(len(group)):
                scores.append(db6b.score_stream(logits[i], S, self.src.val_idx_shuf[j])["bal_acc"])
        return float(np.mean(scores))


# ------------------------------------------------------------ CPU eval fallback ----

class _LongCpuDecoder:
    """The pinned-Mamba-3 CPU port (``mamba3_cpu.CPUDecoder``) with the 128-bin window guard lifted.

    The port already carries the SSM state across its internal 64-bin chunks, so a full stream is decoded in one
    causal pass exactly like the Triton kernel up to bf16 rounding (checked against stored GPU rows by the
    ``--check-ref`` rows of the eval script).
    """

    def __init__(self, state_dict):
        from .mamba3_cpu import CPUDecoder

        class Long(CPUDecoder):
            context = 10 ** 9

        sd = {k[5:]: v.detach().cpu() for k, v in state_dict.items() if k.startswith("base.")}
        self.dec = Long.from_state_dict(sd)
        import hashlib
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(sd[k].numpy().tobytes())
        self.hash = h.hexdigest()

    def __call__(self, z):
        return self.dec.forward(z.float().cpu())


class TrainableCpuTrunk(_LongCpuDecoder):
    """Differentiable (w.r.t. its input only) variant of the CPU trunk: used for source-side episodic training
    when no GPU is available.  The trunk weights are constants."""

    def __call__(self, z):
        from .mamba3_cpu import _ln, _linear
        import torch.nn.functional as F
        d = self.dec
        h = _linear(z.float(), d.in_w, d.in_b)
        for b in d.blocks:
            h = h + d._core(_ln(h, b.norm1_w, b.norm1_b), b)
            h = h + _linear(F.silu(_linear(_ln(h, b.norm2_w, b.norm2_b), b.ff1_w, b.ff1_b)), b.ff2_w, b.ff2_b)
        return _linear(_ln(h, d.final_w, d.final_b), d.out_w, d.out_b)


class Db6CpuEnv:
    """Eval-only DB6 environment on CPU (no training, no source streams): used when no GPU is available."""
    kind = "db6"

    def __init__(self, ckpt, std_floor=None):
        self.dev = torch.device("cpu")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.blob = blob
        self.keys = [tuple(k) for k in blob["keys"]]
        assert self.keys == db6b.source_keys()
        self.tau = blob["cfg"]["tau_ms"]
        self.k = int(blob["cfg"]["k"])
        floor = float(std_floor) if std_floor is not None else blob["cfg"].get("std_floor_train", db6b.TRAIN_STD_FLOOR)
        sd = blob["state_dict"]
        self.tables = torch.stack([sd[f"tables.{i}"].detach().cpu() for i in range(len(self.keys))])
        self.spec = SemgSpec("db6", 14, self.k, db6b.N_CLASS, (0,), std_floor_ref="median", std_floor=floor)
        self.out_scale = table_rms(self.tables)
        self.subject = [k[0] for k in self.keys]
        self.val_sess = sorted({db6b.latest_source_index(self.keys, s) for s in set(self.subject)})
        self.trunk = _LongCpuDecoder(sd)
        self.tables_ref = self.tables

    prior_for = Db6StreamEnv.prior_for
    earlier = Db6StreamEnv.earlier

    def decode(self, z):
        outs = []
        step = max(1, 200_000 // z.shape[1])
        for lo in range(0, z.shape[0], step):
            outs.append(self.trunk(z[lo:lo + step]))
        return torch.cat(outs).numpy()

    def trunk_hash(self):
        return self.trunk.hash
