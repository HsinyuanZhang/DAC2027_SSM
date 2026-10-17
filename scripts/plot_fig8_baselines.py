"""Plot the published-baseline comparison figure (exp-log fig8) from the reviewed baselines cells.

Reads results/analysis/baselines_20261016/cells.csv (two-pass reviewed numbers; query R2 in physical
units, mean over draws) and writes a 3-panel PDF (days / month / year).  APST rows are deliberately
not plotted (APST is unpublished work and is not part of the paper's comparison).

Usage:
  python scripts/plot_fig8_baselines.py \
      --cells results/analysis/baselines_20261016/cells.csv \
      --out overleaf/figures/fig8_baselines_688.pdf
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUDGETS = [1, 4, 8, 16, 33]
STRATA = [('days', '(a) days (2 targets: 0616, 0617)'),
          ('month', '(b) month (1 target: 20140626)'),
          ('year', '(c) year (1 target: 20150511)')]

# Okabe-Ito colorblind-safe palette.
BLUE, ORANGE, GREEN, VERMILLION = '#0072B2', '#E69F00', '#009E73', '#D55E00'
SKY, PINK, YELLOW, GREY, BLACK = '#56B4E9', '#CC79A7', '#F0E442', '#7f7f7f', '#000000'

# (method key in cells.csv, legend label, color, linestyle, marker, linewidth)
LINES = [
    ('p_prior', 'P-prior (level 0)', GREY, ':', 'v', 1.4),
    ('p0', 'P0 (level 1)', BLUE, '-', 'o', 2.6),
    ('R3', 'P0 + R3 (level 3)', VERMILLION, '-', 's', 2.6),
    ('stab_pool_a', 'Stabilizer, electrode-pooled', GREEN, '-', '^', 1.6),
    ('stab_a', 'Stabilizer, sorted units [variant]', GREEN, '--', '^', 1.2),
    ('cca_pool_sqrt_a', 'Gallego PCA+CCA, electrode-pooled', ORANGE, '-', 'X', 1.6),
    ('cca_sqrt_a', 'Gallego PCA+CCA, sorted units', ORANGE, '--', 'X', 1.2),
    ('cca_a_ridge', 'Ridge-CCA (ours, uses direction labels)', PINK, '-.', 'D', 1.8),
]
# B-independent rows (use the whole target recording, violate H4): drawn as one marker at the right edge.
NOMAD = ('nomad_pooled', 'NoMAD, electrode-pooled (all target trials, backprop)', BLACK, 'P')


def load(path):
    cells = defaultdict(dict)
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            cells[(row['method'], row['stratum'])][int(row['B'])] = (float(row['mean']), float(row['ci95']))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cells = load(args.cells)
    xs = list(range(len(BUDGETS)))
    x_nomad = len(BUDGETS) + 0.6
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    handles = {}
    for axis, (stratum, title) in zip(axes, STRATA):
        for key, label, color, style, marker, width in LINES:
            series = cells[(key, stratum)]
            ys = [series[b][0] for b in BUDGETS]
            handle, = axis.plot(xs, ys, color=color, linestyle=style, marker=marker, linewidth=width,
                                markersize=5.5, label=label,
                                zorder=4 if key in ('p0', 'R3') else 3)
            handles.setdefault(label, handle)
        key, label, color, marker = NOMAD
        series = cells[(key, stratum)]
        # B-independent: mean over the five budgets (they differ only by query mask).
        value = sum(series[b][0] for b in BUDGETS) / len(BUDGETS)
        handle = axis.scatter([x_nomad], [value], color=color, marker=marker, s=70, zorder=5, label=label)
        handles.setdefault(label, handle)
        axis.axhline(0, color='k', linewidth=0.6)
        axis.set_xticks(xs + [x_nomad])
        axis.set_xticklabels([f'B={b}' for b in BUDGETS] + ['all\ntrials'])
        axis.axvline(len(BUDGETS) - 0.4, color=GREY, linewidth=0.6, linestyle=':')
        axis.set_xlabel('Calibration budget B (labeled trials)')
        axis.set_title(title, fontsize=10, fontweight='bold')
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel('Query R$^2$ (physical units)')
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)


if __name__ == '__main__':
    main()
