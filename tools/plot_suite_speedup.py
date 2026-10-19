"""Render the README figure from the published aggregate CSV; no simulation."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'media/benchmarks/suite-throughput.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'media/benchmarks/libero-suite-speedup.svg')
    parser.add_argument('--png', type=Path, help='optional external preview/export file')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    rows = list(csv.DictReader(args.data.open()))
    matched = np.array([float(r['matched_speedup']) for r in rows])
    recorded = np.array([float(r['recorded_speedup']) for r in rows])
    for row in rows:
        for prefix in ('matched', 'recorded'):
            ratio = float(row[prefix + '_cpu_elapsed_s']) / float(row[prefix + '_gpu_elapsed_s'])
            if abs(ratio - float(row[prefix + '_speedup'])) > 1e-12:
                raise ValueError('aggregate elapsed time and speedup differ: ' + row['suite'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(10.5, 7.8), facecolor='white')
    fig.subplots_adjust(left=.27, right=.95, top=.79, bottom=.18)
    y = np.arange(len(rows), dtype=float)
    y[5:] += .35
    y[6:] += .35
    teal, gray = '#087F8C', '#B3BEC9'
    ax.barh(y - .17, matched, height=.28, color=teal, label='Matching time events')
    ax.barh(y + .17, recorded, height=.28, color=gray, label='All recorded tasks')
    for offset, values, color in [(-.17, matched, '#065E68'), (.17, recorded, '#536170')]:
        for i, (position, value) in enumerate(zip(y, values)):
            ax.text(value + .014, position + offset, f'{value:.3f}×', va='center',
                    fontsize=10.5, color=color, weight='bold' if i == len(rows)-1 else 'normal')
    labels = [f"{r['suite']}  ({r['matched_tasks']}/{r['recorded_tasks']})" for r in rows]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.10, float(max(matched.max(), recorded.max())) * 1.18))
    ax.xaxis.set_major_locator(MultipleLocator(.25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}×'))
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E7ECF0', linewidth=.8)
    ax.axvline(1, color='#334155', linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(1, -.66, 'CPU parity', ha='center', color='#334155', fontsize=10)
    for position in (4.67, 6.02):
        ax.axhline(position, color='#E7ECF0', linewidth=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, pad=9, labelcolor='#334155')
    ax.set_xlabel('GPU throughput / CPU throughput · higher is faster', labelpad=13, color='#334155')
    fig.text(.04, .958, 'LIBERO suite performance', fontsize=23, weight='bold', color='#14263D')
    fig.text(.04, .914, 'A100 SXM4 40GB · N=2 · FP64 · no cameras', fontsize=12.5, color='#536170')
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc='upper left', bbox_to_anchor=(.032, .883),
               frameon=False, ncol=2, fontsize=11, handlelength=1.5, columnspacing=2)
    fig.text(.04, .817, 'Task counts: matching / recorded', fontsize=10, color='#536170')
    fig.text(.04, .067, '5 warmup + 20 timed control steps per task · one measurement · recorded 2026-10-19',
             fontsize=9.5, color='#536170')
    fig.text(.04, .037, 'One reset-event mismatch is excluded from matching tasks. LIBERO-100 = LIBERO-90 + LIBERO-10.',
             fontsize=9.5, color='#536170')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Date': None}, facecolor='white')
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=180, facecolor='white')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
