"""Aggregate and plot published four-backend measurements; no simulation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('cpu_fp64', 'gpu_fp64', 'mjwarp_fp32', 'gpu_fp32')
SUITES = (('LIBERO-Spatial', ('libero_spatial',), 10),
          ('LIBERO-Object', ('libero_object',), 10),
          ('LIBERO-Goal', ('libero_goal',), 10),
          ('LIBERO-90', ('libero_90',), 90),
          ('LIBERO-10', ('libero_10',), 10),
          ('All distinct tasks', (), 130))


def aggregate(rows):
    """Sum measured work and times; keep every recorded task, including flags."""
    if len(rows) != 130 or len({r['task'] for r in rows}) != 130:
        raise ValueError('Expected 130 distinct recorded tasks')
    for row in rows:
        if (int(row['worlds']), int(row['warmup_control_steps']),
                int(row['timed_control_steps'])) != (2, 5, 20):
            raise ValueError('Mixed benchmark protocol: ' + row['task'])
        for backend in BACKENDS:
            value = float(row[backend + '_elapsed_s'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid elapsed time: ' + row['task'])
    result = []
    for label, suites, expected in SUITES:
        selected = [r for r in rows if not suites or r['suite'] in suites]
        if len(selected) != expected:
            raise ValueError('Unexpected task coverage: ' + label)
        work = sum(int(r['worlds']) * int(r['timed_control_steps']) for r in selected)
        item = dict(suite=label, tasks=len(selected), environment_control_steps=work)
        for backend in BACKENDS:
            elapsed = math.fsum(float(r[backend + '_elapsed_s']) for r in selected)
            item[backend + '_elapsed_s'] = elapsed
            item[backend + '_env_steps_s'] = work / elapsed
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'media/benchmarks/task-throughput.csv')
    parser.add_argument('--aggregate', type=Path, default=ROOT / 'media/benchmarks/suite-throughput.csv')
    parser.add_argument('--provenance', type=Path, default=ROOT / 'media/benchmarks/provenance.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'media/benchmarks/libero-suite-throughput.svg')
    parser.add_argument('--png', type=Path, help='optional external preview/export')
    args = parser.parse_args()
    with args.data.open(newline='') as source:
        rows = aggregate(list(csv.DictReader(source)))
    metadata = json.loads(args.provenance.read_text())
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != metadata['published_task_csv_sha256']:
        raise ValueError('Published per-task data differs from its provenance')
    if not metadata['unlike_timing_scopes'] or metadata['new_physics_executed']:
        raise ValueError('This figure represents historical measurements with different timing scopes')
    args.aggregate.parent.mkdir(parents=True, exist_ok=True)
    with args.aggregate.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False,
                         'svg.hashsalt': 'flash-libero-recorded-throughput-v1'})
    colors = ('#64748B', '#325DAB', '#D88726', '#087F8C')
    names = ('CPU FP64 · public step', 'GPU FP64 · public step',
             'MJWarp FP32 · device control', 'GPU FP32 · device control')
    fig, ax = plt.subplots(figsize=(11.8, 10.0), facecolor='white')
    fig.subplots_adjust(left=.235, right=.94, top=.79, bottom=.235)
    positions = np.arange(len(rows), dtype=float)
    positions[-1] += .35
    maximum = 0.0
    for index, (backend, color, name) in enumerate(zip(BACKENDS, colors, names)):
        values = np.array([r[backend + '_env_steps_s'] for r in rows])
        maximum = max(maximum, float(values.max()))
        offset = (index - 1.5) * .195
        ax.barh(positions + offset, values, height=.165, color=color,
                edgecolor='white', linewidth=.4, hatch='//' if index >= 2 else None, label=name)
        for y, value in zip(positions + offset, values):
            ax.text(value + 1.0, y, f'{value:.2f}', va='center', color=color, fontsize=10)
    ax.set_yticks(positions, [f"{r['suite']}  ({r['tasks']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.14)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E7ECF0', linewidth=.8)
    ax.axhline((positions[-1] + positions[-2]) / 2, color='#D7DFE8', linewidth=.9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, pad=8, labelcolor='#334155')
    ax.set_xlabel('Environment control steps / second · both worlds combined', labelpad=12)
    fig.text(.035, .963, 'LIBERO recorded throughput', fontsize=23, weight='bold', color='#14263D')
    fig.text(.035, .925, 'A100 SXM4 40GB · N=2 · no cameras · 130 distinct tasks',
             fontsize=12.5, color='#536170')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.027, .904),
               frameon=False, ncol=2, fontsize=10.5, columnspacing=2)
    fig.text(.035, .815, 'Historical measurements · different timing scopes; not a controlled four-way ranking',
             fontsize=11, weight='bold', color='#7C4922')
    notes = (
        '5 warmup + 20 timed controls per task · one measurement · throughput = total work / total time',
        'FP64: 2026-10-19 public step wall time, including host observations and task predicates.',
        'FP32: 2026-10-21 device control time, excluding those host stages; measured on a shared GPU.',
        'All recorded tasks included; LIBERO-90 task 85 has a pre-existing FP64 reset/time-event difference.',
        'GPU FP32 measurements precede the ND-05 GJK correction. No new physics or timing was run.'
    )
    for i, note in enumerate(notes):
        fig.text(.035, .145 - i * .024, note, fontsize=9.4, color='#536170')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Date': None}, facecolor='white')
    if args.output.suffix.lower() == '.svg':
        args.output.write_text('\n'.join(line.rstrip() for line in
                                        args.output.read_text().splitlines()) + '\n')
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=160, facecolor='white')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
