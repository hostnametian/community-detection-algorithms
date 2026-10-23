"""Draw Q--ACD panels with open axes and collision-free city labels."""

import csv
import os
from pathlib import Path
from paths import ANALYSIS_ROOT, table_path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-r1-scatter')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = ANALYSIS_ROOT
ALGORITHMS = ['Infomap', 'LeidenRBC', 'DCSBM', 'GeoLP', 'Spinglass', 'DirLou']
CITIES = {'BeiJing': 'BJ', 'ShangHai': 'SH', 'HongKong': 'HK',
          'Moscow': 'MS', 'StPetersburg': 'SP', 'Kazan': 'KZ',
          'Yekaterinburg': 'YE'}


def place_labels(fig, ax, rows):
    """Prefer short offsets, keeping labels inside the axes and off markers."""
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    inside = Bbox.from_extents(bounds.x0 + 3, bounds.y0 + 3,
                               bounds.x1 - 3, bounds.y1 - 3)
    points = [ax.transData.transform((float(r['Q']), float(r['ACD']))) for r in rows]
    markers = [Bbox.from_extents(x - 5, y - 5, x + 5, y + 5) for x, y in points]
    occupied = []
    offsets = [(6, 6), (6, -6), (-6, 6), (-6, -6), (0, 10), (0, -10)]
    offsets += [(dx, dy) for radius in (15, 22, 30)
                for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, radius), (-radius, radius),
                               (radius, -radius), (-radius, -radius))]
    for row in rows:
        label = ax.annotate(CITIES[row['city']],
                            (float(row['Q']), float(row['ACD'])),
                            xytext=(6, 6), textcoords='offset points', fontsize=9,
                            annotation_clip=False)
        for dx, dy in offsets:
            label.set_position((dx, dy))
            label.set_ha('left' if dx > 0 else 'right' if dx < 0 else 'center')
            label.set_va('bottom' if dy > 0 else 'top' if dy < 0 else 'center')
            box = label.get_window_extent(renderer).padded(1.5)
            if (inside.contains(box.x0, box.y0) and inside.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in occupied + markers)):
                occupied.append(box)
                break
        else:
            raise RuntimeError(f"No clear label position: {ax.get_title()}/{row['city']}")


def main(input_path=None, output_dir=None):
    with (Path(input_path) if input_path is not None else table_path('results_summary/master_results.csv')).open(
            encoding='utf-8-sig', newline='') as stream:
        data = list(csv.DictReader(stream))
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
        fig.subplots_adjust(left=.075, right=.985, top=.94, bottom=.13,
                            wspace=.35, hspace=.36)
        groups = []
        for ax, algorithm in zip(axes.flat, ALGORITHMS):
            rows = [r for r in data if r['algorithm'] == algorithm]
            if len(rows) != 7:
                raise ValueError(f'{algorithm}: expected seven cities')
            groups.append(rows)
            ax.scatter([float(r['Q']) for r in rows],
                       [float(r['ACD']) for r in rows],
                       s=32, color='#2b6cb0', edgecolors='white', linewidths=.6, zorder=3)
            ax.set_title(algorithm, weight='bold', pad=10)
            ax.set_xlabel('Q')
            ax.set_ylabel('ACD')
            ax.margins(x=.17, y=.14)
            ax.grid(color='#d9d9d9', linewidth=.65, alpha=.65)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(top=False, right=False, labelsize=9)
        fig.text(.5, .035,
                 'BJ: Beijing; SH: Shanghai; HK: Hong Kong; MS: Moscow; '
                 'SP: St Petersburg; KZ: Kazan; YE: Yekaterinburg',
                 ha='center', fontsize=8)
        fig.canvas.draw()
        for ax, rows in zip(axes.flat, groups):
            place_labels(fig, ax, rows)
        output = Path(output_dir) if output_dir is not None else ROOT / 'figures'
        output.mkdir(parents=True, exist_ok=True)
        pdf = output / 'within_method_scatter.pdf'
        fig.savefig(pdf)
        fig.savefig(output / 'within_method_scatter.png', dpi=180)
        plt.close(fig)
    print('Updated 6 panels / 42 points: left and bottom axes only; '
          'all city labels inside axes, with no label/marker overlaps.')
    print(pdf)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    main(args.input, args.output_dir)
