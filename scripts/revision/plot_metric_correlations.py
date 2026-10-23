"""Draw the corrected association figure from the pairwise-complete CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metric_associations import METRICS
from paths import ANALYSIS_ROOT


def main(input_path=None, output_dir=None):
    root = ANALYSIS_ROOT
    data = pd.read_csv(input_path if input_path is not None else root / 'metric_associations.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), layout='constrained')
    for ax, scope, title in zip(axes, ['pooled', 'residual_city_algorithm_logK'],
                                ['Pooled', 'Residuals: city, method and log K']):
        matrix = np.eye(len(METRICS))
        rows = data[data.scope == scope]
        assert len(rows) == 15
        for row in rows.itertuples():
            i, j = METRICS.index(row.metric_1), METRICS.index(row.metric_2)
            matrix[i, j] = matrix[j, i] = row.pearson
        assert np.isfinite(matrix).all()
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(6), METRICS, rotation=45)
        ax.set_yticks(range(6), METRICS)
        ax.set_title(title + '\nPairwise complete observations: n = 40 or 42', fontsize=10)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > .6 else 'black')
    fig.colorbar(im, ax=axes, shrink=.8, label='Pearson correlation')
    target = Path(output_dir) if output_dir is not None else root / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'correlations.pdf')
    fig.savefig(target / 'correlations.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    main(args.input, args.output_dir)
