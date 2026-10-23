"""Pairwise-complete descriptive associations of archived Endpoint diagnostics.

No missing value is imputed. Adjustments are refitted on each complete pair.
Run independently to avoid repeating unrelated POI or model-fitting analyses.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

METRICS = ['Q', 'ACT', 'ACD', 'ACBC', 'CGCC', 'PL']
from paths import ANALYSIS_ROOT as ROOT, table_path


def compute_associations(master):
    records = []
    scopes = [('pooled', master), ('without_DCSBM', master[master.algorithm != 'DCSBM'])]
    scopes += list(master.groupby('algorithm'))
    for a, b in itertools.combinations(METRICS, 2):
        for scope, group in scopes:
            complete = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[a, b])
            records.append(dict(scope=scope, metric_1=a, metric_2=b, n=len(complete),
                                pearson=complete[a].corr(complete[b]),
                                spearman=complete[a].corr(complete[b], method='spearman')))
        complete = master.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[a, b, 'city', 'algorithm', 'community_count'])
        assert (complete.community_count > 0).all()
        dummy = pd.get_dummies(complete[['city', 'algorithm']], drop_first=True, dtype=float)
        design = np.column_stack([np.ones(len(complete)), np.log(complete.community_count), dummy])
        values = complete[[a, b]].to_numpy(dtype=float)
        residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        records.append(dict(scope='residual_city_algorithm_logK', metric_1=a, metric_2=b,
                            n=len(complete), pearson=np.corrcoef(residual.T)[0, 1],
                            spearman=pd.Series(residual[:, 0]).corr(pd.Series(residual[:, 1]), method='spearman')))
    result = pd.DataFrame(records)
    assert np.isfinite(result[['pearson', 'spearman']].to_numpy()).all()
    return result


def main(input_path=None, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(input_path if input_path is not None else table_path('results_summary/master_results.csv'))
    assert len(master) == 42
    result = compute_associations(master)
    result.to_csv(output_dir / 'metric_associations.csv', index=False)
    missing = master.loc[master[METRICS].isna().any(axis=1), ['city', 'algorithm', 'community_count'] + METRICS].copy()
    missing['reason'] = 'Archived ACBC and CGCC reporting convention: fewer than three community-graph vertices; not imputed as zero.'
    missing.to_csv(output_dir / 'metric_missingness_audit.csv', index=False)
    print(result.query("scope == 'residual_city_algorithm_logK'").to_string(index=False))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    main(args.input, args.output_dir)
