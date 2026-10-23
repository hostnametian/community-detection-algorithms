"""Aggregate completed fits; report missing conditions explicitly."""
from common import *
import argparse
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def combine(pattern):
    paths=sorted((ROOT/'results').glob(pattern))
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True) if paths else pd.DataFrame()

def table(frame,columns):
    if frame.empty:return '(No completed records yet.)'
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        vals=[]
        for col in columns:
            v=r[col]; vals.append(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    (ROOT/'figures').mkdir(exist_ok=True)
    models=combine('model_comparison__*.csv');matched=combine('matched_diagnostics__*.csv')
    coverage=combine('matched_selection__*.csv');candidates=combine('candidates__*.csv')
    fixed=combine('fixed_k_dcsbm__*.csv')
    status={'model_fits':len(models),'expected_model_fits':105,'fixed_K_fits':len(fixed),'expected_fixed_K_fits':140,
            'selection_conditions':len(coverage),'expected_selection_conditions':196,
            'evaluated_matched_conditions':len(matched)}
    if not coverage.empty:status['unmatched_conditions']=int((coverage.matched==False).sum())
    for name,f in [('all_model_fits',models),('all_fixed_K_fits',fixed),('all_matched_diagnostics',matched),
                   ('all_matched_selection',coverage),('all_parameter_candidates',candidates)]:
        if not f.empty:f.to_csv(ROOT/'results'/f'{name}.csv',index=False)
    best=pd.DataFrame();comparisons=pd.DataFrame()
    if not models.empty:
        best=models.sort_values(['description_length','seed']).groupby(['city','family'],as_index=False).first()
        counts=models.groupby(['city','family']).size().rename('n_fits_available').reset_index()
        best=best.merge(counts,on=['city','family'])
        best.to_csv(ROOT/'results/model_family_best.csv',index=False)
        pivot=best[best.family!='real-normal'].pivot(index='city',columns='family',values='description_length')
        if {'discrete-geometric','discrete-poisson'}<=set(pivot.columns):
            comparisons=pivot.reset_index();comparisons['DL_poisson_minus_geometric']=comparisons['discrete-poisson']-comparisons['discrete-geometric']
            comparisons.to_csv(ROOT/'results/discrete_model_DL_contrast.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for family,group in best.groupby('family'):
            for ax,key in zip(axes,['K','fragmentation_macro']):
                ax.plot(group.city,group[key],marker='o',label=family)
                ax.tick_params(axis='x',labelrotation=45);ax.set_ylabel(key)
        axes[0].legend(fontsize=7);fig.suptitle('Best available fit (up to five per family); no common quality ranking')
        fig.savefig(ROOT/'figures/model_family_diagnostics.pdf');plt.close(fig)
    summary=pd.DataFrame()
    if not matched.empty:
        fields=['delta_vs_coordinate_fragmentation_macro','delta_vs_coordinate_mean_pairwise_m_macro',
                'delta_vs_coordinate_Q','delta_vs_coordinate_ACD']
        summary=matched.groupby('algorithm')[fields].median().reset_index()
        summary.to_csv(ROOT/'results/matched_coordinate_median_contrasts.csv',index=False)
        algos=[a for a in matched.algorithm.unique() if a!='CoordinateKMeans']
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,field,label in zip(axes,fields[:2],['Fragmentation difference','Mean pairwise distance difference (m)']):
            values=[matched.loc[matched.algorithm==a,field].dropna().values for a in algos]
            ax.boxplot(values,tick_labels=algos,showfliers=True);ax.axhline(0,color='0.4',linestyle='--')
            ax.tick_params(axis='x',labelrotation=40);ax.set_ylabel(label)
        fig.suptitle('Network partition minus coordinate clustering at identical actual K\nDescriptive city-target contrasts; not independent replicates')
        fig.savefig(ROOT/'figures/matched_coordinate_contrasts.pdf');plt.close(fig)
    resolution=[]
    if not candidates.empty:
        for city in [c['city'] for c in legacy.CITIES]:
            for algorithm in ['LeidenRBC','DirLou']:
                f=candidates[(candidates.city==city)&(candidates.algorithm==algorithm)]
                l=f[f.representation=='L'];p=f[f.representation=='P']
                if l.empty or p.empty:continue
                gl=load_graph(city,'L');gp=load_graph(city,'P')
                for _,left in l.iterrows():
                    right=p[p.parameter_value==left.parameter_value]
                    if right.empty:continue
                    right=right.iloc[0]
                    lp,_=load_partition(left.partition,gl);pp,_=load_partition(right.partition,gp)
                    support=sorted(set(lp)&set(pp));a=[lp[n] for n in support];b=[pp[n] for n in support]
                    resolution.append({'city':city,'algorithm':algorithm,'resolution':left.parameter_value,
                                       'K_L':int(left.K),'K_P':int(right.K),'NMI':normalized_mutual_info_score(a,b),
                                       'ARI':adjusted_rand_score(a,b),'common_nodes':len(support),
                                       'degenerate':len(set(a))==1 or len(set(b))==1})
        pd.DataFrame(resolution).to_csv(ROOT/'results/LP_resolution_agreement.csv',index=False)
        cities=[c['city'] for c in legacy.CITIES]
        fig,axes=plt.subplots(4,2,figsize=(11,12),layout='constrained')
        for ax,city in zip(axes.flat,cities):
            f=candidates[(candidates.city==city)&(candidates.algorithm=='LeidenRBC')]
            for rep,group in f.groupby('representation'):
                group=group.sort_values('parameter_value');ax.plot(group.parameter_value,group.K,marker='.',label=rep)
            ax.set_xscale('log');ax.set_title(city);ax.set_xlabel('Resolution');ax.set_ylabel('K');ax.legend()
        axes.flat[-1].set_visible(False);fig.savefig(ROOT/'figures/leiden_resolution_counts.pdf');plt.close(fig)
    status['LP_resolution_conditions']=len(resolution);status['expected_LP_resolution_conditions']=99
    status['complete_counts']=(len(models)==105 and len(fixed)==140 and len(coverage)==196 and
                               len(matched)==int(coverage.matched.sum()) and len(resolution)==99)
    write_json(ROOT/'results/experiment_status.json',status)
    hashes={str(p.relative_to(SCRIPT_ROOT)):digest(p) for p in sorted(SCRIPT_ROOT.glob('*.py'))}
    write_json(ROOT/'audit/revision_code_hashes.json',hashes)
    print(json.dumps(status,ensure_ascii=False),flush=True)
    if not args.partial and not status['complete_counts']:raise SystemExit('Incomplete experimental design')

if __name__=='__main__':main()
