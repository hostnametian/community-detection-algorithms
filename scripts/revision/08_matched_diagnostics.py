"""Select by K only, then evaluate selected network and coordinate partitions."""
from common import *
from diagnostics import spatial
from models import coordinate_kmeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import argparse,time

ALGORITHMS=['Infomap','LeidenRBC','DirLou','GeoLP','Spinglass','CoordinateKMeans']
TARGETS=[5,10,15,20]

def evaluate(g,p,path):
    report=ROOT/'results/selected'/f'{path.stem}.json'
    if report.exists():
        saved=json.loads(report.read_text())
        if saved['partition_sha256']!=digest(path) or saved['diagnostics_sha256']!=digest(SCRIPT_ROOT/'diagnostics.py'):
            raise ValueError('Selected diagnostic cache mismatch')
        return saved['metrics']
    start=time.monotonic();m=legacy.compute_standard_metrics(g,p,sample_efficiency=False)
    t,detail=topology_stats(g,p);s,geo=spatial(g,p)
    metrics={**m,**t,**s}
    metrics={key:(None if isinstance(v,float) and not np.isfinite(v) else v) for key,v in metrics.items()}
    write_json(report,{'partition_sha256':digest(path),'graph_fingerprint':fingerprint(g),
                       'diagnostics_sha256':digest(SCRIPT_ROOT/'diagnostics.py'),
                       'exact_efficiency_native_threads':legacy.nk.getMaxNumberOfThreads() if legacy.NETWORKIT_OK else 1,
                       'metrics':metrics,'seconds':time.monotonic()-start})
    detail.merge(geo,on=['community','nodes']).to_csv(report.with_suffix('.csv'),index=False)
    return metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',action='append');ap.add_argument('--dcsbm-restarts',type=int,default=5)
    ap.add_argument('--metric-threads',type=int,default=4)
    args=ap.parse_args()
    if legacy.NETWORKIT_OK:legacy.nk.setNumberOfThreads(args.metric_threads)
    for cfg in legacy.CITIES:
        city=cfg['city']
        if args.city and city not in args.city:continue
        lock=acquire_lock(f'matched_diagnostics__{city}')
        g=load_graph(city); selections=[];summaries=[]
        for algorithm in ALGORITHMS:
            f=pd.read_csv(ROOT/'results'/f'candidates__{city}__Endpoint__{algorithm}.csv')
            expected={'Infomap':10,'LeidenRBC':11,'DirLou':11,'GeoLP':4,'Spinglass':4,'CoordinateKMeans':4}[algorithm]
            if len(f)!=expected or f.parameter_value.nunique()!=expected:
                raise RuntimeError(f'{city} {algorithm}: incomplete candidate grid')
            for k in TARGETS:
                tolerance=max(1,round(.1*k));eligible=f.assign(K_distance=(f.K-k).abs())
                eligible=eligible[eligible.K_distance<=tolerance].sort_values(['K_distance','parameter','parameter_value'])
                if eligible.empty:
                    selections.append({'city':city,'algorithm':algorithm,'target_K':k,'matched':False,
                                       'reason':'no candidate in declared tolerance'})
                else:
                    selected=eligible.iloc[0]
                    selections.append({'city':city,'algorithm':algorithm,'target_K':k,'matched':True,
                                       'actual_K':int(selected.K),'partition':selected.partition,
                                       'parameter':selected.parameter,'parameter_value':selected.parameter_value})
        df=pd.read_csv(ROOT/'results'/f'fixed_k_dcsbm__{city}.csv')
        for k in TARGETS:
            group=df[df.target_K==k]
            if len(group)!=args.dcsbm_restarts or group.seed.nunique()!=args.dcsbm_restarts:
                raise RuntimeError(f'{city} K{k}: incomplete DCSBM restarts')
            selected=group.sort_values(['description_length','seed']).iloc[0]
            selections.append({'city':city,'algorithm':'DCSBM-count-fixedK','target_K':k,'matched':True,
                               'actual_K':int(selected.K),'partition':selected.partition,
                               'parameter':'fixed_K','parameter_value':k})
        pd.DataFrame(selections).to_csv(ROOT/'results'/f'matched_selection__{city}.csv',index=False)
        cache={}
        for selection in selections:
            if not selection['matched']:continue
            path=Path(selection['partition']);p,meta=load_partition(path,g)
            metrics=evaluate(g,p,path)
            # Additional exact-actual-K geometry control, since target matching allows tolerance.
            k=len(set(p.values()))
            if k not in cache:
                gp,gmeta=coordinate_kmeans(g,k)
                gpath=ROOT/'partitions/coordinate_actual_K'/f'{city}__CoordinateKMeans__K{k}.npz'
                if gpath.exists():gp,gmeta=load_partition(gpath,g)
                else:save_partition(gpath,g,gp,gmeta)
                gm=evaluate(g,gp,gpath);cache[k]=(gp,gm)
            gp,gm=cache[k];nodes=sorted(g)
            comparison={'NMI_coordinate_same_actual_K':normalized_mutual_info_score([p[n] for n in nodes],[gp[n] for n in nodes]),
                        'ARI_coordinate_same_actual_K':adjusted_rand_score([p[n] for n in nodes],[gp[n] for n in nodes])}
            for key in ['fragmentation_macro','diameter_m_macro','mean_pairwise_m_macro','Q','ACT','ACD','PL']:
                comparison['coordinate_same_actual_K_'+key]=gm[key]
                comparison['delta_vs_coordinate_'+key]=metrics[key]-gm[key] if metrics[key] is not None and gm[key] is not None else None
            summaries.append({**selection,**metrics,**comparison})
            pd.DataFrame(summaries).to_csv(ROOT/'results'/f'matched_diagnostics__{city}.csv',index=False)
            print(city,selection['algorithm'],'target',selection['target_K'],'actual',k,'diagnostics saved',flush=True)
        legacy._METRIC_GRAPH_CACHE.clear()
        lock.close()

if __name__=='__main__':main()
