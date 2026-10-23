"""Re-evaluate saved manuscript assignments on fingerprint-verified graphs."""
from common import *
import math,time

def main():
    master=pd.read_csv(table_path('results_summary/master_results.csv'))
    spatial=pd.read_csv(table_path('results_summary/spatial_compactness_summary.csv')).set_index(['city','algorithm'])
    rows=[];differences=[]
    for city,group in master.groupby('city',sort=True):
        g=load_graph(city)
        for _,r in group.iterrows():
            start=time.monotonic();run=run_path(r.run_dir)
            frame=pd.read_csv(run/'communities.csv')
            partition=dict(zip(frame.node_id.astype(int),frame.community_id.astype(int)))
            validate_partition(g,partition)
            computed=legacy.compute_standard_metrics(g,partition,sample_efficiency=False)
            topo,_=topology_stats(g,partition)
            for key in ['community_count','Q','Q_directed','ACT','ACD','ACBC','CGCC','PL']:
                expected=float(r[key]);actual=float(computed[key])
                if not ((math.isnan(expected) and math.isnan(actual)) or math.isclose(expected,actual,abs_tol=1e-8,rel_tol=1e-8)):
                    differences.append({'city':city,'algorithm':r.algorithm,'metric':key,'old':expected,'recomputed':actual})
            sf=float(spatial.loc[(city,r.algorithm),'avg_fragmentation_index'])
            if not math.isclose(topo['fragmentation_macro'],sf,abs_tol=1e-10):
                differences.append({'city':city,'algorithm':r.algorithm,'metric':'fragmentation','old':sf,'recomputed':topo['fragmentation_macro']})
            save_partition(ROOT/'partitions/legacy'/f'{city}__{r.algorithm}.npz',g,partition,
                           {'origin':'saved manuscript assignment; algorithms not rerun',
                            'historical_run':str(run),'historical_config':json.loads((run/'config.json').read_text()),
                            'historical_partition_sha256':digest(run/'communities.csv')})
            rows.append({'city':city,'algorithm':r.algorithm,**computed,**topo,'seconds':time.monotonic()-start})
            pd.DataFrame(rows).to_csv(ROOT/'audit/replayed_metrics.csv',index=False)
            write_json(ROOT/'audit/replay_discrepancies.json',differences)
            print(city,r.algorithm,'replayed',round(time.monotonic()-start,2),'seconds; discrepancies',len(differences),flush=True)
        legacy._METRIC_GRAPH_CACHE.clear()
    write_json(ROOT/'audit/replay_status.json',{'partitions':len(rows),'discrepancies':len(differences),'passed':not differences})
    if differences:raise RuntimeError('Historical metrics disagree: inspect before downstream claims')

if __name__=='__main__':main()
