"""Rebuild frozen graphs offline, validate against the manuscript inventory."""
from common import *
import argparse, time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',action='append');args=ap.parse_args()
    historic=pd.read_csv(table_path('01_data_and_graphs/graph_inventory.csv')).set_index('city')
    for cfg in legacy.CITIES:
        city=cfg['city']
        if args.city and city not in args.city:continue
        manifest=ROOT/'graphs'/f'{city}.json'
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['source_sha256']!=digest(SOURCE_ROOT/'analysis_pipeline.py'):raise ValueError('Source changed')
            print('Already built',city,flush=True);continue
        cache=BOUNDARY_ROOT/f"{cfg['city_query'].replace(' ','_')}_{cfg['country']}_boundary.geojson"
        if not cache.exists():raise FileNotFoundError('Offline boundary required: '+str(cache))
        start=time.monotonic()
        builder=legacy.PTNGraphBuilder(cfg['json_file'],city,cfg['country'],100,city_query=cfg['city_query'])
        endpoint=builder.build()
        if builder.city_boundary is None:raise RuntimeError('Boundary failure must not silently change graph extent')
        graphs={'Endpoint':endpoint,'L':builder.build_stop_graph(),'P':builder.build_p_space_graph()}
        records={}
        for name,g in graphs.items():
            if not all(np.isfinite([d.get('lat',np.nan),d.get('lon',np.nan)]).all() for n,d in g.nodes(data=True)):
                raise ValueError('Missing coordinates')
            weights=np.array([d['weight'] for u,v,d in g.edges(data=True)])
            if not np.all((weights>=1)&(weights==weights.astype(np.int64))):raise ValueError('Not positive route counts')
            path=save_graph(g,city,name)
            records[name]={'nodes':len(g),'edges':g.number_of_edges(),'fingerprint':fingerprint(g),
                          'legacy_fingerprint':legacy.graph_fingerprint(g),'file_sha256':digest(path),
                          'self_loops':nx.number_of_selfloops(g),'coordinate_coverage':1.0,
                          'weight_mean':float(weights.mean()),'weight_variance':float(weights.var()),
                          'weight_max':float(weights.max()),'weight_quantiles':np.quantile(weights,[.5,.9,.99]).tolist()}
        matches=records['Endpoint']['legacy_fingerprint']==historic.loc[city,'graph_fingerprint']
        meta={'city':city,'source_sha256':digest(SOURCE_ROOT/'analysis_pipeline.py'),
              'input_sha256':digest(cfg['json_file']),'boundary_sha256':digest(cache),
              'settings':{'boundary':'frozen administrative polygon; simplify 0.001 degrees; all-way-nodes-inside',
                          'stop_threshold_m':100,'direction':'route_relation_member_order','component':'GWCC'},
              'graphs':records,'matches_manuscript_endpoint_fingerprint':bool(matches),
              'route_quality':builder.stats,'elapsed_seconds':time.monotonic()-start}
        write_json(manifest,meta)
        print(city,'Endpoint fingerprint matches manuscript:',matches, 'seconds:',round(meta['elapsed_seconds'],1),flush=True)
        if not matches:raise RuntimeError('Graph mismatch: stop before downstream experiments')
    print('Graph build stage completed',flush=True)

if __name__=='__main__':main()
