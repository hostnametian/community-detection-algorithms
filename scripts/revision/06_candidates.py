"""Predeclared Endpoint candidates and L/P resolution controls, saved per fit."""
from common import *
from models import infomap,spinglass,coordinate_kmeans,dcsbm
from diagnostics import quick
import argparse,time

RESOLUTIONS=(.001,.0025,.005,.01,.025,.05,.1,.2,.5,1.,2.)
MARKOV_TIMES=(.05,.1,.2,.35,.5,.75,1.,1.5,2.,4.)
TARGETS=(5,10,15,20)

def candidates(algorithm):
    if algorithm in ('LeidenRBC','DirLou'):
        return [('resolution',x,{'resolution':x}) for x in RESOLUTIONS]
    if algorithm=='Infomap':return [('markov_time',x,{'markov_time':x}) for x in MARKOV_TIMES]
    if algorithm=='GeoLP':return [('initial_communities',x,{'initial_communities':x}) for x in TARGETS]
    if algorithm=='Spinglass':return [('spins',x,{'spins':x}) for x in TARGETS]
    if algorithm=='CoordinateKMeans':return [('K',x,{'k':x}) for x in TARGETS]
    raise ValueError(algorithm)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',action='append');ap.add_argument('--algorithm',action='append')
    ap.add_argument('--representation',choices=['Endpoint','L','P'],default='Endpoint');args=ap.parse_args()
    if not json.loads((ROOT/'audit/replay_status.json').read_text())['passed']:raise RuntimeError('Audit not passed')
    algorithms=args.algorithm or ['LeidenRBC','DirLou','Infomap','GeoLP','CoordinateKMeans','Spinglass']
    runners={'LeidenRBC':legacy.run_leiden_rbc,'DirLou':legacy.run_dirlou,'Infomap':infomap,
             'GeoLP':legacy.run_geolp,'Spinglass':spinglass,'CoordinateKMeans':coordinate_kmeans}
    for cfg in legacy.CITIES:
        city=cfg['city']
        if args.city and city not in args.city:continue
        g=load_graph(city,args.representation)
        for algorithm in algorithms:
            lock=acquire_lock(f'candidates__{city}__{args.representation}__{algorithm}')
            rows=[]
            for parameter,value,kwargs in candidates(algorithm):
                path=ROOT/'partitions/candidates'/f'{city}__{args.representation}__{algorithm}__{parameter}_{value}.npz'
                start=time.monotonic()
                if path.exists():p,meta=load_partition(path,g)
                else:
                    p,details=runners[algorithm](g,**kwargs)
                    meta={'algorithm':algorithm,'parameter':parameter,'parameter_value':value,'settings':details,
                          'seconds':time.monotonic()-start,'representation':args.representation,
                          'models_sha256':digest(SCRIPT_ROOT/'models.py')}
                    save_partition(path,g,p,meta)
                row={'city':city,'representation':args.representation,'algorithm':algorithm,
                     'parameter':parameter,'parameter_value':value,**quick(g,p),'seconds':meta['seconds'],'partition':str(path)}
                rows.append(row)
                out=ROOT/'results'/f'candidates__{city}__{args.representation}__{algorithm}.csv'
                pd.DataFrame(rows).to_csv(out,index=False)
                print(city,args.representation,algorithm,parameter,value,'K',row['K'],'seconds',round(meta['seconds'],1),flush=True)
            lock.close()

if __name__=='__main__':main()
