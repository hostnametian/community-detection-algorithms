"""Paired-restart weight-family comparison on the frozen Endpoint graphs."""
from common import *
from models import dcsbm
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',action='append');ap.add_argument('--restarts',type=int,default=5)
    args=ap.parse_args()
    if not json.loads((ROOT/'audit/replay_status.json').read_text())['passed']:raise RuntimeError('Replay audit not passed')
    rows=[]
    for cfg in legacy.CITIES:
        city=cfg['city']
        if args.city and city not in args.city:continue
        g=load_graph(city)
        for family in ['discrete-geometric','discrete-poisson','real-normal']:
            for i in range(args.restarts):
                seed=42+17*i;path=ROOT/'partitions/model_comparison'/f'{city}__{family}__{seed}.npz'
                start=time.monotonic()
                if path.exists():p,meta=load_partition(path,g)
                else:
                    p,meta=dcsbm(g,family,seed)
                    meta.update({'stage':'paired weight-family diagnostic','restarts_requested':args.restarts,'seconds':time.monotonic()-start})
                    save_partition(path,g,p,meta)
                topo,_=topology_stats(g,p)
                rows.append({'city':city,'family':family,'seed':seed,**topo,
                             'description_length':meta['description_length'],'seconds':meta['seconds'],'partition':str(path)})
                pd.DataFrame(rows).to_csv(ROOT/'results'/f'model_comparison__{city}.csv',index=False)
                print(city,family,seed,'K',topo['K'],'DL',round(meta['description_length'],2),'seconds',round(meta['seconds'],1),flush=True)
        rows=[]

if __name__=='__main__':main()
