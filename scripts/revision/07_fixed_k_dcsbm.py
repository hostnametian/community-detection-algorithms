"""Exact-K count-DCSBM supplement, distinct from free-MDL model selection."""
from common import *
from models import dcsbm
from diagnostics import quick
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',action='append');ap.add_argument('--restarts',type=int,default=5)
    ap.add_argument('--targets',type=int,nargs='+',default=[5,10,15,20]);args=ap.parse_args()
    if not json.loads((ROOT/'audit/replay_status.json').read_text())['passed']:raise RuntimeError('Audit not passed')
    for cfg in legacy.CITIES:
        city=cfg['city']
        if args.city and city not in args.city:continue
        lock=acquire_lock(f'fixed_k_dcsbm__{city}')
        g=load_graph(city);rows=[]
        for k in args.targets:
            for i in range(args.restarts):
                seed=42+17*i;path=ROOT/'partitions/fixed_k_dcsbm'/f'{city}__K{k}__seed{seed}.npz'
                start=time.monotonic()
                if path.exists():p,meta=load_partition(path,g)
                else:
                    p,meta=dcsbm(g,'discrete-geometric',seed,k)
                    meta.update({'seconds':time.monotonic()-start,'models_sha256':digest(SCRIPT_ROOT/'models.py'),
                                 'stage':'conditional on K, not free MDL K inference','restarts_requested':args.restarts})
                    save_partition(path,g,p,meta)
                rows.append({'city':city,'algorithm':'DCSBM-count-fixedK','target_K':k,'seed':seed,
                             'description_length':meta['description_length'],**quick(g,p),
                             'seconds':meta['seconds'],'partition':str(path)})
                pd.DataFrame(rows).to_csv(ROOT/'results'/f'fixed_k_dcsbm__{city}.csv',index=False)
                print(city,'fixed K',k,'seed',seed,'DL',round(meta['description_length'],2),'seconds',round(meta['seconds'],1),flush=True)
        lock.close()

if __name__=='__main__':main()
