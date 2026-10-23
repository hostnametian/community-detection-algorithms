"""Compute a slow, not-yet-started spin setting in a separate staging directory.

The result is never silently installed over an active candidate; integration is
an explicit subsequent operation after inspecting the primary grid's progress.
"""
from common import *
from models import spinglass
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',required=True);ap.add_argument('--spins',required=True,type=int)
    args=ap.parse_args();g=load_graph(args.city);start=time.monotonic()
    p,details=spinglass(g,spins=args.spins,seed=42)
    meta={'algorithm':'Spinglass','parameter':'spins','parameter_value':args.spins,'settings':details,
          'seconds':time.monotonic()-start,'representation':'Endpoint','models_sha256':digest(SCRIPT_ROOT/'models.py'),
          'execution':'staged parallel computation of the same declared seed and setting'}
    path=ROOT/'partitions/prefit'/f'{args.city}__Endpoint__Spinglass__spins_{args.spins}.npz'
    save_partition(path,g,p,meta)
    print(args.city,args.spins,'completed staging in',meta['seconds'],'seconds',flush=True)

if __name__=='__main__':main()
