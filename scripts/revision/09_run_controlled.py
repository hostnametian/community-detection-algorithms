"""Run independent city jobs with bounded process concurrency and explicit status."""
from pathlib import Path
import argparse,concurrent.futures,subprocess,sys,json,os,time
from paths import ROOT, SCRIPT_ROOT
for folder in ('logs', 'results'):
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
CITIES=['Kazan','Yekaterinburg','StPetersburg','BeiJing','HongKong','Moscow','ShangHai']

def run(city,phase):
    commands=[]
    if phase=='matched':
        commands=[('07_fixed_k_dcsbm.py',['--city',city]),
                  ('06_candidates.py',['--city',city]),
                  ('08_matched_diagnostics.py',['--city',city])]
    else:
        for rep in ['L','P']:
            commands.append(('06_candidates.py',['--city',city,'--representation',rep,'--algorithm','LeidenRBC']))
            if city in ['Kazan','Yekaterinburg']:
                commands.append(('06_candidates.py',['--city',city,'--representation',rep,'--algorithm','DirLou']))
    log=ROOT/'logs'/f'controlled__{phase}__{city}.log'
    start=time.monotonic()
    with log.open('a') as f:
        for script,args in commands:
            f.write('\nCOMMAND '+repr([script,*args])+'\n');f.flush()
            result=subprocess.run([sys.executable,'-u',str(SCRIPT_ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                return {'city':city,'phase':phase,'passed':False,'script':script,'returncode':result.returncode,'log':str(log)}
    return {'city':city,'phase':phase,'passed':True,'seconds':time.monotonic()-start,'log':str(log)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['matched','resolution'],default='matched')
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--city',action='append');args=ap.parse_args()
    envfile=ROOT/'audit/replay_status.json'
    assert json.loads(envfile.read_text())['passed']
    statuses=[]; cities=args.city or CITIES
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,c,args.phase) for c in cities]
        for future in concurrent.futures.as_completed(futures):
            status=future.result();statuses.append(status)
            out=ROOT/'results'/f'controlled_status__{args.phase}.json'
            temp=out.with_suffix('.tmp');temp.write_text(json.dumps(statuses,indent=2));temp.replace(out)
            print(json.dumps(status),flush=True)
    if not all(s['passed'] for s in statuses):raise SystemExit(1)

if __name__=='__main__':main()
