"""Inventory historical provenance without rewriting historical manifests."""
from common import *
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-root',type=Path,action='append',help='Optional additional historical output directory')
    args=ap.parse_args()
    sources={f'baseline_{i}':p for i,p in enumerate(args.baseline_root or [BASELINE_ROOT])}
    manifests, records = {}, []
    for label, root in sources.items():
        p=root/'01_data_and_graphs/run_manifest.json'
        if p.is_file():
            manifests[label]={'path':str(p),'sha256':digest(p),'contents':json.loads(p.read_text())}
        for config in sorted(root.glob('results/*/*/*/config.json')):
            d=json.loads(config.read_text()); run=config.parent
            row={'result_set':label,'city':d.get('city'),'algorithm':d.get('algorithm'),
                 'run':str(run),'script_version':d.get('script_version'),
                 'graph_nodes':d.get('graph_nodes'),'graph_edges':d.get('graph_edges'),
                 'parameters':json.dumps(d.get('algorithm_parameters',{}),sort_keys=True),
                 'config_sha256':digest(config)}
            for name in ['metrics.json','communities.csv']:
                if (run/name).exists():row[name+'_sha256']=digest(run/name)
            records.append(row)
    pd.DataFrame(records).to_csv(ROOT/'audit/historical_runs.csv',index=False)
    write_json(ROOT/'audit/historical_manifests.json',manifests)
    files=[{'path':p.name,'sha256':digest(p)} for p in sorted(SOURCE_ROOT.glob('*.py'))]
    write_json(ROOT/'audit/source_snapshot.json',{'files':files,'source':'repository scripts'})
    write_json(ROOT/'audit/environment.json',environment())
    inputs=[]
    for c in legacy.CITIES:
        for field in ['json_file','poi_file']:
            p=Path(c[field]);inputs.append({'city':c['city'],'kind':field,'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
    write_json(ROOT/'audit/input_hashes.json',inputs)
    print('Inventoried',len(records),'historical runs and',len(inputs),'raw input files',flush=True)

if __name__=='__main__': main()
