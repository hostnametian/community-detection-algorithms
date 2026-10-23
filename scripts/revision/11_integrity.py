"""Final integrity audit of the completed experiment, without refitting models."""
from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    metadata=[];issues=[];notes=[]
    for path in sorted((ROOT/'partitions').rglob('*.npz')):
        meta_path=path.with_suffix('.json')
        if not meta_path.exists():issues.append({'file':str(path),'issue':'metadata missing'});continue
        m=json.loads(meta_path.read_text())
        if digest(path)!=m['partition_sha256']:issues.append({'file':str(path),'issue':'partition hash mismatch'})
        with np.load(path,allow_pickle=False) as a:
            if len(a['nodes'])!=len(a['labels']) or len(set(a['nodes']))!=len(a['nodes']) or np.any(a['labels']==-1):
                issues.append({'file':str(path),'issue':'invalid assignment arrays'})
        settings=m.get('settings',{}).get('algorithm_parameters',{})
        if settings.get('objective_monotonic_non_decreasing') is False:
            issues.append({'file':str(path),'issue':'DirLou objective trace decreased'})
        if settings.get('converged') is False:
            notes.append({'file':str(path),'issue':'GeoLP reached iteration cap; report instead of hiding'})
        metadata.append({'file':str(path.relative_to(ROOT)),'sha256':digest(path),'graph_fingerprint':m['graph_fingerprint']})
    for graph_meta in sorted((ROOT/'graphs').glob('*.json')):
        m=json.loads(graph_meta.read_text());city=m['city']
        for rep,d in m['graphs'].items():
            if digest(ROOT/'graphs'/f'{city}__{rep}.npz')!=d['file_sha256']:
                issues.append({'city':city,'representation':rep,'issue':'graph file hash mismatch'})
    snapshot=json.loads((ROOT/'audit/source_snapshot.json').read_text())
    for record in snapshot['files']:
        if digest(SOURCE_ROOT/record['path'])!=record['sha256']:
            issues.append({'file':record['path'],'issue':'historical source snapshot modified'})
    write_json(ROOT/'audit/partition_inventory.json',metadata)
    write_json(ROOT/'audit/final_integrity.json',{'partitions_checked':len(metadata),'issues':issues,'notes':notes,
               'passed':not issues,'scope':'files present at audit time; count completeness checked separately'})
    print('Partitions',len(metadata),'integrity failures',len(issues),'diagnostic notes',len(notes),flush=True)
    if issues and not args.partial:raise SystemExit(1)

if __name__=='__main__':main()
