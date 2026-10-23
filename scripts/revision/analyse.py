"""Read-only analyses of archived fits and source POI tags for Reviewer 1."""
from pathlib import Path
import os,sys,json,itertools,hashlib
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from paths import ROOT, ANALYSIS_ROOT, table_path
R=ANALYSIS_ROOT; R.mkdir(parents=True,exist_ok=True)
from common import legacy
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.neighbors import BallTree

def save(df,name):df.to_csv(R/(name+'.csv'),index=False)
def native(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x))
master=pd.read_csv(table_path('results_summary/master_results.csv')); assert len(master)==42
reps=pd.read_csv(table_path('representation_sensitivity_all_algorithms_output/representation_pairwise_summary.csv'))
reps=reps[reps.transfer_threshold_m.isna()].copy();assert len(reps)==126
reps['degenerate']=(reps.left_community_count==1)|(reps.right_community_count==1)
save(reps,'representation_comparisons_flagged')
inventory=[]
for _,r in reps.iterrows():
 a,b=r.comparison.split('__')
 for side,rep in [('left',a),('right',b)]:inventory.append(dict(city=r.city,algorithm=r.algorithm,representation=rep,nodes=r[side+'_nodes'],edges=r[side+'_edges'],K=r[side+'_community_count'],degenerate=r[side+'_community_count']==1))
inv=pd.DataFrame(inventory).drop_duplicates();assert len(inv)==126
save(inv.sort_values(['city','representation','algorithm']),'complete_126_inventory')
rs=[]
for (city,comparison),g in reps.groupby(['city','comparison']):
 good=g[~g.degenerate];rs.append(dict(city=city,comparison=comparison,n_all=len(g),n_nondegenerate=len(good),mean_all=g.nmi.mean(),mean_nondegenerate=good.nmi.mean(),min_nondegenerate=good.nmi.min(),max_nondegenerate=good.nmi.max()))
save(pd.DataFrame(rs),'nmi_sensitivity')
# Rank/linear association is descriptive; no independence-based p values.
from metric_associations import compute_associations
save(compute_associations(master),'metric_associations')
null=pd.read_csv(table_path('results_summary/poi_null_model_tests.csv'));by=[]
for key,g in null.groupby(['null_type','spatial_block_m','poi_max_distance_m'],dropna=False):
 g=g.copy();n=len(g);p=g.p_value.to_numpy();order=np.argsort(p);q=np.minimum.accumulate((p[order]*n*np.sum(1/np.arange(1,n+1))/np.arange(1,n+1))[::-1])[::-1];vals=np.empty(n);vals[order]=np.minimum(1,q);g['p_value_by']=vals;g['significant_by_0_05']=vals<0.05;by.append(g)
by=pd.concat(by);save(by,'poi_by_sensitivity')
summary=by.groupby(['null_type','spatial_block_m','poi_max_distance_m'],dropna=False).agg(n=('city','size'),bh=('significant_bh_0_05','sum'),by=('significant_by_0_05','sum'),effect_min=('entropy_effect','min'),effect_max=('entropy_effect','max'),z_min=('z_score','min'),z_max=('z_score','max')).reset_index();save(summary,'poi_inference_summary')
# Direct classification audit, keeping the same POI coordinates and assignments.
fields=['amenity','shop','healthcare','leisure','office','craft']
def classify(row,order):
 for f in order:
  v=row.get(f)
  if isinstance(v,str) and v.strip():
   v=v.strip().lower()
   if v in legacy.NOISE_TYPES:return None
   return legacy.CAT_MAP.get(v,legacy.FIELD_FALLBACK.get(f))
 return None
poia=[];profiles=[];density=[];crs=[]
for cfg in legacy.CITIES:
 city=cfg['city']; raw=json.loads(Path(cfg['poi_file']).read_text())
 records=[];multi=0;changes=0;excluded=0;leisure_tagged=0
 for x in raw:
  if x.get('lat') is None or x.get('lon') is None:continue
  tags=[f for f in fields if isinstance(x.get(f),str) and x[f].strip()]
  multi+=len(tags)>1;leisure_tagged+='leisure' in tags
  base=classify(x,fields);alt=classify(x,['leisure']+[f for f in fields if f!='leisure'])
  changes+=base!=alt;excluded+=base is None
  if base is not None:records.append((float(x['lat']),float(x['lon']),base,alt or 'Excluded'))
 p=pd.DataFrame(records,columns=['lat','lon','category','leisure_first_category'])
 a=np.load(ROOT/'graphs'/f'{city}__Endpoint.npz'); coords=a['coordinates']; nodes=a['nodes']; lat0=float(coords[:,0].mean())
 tree=BallTree(np.radians(coords),metric='haversine');ds,idx=tree.query(np.radians(p[['lat','lon']]),k=1)
 p['node']=nodes[idx[:,0]];p['distance_m']=ds[:,0]*6371000
 p=p[p.distance_m<=1000].copy()
 part=np.load(ROOT/'partitions/legacy'/f'{city}__LeidenRBC.npz'); labels=dict(zip(part['nodes'],part['labels']));p['community']=p.node.map(labels)
 ncat=p.category.nunique()
 for cid,g in p.groupby('community'):
  proportions=g.category.value_counts(normalize=True);h=float(-(proportions*np.log(proportions)).sum()/np.log(ncat)) if ncat>1 else 0.
  alt=g.leisure_first_category.value_counts(normalize=True)
  profiles.append(dict(city=city,community=cid,n_pois=len(g),eligible=len(g)>=100 and len(g)>=len(p)*0.01,entropy=h,recreation_share=proportions.get('Recreation',0),leisure_first_recreation_share=alt.get('Recreation',0)))
 # Local planar km grid for descriptive density diagnostics only.
 x=p.lon.to_numpy()*np.pi/180*6371000*np.cos(np.radians(lat0));y=p.lat.to_numpy()*np.pi/180*6371000
 ox=float(coords[:,1].min()*np.pi/180*6371000*np.cos(np.radians(lat0)));oy=float(coords[:,0].min()*np.pi/180*6371000)
 p['gx']=np.floor((x-ox)/1000).astype(int);p['gy']=np.floor((y-oy)/1000).astype(int);p['recreation']=p.category.eq('Recreation')
 grid=p.groupby(['gx','gy']).agg(n_pois=('category','size'),recreation_share=('recreation','mean'))
 xx=coords[:,1]*np.pi/180*6371000*np.cos(np.radians(lat0));yy=coords[:,0]*np.pi/180*6371000
 ep=pd.DataFrame({'gx':np.floor((xx-ox)/1000).astype(int),'gy':np.floor((yy-oy)/1000).astype(int)})
 grid=grid.join(ep.groupby(['gx','gy']).size().rename('endpoints')).fillna({'endpoints':0});grid=grid[grid.n_pois>=20]
 grid.reset_index().assign(city=city).to_csv(R/f'density_grid_{city}.csv',index=False)
 density.append(dict(city=city,n_cells=len(grid),rho_endpoints=grid.recreation_share.corr(grid.endpoints,method='spearman'),rho_poi_count=grid.recreation_share.corr(grid.n_pois,method='spearman')))
 poia.append(dict(city=city,raw_records=len(raw),multiple_category_fields=multi,leisure_tagged=leisure_tagged,classification_changes_leisure_first=changes,excluded_no_category_or_noise=excluded,baseline_valid=len(records),matched_1000m=len(p)))
 # City-level equirectangular parameters; actual historical hull projection is per community.
 crs.append(dict(city=city,mean_latitude_deg=lat0,radius_m=6371000,city_bbox_center_utm_epsg=32600+int((coords[:,1].min()+coords[:,1].max())/2+180)//6+1))
 print('POI audit',city,flush=True)
save(pd.DataFrame(poia),'poi_tag_audit');save(pd.DataFrame(profiles),'all_eligible_profiles');save(pd.DataFrame(density),'poi_density_associations');save(pd.DataFrame(crs),'coordinate_parameters')
# Exact route-member signatures expose candidates, without calling legal variants errors.
cfg=next(c for c in legacy.CITIES if c['city']=='HongKong');raw=json.loads(Path(cfg['json_file']).read_text());els=raw.get('elements',[]) if isinstance(raw,dict) else raw
rels=[e for e in els if e.get('type')=='relation' and e.get('tags',{}).get('route') in ['bus','trolleybus']]
groups={}
for e in rels:
 sig=tuple((m.get('ref'),m.get('role','')) for m in e.get('members',[]) if m.get('type')=='way')
 if sig:groups.setdefault(sig,[]).append(e)
dup=[]
for sig,group in groups.items():
 if len(group)>1:
  for e in group:dup.append(dict(relation_id=e['id'],group_size=len(group),operator=e.get('tags',{}).get('operator',''),ref=e.get('tags',{}).get('ref',''),signature_sha256=hashlib.sha256(repr(sig).encode()).hexdigest()))
save(pd.DataFrame(dup,columns=['relation_id','group_size','operator','ref','signature_sha256']),'hongkong_identical_way_sequences')
summary_record={'inventory_rows':len(inv),'representation_comparisons':len(reps),'degenerate_comparisons':int(reps.degenerate.sum()),'hk_bus_relations_in_extract':len(rels),'hk_duplicate_signature_groups':sum(len(g)>1 for g in groups.values()),'hk_relations_in_duplicate_groups':len(dup),'source':'archived fits; no historical model refits; POI and route audit reads original inputs'}
(R/'summary.json').write_text(json.dumps(summary_record,indent=2,default=native)+'\n');print(summary_record)
