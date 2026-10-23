"""Efficient selected-partition spatial diagnostics, with explicit support."""
from common import *
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import haversine_distances
from pyproj import Transformer

def spatial(graph,partition,seed=42,repeats=20,sample_limit=300):
    validate_partition(graph,partition)
    nodes=sorted(graph)
    coords=np.array([[graph.nodes[n]['lat'],graph.nodes[n]['lon']] for n in nodes])
    if not np.isfinite(coords).all():raise ValueError('Missing coordinate coverage must be handled explicitly')
    zone=min(60,max(1,int((coords[:,1].mean()+180)//6)+1))
    epsg=(32600 if coords[:,0].mean()>=0 else 32700)+zone
    tr=Transformer.from_crs('EPSG:4326',f'EPSG:{epsg}',always_xy=True)
    x,y=tr.transform(coords[:,1],coords[:,0]);xy=np.column_stack([x,y]);radians=np.radians(coords)
    labels=np.array([partition[n] for n in nodes]);rows=[]
    for label in sorted(set(labels)):
        ix=np.flatnonzero(labels==label); points=xy[ix];n=len(ix)
        area=perimeter=compactness=float('nan')
        unique=np.unique(points,axis=0)
        if len(unique)>=3 and np.linalg.matrix_rank(unique-unique[0])==2:
            hull=ConvexHull(unique); area=float(hull.volume);perimeter=float(hull.area)
            boundary=unique[hull.vertices]; compactness=4*np.pi*area/perimeter**2
        else:boundary=unique
        diameter=float(pdist(boundary).max()) if len(boundary)>1 else 0.
        # Seed by actual node set, avoiding dependence on arbitrary numeric labels.
        community_seed=int(hashlib.sha256(np.array([nodes[i] for i in ix],dtype=np.int64).tobytes()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed+community_seed);means=[]
        for _ in range(repeats if n>sample_limit else 1):
            take=rng.choice(ix,sample_limit,replace=False) if n>sample_limit else ix
            d=haversine_distances(radians[take])*6371000.
            means.append(float(d.sum()/(len(take)*(len(take)-1))) if len(take)>1 else 0.)
        rows.append({'community':int(label),'nodes':n,'diameter_m':diameter,'mean_pairwise_m':float(np.mean(means)),
                     'mean_pairwise_sample_q025_m':float(np.quantile(means,.025)),
                     'mean_pairwise_sample_q975_m':float(np.quantile(means,.975)),
                     'compactness':compactness,'area_m2':area})
    f=pd.DataFrame(rows)
    summary={'spatial_crs':f'EPSG:{epsg}','coordinate_coverage':1.,'distance_repeats':repeats,'sample_limit':sample_limit}
    for k in ['diameter_m','mean_pairwise_m','compactness']:
        valid=f[k].notna();summary[k+'_macro']=float(f.loc[valid,k].mean()) if valid.any() else float('nan')
        summary[k+'_node_weighted']=float(np.average(f.loc[valid,k],weights=f.loc[valid,'nodes'])) if valid.any() else float('nan')
    return summary,f

def quick(graph,partition):
    """Grid diagnostics: no efficiency approximation and no spatial sampling."""
    stats,_=topology_stats(graph,partition)
    stats['Q_directed']=legacy.directed_modularity(graph,partition,resolution=1.)
    return stats
