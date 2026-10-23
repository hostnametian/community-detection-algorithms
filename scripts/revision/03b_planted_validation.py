"""Planted directed graphs, including aligned and shuffled spatial metadata.

Recovery scores are descriptive checks, not a theorem or a demand that algorithms
with different objectives recover every planted labelling.
"""
from common import *
from sklearn.metrics import adjusted_rand_score

def main():
    rows=[]
    for mixing in [.02,.10,.25]:
        for seed in range(5):
            rng=np.random.default_rng(seed)
            g=nx.stochastic_block_model([20]*3,[[.5 if i==j else mixing for j in range(3)] for i in range(3)],seed=seed,directed=True)
            for u,v in g.edges:g[u][v]['weight']=int(rng.integers(1,5))
            for spatial_mode in ['aligned','shuffled']:
                centres=np.repeat(np.arange(3),20)
                if spatial_mode=='shuffled':centres=rng.permutation(centres)
                for node in g:
                    g.nodes[node]['lat']=55.+.10*centres[node]+float(rng.normal(0,.003))
                    g.nodes[node]['lon']=37.+float(rng.normal(0,.003))
                for algorithm in ['DirLou','GeoLP']:
                    kwargs={'seed':42,'resolution':1.} if algorithm=='DirLou' else {'seed':42,'initial_communities':3}
                    p,m=legacy.ALGORITHM_REGISTRY[algorithm](g,**kwargs)
                    validate_partition(g,p)
                    if algorithm=='DirLou':
                        trace=m['algorithm_parameters']['objective_trace']
                        if any(b<a-1e-10 for a,b in zip(trace,trace[1:])):raise AssertionError('Coarsening decreases objective')
                    rows.append({'algorithm':algorithm,'mixing_probability':mixing,'seed':seed,
                                 'spatial_mode':spatial_mode,'K':len(set(p.values())),
                                 'ARI_planted':adjusted_rand_score([n//20 for n in sorted(g)],[p[n] for n in sorted(g)]),
                                 'Q_directed':legacy.directed_modularity(g,p,1.)})
    pd.DataFrame(rows).to_csv(ROOT/'audit/planted_validation.csv',index=False)
    print('Completed',len(rows),'planted graph fits without coverage/objective-trace failures',flush=True)

if __name__=='__main__':main()
