"""Explicit model settings for the new, separately versioned comparison."""
from common import *
import random

def dcsbm(graph, family='discrete-geometric', seed=42, target_k=None):
    """Counts are excess route counts (w-1) conditional on an edge being present.

    The real-normal branch reproduces the legacy covariate w and is descriptive
    only: continuous and discrete description lengths are not ranked together.
    """
    gt=legacy.gt
    nodes=sorted(graph);index={n:i for i,n in enumerate(nodes)}
    g=gt.Graph(directed=True);g.add_vertex(len(nodes))
    discrete=family.startswith('discrete-')
    prop=g.new_ep('int64_t' if discrete else 'double')
    for u,v,d in sorted(graph.edges(data=True)):
        w=float(d['weight'])
        if w<1 or w!=int(w):raise ValueError('Route weight must be a positive integer')
        prop[g.add_edge(index[u],index[v])]=int(w)-1 if discrete else w
    gt.seed_rng(seed)
    kwargs={} if target_k is None else {'B_min':target_k,'B_max':target_k}
    state=gt.minimize_blockmodel_dl(g,state_args={'deg_corr':True,'recs':[prop],'rec_types':[family]},
                                  multilevel_mcmc_args=kwargs)
    labels=state.get_blocks().a
    p={n:int(labels[i]) for i,n in enumerate(nodes)}
    actual_k=len(set(p.values()))
    if target_k is not None and actual_k!=target_k:raise RuntimeError(f'Constrained fit returned K={actual_k}, expected {target_k}')
    return p,{'family':family,'weight_transform':'w-1' if discrete else 'w',
              'seed':seed,'target_K':target_k,'actual_K':actual_k,
              'description_length':float(state.entropy()),'directed':True,'deg_corr':True,
              'model':'non-nested weighted degree-corrected SBM; topology not expanded by route weight'}

def infomap(graph,markov_time=1.,seed=42):
    im=legacy.infomap.Infomap(silent=True,directed=True,seed=seed,markov_time=markov_time,
                             recorded_teleportation=False,to_nodes=False,teleportation_probability=.15,
                             num_trials=1,num_threads=1)
    nodes=sorted(graph); index={n:i for i,n in enumerate(nodes)}
    for u,v,d in sorted(graph.edges(data=True)):im.add_link(index[u],index[v],d['weight'])
    im.run(); p={nodes[leaf.node_id]:int(leaf.module_id) for leaf in im.tree if leaf.is_leaf}
    validate_partition(graph,p)
    return p,{'seed':seed,'markov_time':markov_time,'teleportation_probability':.15,
              'recorded_teleportation':False,'to_nodes':False,'num_trials':1,
              'partition_level':'top module_id from leaf iteration, as in legacy code',
              'library_version':legacy.infomap.__version__,'codelength':float(im.codelength)}

def spinglass(graph,spins=25,seed=42):
    ig=legacy.ig
    # The historical wrapper checked a nonexistent set_random_instance method.
    # This documented igraph hook seeds the actual library's RNG.
    ig.set_random_number_generator(random.Random(seed))
    und=legacy.weighted_undirected_projection(graph)
    g=ig.Graph.from_networkx(und)
    part=g.community_spinglass(weights='weight',spins=spins)
    p={g.vs[i]['_nx_name']:int(part.membership[i]) for i in range(g.vcount())}
    return p,{'spins':spins,'seed':seed,'rng':'igraph.set_random_number_generator(random.Random(seed))'}

def coordinate_kmeans(graph,k,seed=42):
    from sklearn.cluster import KMeans
    nodes=sorted(graph)
    # Same local equirectangular coordinate plane as GeoLP; no edges used.
    coords=np.array([[graph.nodes[n]['lat'],graph.nodes[n]['lon']] for n in nodes])
    xy=np.column_stack((6371000*np.radians(coords[:,1])*np.cos(np.radians(coords[:,0].mean())),
                        6371000*np.radians(coords[:,0])))
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(xy)
    return dict(zip(nodes,km.labels_.astype(int))),{'K':k,'seed':seed,'n_init':10,'inertia':float(km.inertia_),
                'projection':'same local equirectangular metric plane as GeoLP','uses_edges':False}
