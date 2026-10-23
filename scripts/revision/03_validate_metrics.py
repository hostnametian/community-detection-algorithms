"""Independent small-graph checks of the published definitions and custom moves."""
from common import *
import math, random

def direct_reference(g,p):
    nodes=sorted(g); labels=sorted(set(p.values())); a=nx.to_numpy_array(g,nodelist=nodes,weight='weight')
    m=a.sum(); out=a.sum(axis=1); inc=a.sum(axis=0)
    same=np.array([[p[i]==p[j] for j in nodes] for i in nodes])
    q=float(((a-np.outer(out,inc)/m)*same).sum()/m)
    acts=[];acds=[]
    for label in labels:
        take=np.array([p[n]==label for n in nodes]); internal=a[np.ix_(take,take)].sum()
        boundary=a[np.ix_(take,~take)].sum()+a[np.ix_(~take,take)].sum()
        volume=(out[take]+inc[take]).sum()
        assert math.isclose(volume,2*internal+boundary)
        acts.append(2*internal/take.sum());acds.append(boundary/volume)
    und=nx.Graph();und.add_nodes_from(nodes)
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes[i:],i):
            w=a[i,j] if i==j else a[i,j]+a[j,i]
            if w:und.add_edge(u,v,weight=w)
    groups=[{n for n in nodes if p[n]==c} for c in labels]
    internal_graph=und.copy();internal_graph.remove_edges_from([(u,v) for u,v in und.edges if p[u]!=p[v]])
    e=nx.global_efficiency(und);ei=nx.global_efficiency(internal_graph)
    return {'ACT':float(np.mean(acts)),'ACD':float(np.mean(acds)), 'Q_directed':q,
            'Q':nx.community.modularity(und,groups,weight='weight'),'PL':(e-ei)/e}

def main():
    tests=[]
    for seed in range(20):
        rng=random.Random(seed);g=nx.DiGraph();g.add_nodes_from(range(8))
        for i in range(8):g.add_edge(i,(i+1)%8,weight=rng.randint(1,5))
        for i in range(8):
            for j in range(8):
                if rng.random()<.25:g.add_edge(i,j,weight=rng.randint(1,9))
        p={n:n%3 for n in g};actual=legacy.compute_standard_metrics(g,p);expected=direct_reference(g,p)
        for k,v in expected.items():
            assert math.isclose(actual[k],v,rel_tol=1e-10,abs_tol=1e-10),(seed,k,actual[k],v)
        # Direct modularity agreement with independent NetworkX implementation.
        for gamma in [.1,1,2]:
            groups=[{n for n in g if p[n]==c} for c in set(p.values())]
            q=nx.community.modularity(g,groups,weight='weight',resolution=gamma)
            assert math.isclose(legacy.directed_modularity(g,p,gamma),q,abs_tol=1e-10)
            # All single-node changes, including donor singleton and self loops.
            m=g.size(weight='weight')
            for n in g:
                old=p[n]
                def insertion_score(label):
                    members={x for x in g if p[x]==label and x!=n}
                    reward=sum(g.get_edge_data(n,x,{}).get('weight',0)+g.get_edge_data(x,n,{}).get('weight',0) for x in members)/m
                    ci=sum(g.in_degree(x,weight='weight') for x in members)
                    co=sum(g.out_degree(x,weight='weight') for x in members)
                    return reward-gamma*(g.out_degree(n,weight='weight')*ci+g.in_degree(n,weight='weight')*co)/m**2
                for label in range(4):
                    new=dict(p);new[n]=label
                    groups_new=[{x for x in g if new[x]==c} for c in set(new.values())]
                    dq=nx.community.modularity(g,groups_new,weight='weight',resolution=gamma)-q
                    assert math.isclose(dq,insertion_score(label)-insertion_score(old),abs_tol=1e-10)
        tests.append({'seed':seed,'metrics':expected,'passed':True})
    # Three-community weighted path: middle quotient node has betweenness 1.
    g=nx.DiGraph();g.add_weighted_edges_from([(0,1,2),(1,0,1),(1,2,3),(2,1,1)])
    result=legacy.compute_standard_metrics(g,{0:0,1:1,2:2})
    assert math.isclose(result['ACBC'],1/3);assert result['CGCC']==0
    triangle=nx.DiGraph();triangle.add_weighted_edges_from([(0,1,1),(1,2,1),(2,0,1)])
    assert math.isclose(legacy.compute_standard_metrics(triangle,{0:0,1:1,2:2})['CGCC'],1)
    # Diagnose the old approximation: it samples an induced subgraph, not source distances.
    path=nx.path_graph(100)
    exact=nx.global_efficiency(path);sampled=legacy._sampled_global_efficiency(path,sample_size=20,seed=42)
    # Missing-coordinate topology is tested without filtering the bridge node away.
    directed=nx.DiGraph([(0,1),(1,2)]);stats,_=topology_stats(directed,{0:0,1:0,2:0})
    assert stats['fragmentation_macro']==0
    import importlib.util
    spec=importlib.util.spec_from_file_location('smoke',SOURCE_ROOT/'00_validate_algorithms.py')
    smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke);smoke.main()
    write_json(ROOT/'audit/metric_validation.json',{'passed':True,'random_graph_checks':tests,
        'quotient_ACBC_CGCC_checks_passed':True,'existing_custom_smoke_checks_passed':True,
        'ACT_ACD_interpretation':'2*internal is internal total in+out strength, not double counting under that definition',
        'legacy_sampled_efficiency_counterexample':{'graph':'100-node path','exact':exact,'20_node_induced_sample':sampled},
        'policy':'Do not use old induced-subgraph efficiency approximation in new experiments. Exact PL for selected partitions; omit PL for candidate grid.'})
    print('Metric and local-gain checks passed; legacy sampled-efficiency bias recorded',flush=True)

if __name__=='__main__':main()
