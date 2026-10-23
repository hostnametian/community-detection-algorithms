"""Audit boundary losses and route-gap reporting without changing the frozen graph."""
from common import *
from collections import Counter

def main():
    rows=[]
    for cfg in legacy.CITIES:
        b=legacy.PTNGraphBuilder(cfg['json_file'],cfg['city'],cfg['country'],100,city_query=cfg['city_query'])
        b.build()
        referenced=set(m['ref'] for r in b.relations.values() if b._is_bus_route(r)
                       for m in r.get('members',[]) if m.get('type')=='way')
        counts=Counter()
        for wid in referenced:
            if wid not in b.ways:counts['referenced_way_absent_from_extract']+=1;continue
            ns=b.ways[wid].get('nodes',[])
            if not ns:counts['empty_way']+=1;continue
            if any(n not in b.nodes for n in ns):counts['way_with_missing_nodes']+=1;continue
            inside=[n in b.fn for n in ns]
            if all(inside):counts['all_nodes_inside_retained']+=1
            elif any(inside):
                counts['mixed_inside_outside_way_discarded']+=1
                counts['inside_segments_on_discarded_ways']+=sum(a and c for a,c in zip(inside,inside[1:]))
            else:counts['all_nodes_outside_way_discarded']+=1
        actual_continuous=0;gap_count=0
        for route in b.bus_routes.values():
            edges=route['way_edges']
            gaps=sum(a['end_node']!=c['start_node'] for a,c in zip(edges,edges[1:]))
            actual_continuous+=gaps==0;gap_count+=gaps
        original=b.stats['route_quality']
        rows.append({'city':cfg['city'],**counts,'retained_routes':len(b.bus_routes),
                     'legacy_continuous_routes':original.get('continuous_route_relations',0),
                     'all_roles_continuous_routes':actual_continuous,'all_roles_adjacent_way_gaps':gap_count,
                     'gap_definition':'retained directed way endpoints, all member roles; not a reconstruction of missing ways'})
        pd.DataFrame(rows).to_csv(ROOT/'audit/graph_quality.csv',index=False)
        print(cfg['city'],'reported continuous',original.get('continuous_route_relations'),
              'all-role audit',actual_continuous,flush=True)

if __name__=='__main__':main()
