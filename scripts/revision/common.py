"""Shared, provenance-aware helpers for first-review experiments."""
from __future__ import annotations
import os
from pathlib import Path
from paths import ROOT, SCRIPT_ROOT, SOURCE_ROOT, BASELINE_ROOT, BOUNDARY_ROOT, table_path, run_path
for folder in ('audit', 'graphs', 'partitions', 'results', 'logs', 'runtime'):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/matplotlib'))
os.environ['PTN_OUTPUT_ROOT'] = str(ROOT / 'runtime/legacy')
os.environ.setdefault('PTN_BOUNDARY_ROOT', str(BOUNDARY_ROOT))
import sys, json, hashlib, importlib, platform, fcntl
sys.path.insert(0, str(SOURCE_ROOT))
import numpy as np
import networkx as nx
import pandas as pd
import analysis_pipeline as legacy

def acquire_lock(key):
    directory=ROOT/'runtime/locks';directory.mkdir(parents=True,exist_ok=True)
    handle=(directory/(key+'.lock')).open('a')
    fcntl.flock(handle,fcntl.LOCK_EX)
    return handle


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    temp.replace(path)


def environment():
    packages = {}
    for name in ('numpy', 'networkx', 'networkit', 'scipy', 'pandas', 'sklearn',
                 'infomap', 'igraph', 'leidenalg', 'graph_tool', 'osmnx', 'geopandas'):
        module = importlib.import_module(name)
        packages[name] = getattr(module, '__version__', 'unknown')
    return {'python': platform.python_version(), 'executable': sys.executable, 'packages': packages}


def graph_arrays(graph):
    nodes = np.array(sorted(graph.nodes()), dtype=np.int64)
    edges = sorted(graph.edges(data=True))
    return {'nodes': nodes,
            'coordinates': np.array([[graph.nodes[n]['lat'], graph.nodes[n]['lon']] for n in nodes]),
            'source': np.array([u for u, v, d in edges], dtype=np.int64),
            'target': np.array([v for u, v, d in edges], dtype=np.int64),
            'weight': np.array([d.get('weight', 1) for u, v, d in edges], dtype=np.float64)}


def fingerprint(graph):
    h = hashlib.sha256()
    for k, a in graph_arrays(graph).items():
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_graph(graph, city, representation):
    path = ROOT / 'graphs' / f'{city}__{representation}.npz'
    np.savez_compressed(path, **graph_arrays(graph))
    return path


def load_graph(city, representation='Endpoint'):
    path = ROOT / 'graphs' / f'{city}__{representation}.npz'
    with np.load(path, allow_pickle=False) as a:
        g = nx.DiGraph()
        for n, coord in zip(a['nodes'], a['coordinates']):
            g.add_node(int(n), lat=float(coord[0]), lon=float(coord[1]))
        g.add_weighted_edges_from((int(u), int(v), float(w)) for u,v,w in zip(a['source'],a['target'],a['weight']))
    return g


def validate_partition(graph, partition):
    if set(graph) != set(partition) or any(v == -1 for v in partition.values()):
        raise ValueError('Partition must assign every graph node exactly once; no -1 labels allowed')


def topology_stats(graph, partition):
    validate_partition(graph, partition)
    groups = {}
    for node, label in partition.items(): groups.setdefault(label, []).append(node)
    rows = []
    for label, nodes in groups.items():
        sizes = [len(c) for c in nx.weakly_connected_components(graph.subgraph(nodes))]
        rows.append({'community': int(label), 'nodes': len(nodes), 'components': len(sizes),
                     'fragmentation': 1-max(sizes)/len(nodes)})
    frame = pd.DataFrame(rows)
    return {'K': len(rows), 'fragmentation_macro': float(frame.fragmentation.mean()),
            'fragmentation_node_weighted': float(np.average(frame.fragmentation, weights=frame.nodes)),
            'disconnected_communities': int((frame.components > 1).sum()),
            'disconnected_node_share': float(frame.loc[frame.components > 1, 'nodes'].sum()/len(graph)),
            'max_components': int(frame.components.max())}, frame


def save_partition(path, graph, partition, metadata):
    validate_partition(graph, partition)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.array(sorted(graph), dtype=np.int64)
    labels = np.array([partition[n] for n in nodes], dtype=np.int64)
    np.savez_compressed(path, nodes=nodes, labels=labels)
    write_json(path.with_suffix('.json'), {**metadata, 'graph_fingerprint': fingerprint(graph),
               'partition_sha256': digest(path), 'environment': environment(),
               'source_sha256': digest(SOURCE_ROOT/'analysis_pipeline.py')})


def load_partition(path, graph):
    with np.load(path, allow_pickle=False) as a:
        partition = dict(zip(a['nodes'].tolist(), a['labels'].tolist()))
    validate_partition(graph, partition)
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    if meta['graph_fingerprint'] != fingerprint(graph): raise ValueError('Graph fingerprint mismatch')
    if meta['partition_sha256'] != digest(path): raise ValueError('Partition hash mismatch')
    return partition, meta
