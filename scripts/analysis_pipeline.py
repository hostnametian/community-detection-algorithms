# %%
# %% Cell 1 — 导入 + 全局配置
from __future__ import annotations

import json, csv, os, random, time, traceback, inspect, webbrowser, hashlib
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise, combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from sklearn.neighbors import BallTree
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox
import folium
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

# Use the same standard OSM/Mapnik background as the original notebook.
# A descriptive User-Agent prevents the generic contextily client from being
# rejected by the official tile service; contextily keeps downloaded tiles in
# its cache and fetches them single-threaded by default.
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_TILE_HEADERS = {
    "User-Agent": "BusNetworkCommunityResearch/1.0 (academic figure generation; cached single-threaded requests)"
}

from geopy.distance import geodesic
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from IPython.display import IFrame, display as ipy_display
from IPython.display import display
from plotly.subplots import make_subplots

# 设置 plotly 默认导出格式
pio.kaleido.scope.default_format = "png"

# ======== 依赖检查 ========
try:
    import infomap
    INFOMAP_OK = True
except ImportError:
    INFOMAP_OK = False
    print("⚠ infomap missing")

try:
    import leidenalg as la
    import igraph as ig
    LEIDEN_OK = True
except ImportError:
    LEIDEN_OK = False
    print("⚠ leidenalg/igraph missing")

try:
    import graph_tool.all as gt
    GT_OK = True
except ImportError:
    GT_OK = False
    print("⚠ graph-tool missing")

try:
    import igraph as ig
    IGRAPH_OK = True
except ImportError:
    IGRAPH_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠ scikit-learn missing")

try:
    import networkit as nk
    NETWORKIT_OK = True
except ImportError:
    NETWORKIT_OK = False

print("依赖检查完成。")

PIPELINE_VERSION = "ceus_v3"

# ======== Repository, data, and output roots ========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PTN_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("PTN_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = OUTPUT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUMMARY_DIR = OUTPUT_ROOT / "results_summary"
SUMMARY_DIR.mkdir(exist_ok=True)

# ======== City configuration ========
_DEFAULT_CITIES = [
    {
        "city": "BeiJing",
        "country": "China",
        "city_query": "Beijing",
        "json_file": "BeiJing_bus.json",
        "poi_file": "BeiJing_pois.json",
    },
    {
        "city": "ShangHai",
        "country": "China",
        "city_query": "Shanghai",
        "json_file": "ShangHai_bus.json",
        "poi_file": "ShangHai_pois.json",
    },
    {
        "city": "HongKong",
        "country": "China",
        "city_query": "Hong Kong",
        "json_file": "HongKong_bus.json",
        "poi_file": "HongKong_pois.json",
    },
    {
        "city": "Moscow",
        "country": "Russia",
        "city_query": "Moscow",
        "json_file": "Москва_bus.json",
        "poi_file": "Москва_pois.json",
    },
    {
        "city": "StPetersburg",
        "country": "Russia",
        "city_query": "Saint Petersburg",
        "json_file": "Санкт-Петербург_bus.json",
        "poi_file": "Санкт-Петербург_pois.json",
    },
    {
        "city": "Kazan",
        "country": "Russia",
        "city_query": "Kazan",
        "json_file": "городской округ Казань_bus.json",
        "poi_file": "городской округ Казань_pois.json",
    },
    {
        "city": "Yekaterinburg",
        "country": "Russia",
        "city_query": "Yekaterinburg",
        "json_file": "городской округ Екатеринбург_bus.json",
        "poi_file": "городской округ Екатеринбург_pois.json",
    },
]


def _load_city_configuration():
    """Load a public, path-portable city configuration.

    ``PTN_CITY_CONFIG`` may point to a JSON array with the same fields as
    ``config/cities.json``. Relative ``json_file`` and ``poi_file`` values are
    resolved under ``PTN_DATA_ROOT`` (``data/`` by default).
    """
    config_path = Path(
        os.environ.get("PTN_CITY_CONFIG", PROJECT_ROOT / "config" / "cities.json")
    ).expanduser()
    if config_path.is_file():
        configured = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(configured, list):
            raise ValueError(f"City configuration must be a JSON array: {config_path}")
    else:
        configured = _DEFAULT_CITIES

    cities = []
    required = {"city", "country", "city_query", "json_file", "poi_file"}
    for raw in configured:
        missing = required - set(raw)
        if missing:
            raise ValueError(f"City configuration is missing {sorted(missing)}: {raw}")
        city = dict(raw)
        for field in ("json_file", "poi_file"):
            path = Path(city[field]).expanduser()
            if not path.is_absolute():
                path = DATA_ROOT / path
            city[field] = str(path.resolve())
        cities.append(city)
    return cities


CITIES = _load_city_configuration()

CITY_POI_FILES = {c["city"]: c["poi_file"] for c in CITIES}

# ======== 全局方法配置 ========
GLOBAL_CFG = {
    "distance_threshold_m": 100,
    "connectivity_filter": "GWCC",
    "boundary_method": "osmnx_geocode_to_gdf",
    "graph_type": "directed_weighted_route_endpoint_graph",
    "direction_method": "route_relation_member_order",
    "random_seed": 42,
    "poi_max_distance_m": 1000,
    "geolp_bandwidth_m": 5000,
    "geolp_initial_communities": 25,
    "spinglass_spins": 25,
    "representation_transfer_thresholds_m": [100, 250, 500, 1000],
    "dcsbm_restarts": 50,
    "poi_permutations": 9999,
    "poi_match_distances_m": [250, 500, 750, 1000],
    "poi_spatial_null_blocks_m": [1000, 2000, 5000]
}

ALGORITHMS = ["Infomap", "LeidenRBC", "DCSBM", "GeoLP", "Spinglass", "DirLou"]

MAIN_ALGORITHMS = ["Infomap", "LeidenRBC", "DCSBM"]
BASELINE_ALGORITHMS = ["GeoLP", "Spinglass"]
EXPERIMENTAL_ALGORITHMS = ["DirLou"]


# ======== 论文出图配置 ========
FIGURE_CFG = {
    "mode": "single_column",   # 可选: "single_column", "double_column"
    "dpi": 600,
    "single_column_mm": 85,
    "double_column_mm": 180
}

def get_figure_width_mm(kind="map"):
    """
    统一返回论文出图宽度（毫米）。
    kind 预留扩展，目前 map / network / table 可共用。
    """
    mode = FIGURE_CFG.get("mode", "double_column")
    if mode == "single_column":
        return FIGURE_CFG["single_column_mm"]
    return FIGURE_CFG["double_column_mm"]

DISPLAY_OUTPUT = False

# %%
# %% Cell 2 — 工具 + 保存

CB_HEX = [
    "#fb0000", "#000080", "#04F604", "#4b0082", "#e8985f", "#2b8282", "#773c00",
    "#9c4254", "#226fbc", "#2c822c", "#9932cc", "#dedb0a", "#00f7f7", "#ffb803",
    "#A68E89", "#7c96e4", "#6fae6f", "#da70d6", "#ff5e23", "#2B847E", "#896103",
    "#f08080", "#87ceeb", "#005500", "#dda0dd"
]

class CommunityColorManager:
    def __init__(self, palette=None):
        self._p = list(palette) if palette else CB_HEX
        self._m = {}
        self._u = set()

    def get_color(self, cid):
        if cid < 0:
            return "#808080"
        if cid not in self._m:
            available = set(range(len(self._p))) - self._u
            idx = min(available) if available else len(self._m) % len(self._p)
            self._m[cid] = self._p[idx]
            self._u.add(idx)
        return self._m[cid]

    def reset(self):
        self._m.clear()
        self._u.clear()

    def assign_batch(self, cids):
        self.reset()
        for i, cid in enumerate(cids):
            idx = i % len(self._p)
            self._m[cid] = self._p[idx]
            self._u.add(idx)

def haversine(c1, c2):
    """
    纯 haversine 公式，返回两点间距离（米）。
    c1, c2: (lat, lon)
    比 geopy.geodesic 快 50~100 倍，精度足够（误差 <0.3%）。
    """
    lat1, lon1 = np.radians(c1[0]), np.radians(c1[1])
    lat2, lon2 = np.radians(c2[0]), np.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(a))


def _find_closest_segment(poly, pt):
    """
    在 polyline 上找到离 pt 最近的线段及投影点。
    poly: [(lat, lon), ...]
    pt  : (lat, lon)
    """
    line = LineString([(lon, lat) for lat, lon in poly])
    proj_len = line.project(Point(pt[1], pt[0]))
    proj_pt = line.interpolate(proj_len)
    proj_coord = (proj_pt.y, proj_pt.x)

    acc = 0.0
    for i, (a, b) in enumerate(pairwise(poly)):
        seg_len = haversine(a, b)
        if acc + seg_len >= proj_len:
            return i, proj_coord
        acc += seg_len

    return max(len(poly) - 2, 0), proj_coord

def inject_stop_to_polyline(poly, stop_coord, tol=3.0):
    """
    若 stop 已接近 poly 中某点，则返回该索引；
    否则将投影点插入 poly 并返回插入位置。
    """
    for i, p in enumerate(poly):
        if haversine(p, stop_coord) <= tol:
            return i
    idx, proj = _find_closest_segment(poly, stop_coord)
    poly.insert(idx + 1, proj)
    return idx + 1

def is_confirmed_bus_stop(tags):
    pos = [
        tags.get("public_transport") in ["stop_position", "platform", "station"],
        tags.get("highway") == "bus_stop",
        tags.get("amenity") == "bus_station",
        tags.get("bus") == "yes",
        tags.get("trolleybus") == "yes"
    ]
    neg = [
        tags.get("highway") in ["traffic_signals", "motorway_junction", "crossing", "give_way", "stop"],
        tags.get("amenity") in ["fuel", "parking", "restaurant", "cafe"],
        tags.get("junction") is not None,
        tags.get("highway") in ["primary", "secondary", "tertiary", "residential", "service"],
        "bridge" in tags or "tunnel" in tags,
        tags.get("barrier") in ["toll_booth"]
    ]
    return any(pos) and not any(neg)

def _json_safe(obj):
    try:
        np_types = (np.integer, np.floating, np.bool_)
    except Exception:
        np_types = tuple()

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if np_types and isinstance(obj, np_types):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_safe(v) for v in obj], key=str)
    return str(obj)

def _gini(vals):
    v = np.sort(np.array(vals, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))

# =========================================================
# 保存函数
# =========================================================
def _save_plotly(fig, city, algo, name, width=None, height=None):
    """
    保存 Plotly 图：HTML + PNG + PDF
    """
    d = RESULTS_DIR / city / algo / "figures"
    d.mkdir(parents=True, exist_ok=True)

    html_path = d / f"{name}.html"
    png_path = d / f"{name}.png"
    pdf_path = d / f"{name}.pdf"

    fig.write_html(str(html_path))

    export_w = width or fig.layout.width or 1200
    export_h = height or fig.layout.height or 800

    try:
        fig.write_image(str(png_path), width=export_w, height=export_h, scale=2)
        fig.write_image(str(pdf_path), width=export_w, height=export_h)
        print(f"  [fig] {png_path.name} + {pdf_path.name}")
    except Exception as e:
        print(f"  [fig] PNG/PDF failed ({e}), HTML saved: {html_path.name}")

    return html_path

def _save_plotly_summary(fig, name, width=None, height=None):
    """
    保存到 results_summary/
    """
    html_path = SUMMARY_DIR / f"{name}.html"
    png_path = SUMMARY_DIR / f"{name}.png"
    pdf_path = SUMMARY_DIR / f"{name}.pdf"

    fig.write_html(str(html_path))

    export_w = width or fig.layout.width or 1200
    export_h = height or fig.layout.height or 800

    try:
        fig.write_image(str(png_path), width=export_w, height=export_h, scale=2)
        fig.write_image(str(pdf_path), width=export_w, height=export_h)
        print(f"  [fig] {png_path.name} + {pdf_path.name}")
    except Exception as e:
        print(f"  [fig] PNG/PDF failed ({e}), HTML only: {html_path.name}")

    return html_path

def _apply_heatmap_style(
    fig,
    annotation_size=17,
    width=900,
    height=500,
    left=120,
    right=48,
    bottom=105,
    x_tickangle=-38,
    x_title=None,
    y_title=None,
):
    """Apply a common compact style without changing heatmap data or colours."""
    fig.update_traces(
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=annotation_size),
        xgap=1,
        ygap=1,
        colorbar=dict(
            tickfont=dict(size=15),
            thickness=16,
            len=0.98,
            x=1.015,
            xpad=4,
        ),
        selector=dict(type="heatmap"),
    )
    fig.update_layout(
        title=None,
        width=width,
        height=height,
        margin=dict(l=left, r=right, t=15, b=bottom),
        plot_bgcolor="#303030",
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(
        tickfont=dict(size=16),
        title_font=dict(size=18),
        tickangle=x_tickangle,
        automargin=True,
        showline=True,
        linecolor="#303030",
        linewidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=16),
        title_font=dict(size=18),
        automargin=True,
        showline=True,
        linecolor="#303030",
        linewidth=1,
        mirror=True,
    )
    return fig

def _save_folium(m, city, algo, name, auto_open=False, use_iframe=False):
    """
    保存 folium 地图（HTML）。
    在无桌面/无浏览器环境下建议 auto_open=False。
    """
    d = RESULTS_DIR / city / algo / "figures"
    d.mkdir(parents=True, exist_ok=True)

    path = d / f"{name}.html"
    m.save(str(path))
    print(f"  [map saved] {path}")

    if use_iframe:
        try:
            ipy_display(IFrame(src=str(path), width=1100, height=700))
        except Exception as e:
            print(f"  [map] IFrame display failed: {e}")

    if auto_open:
        try:
            ok = webbrowser.open(path.as_uri())
            if ok:
                print("  [map] opened in browser")
            else:
                print("  [map] browser not available; file saved only")
        except Exception as e:
            print(f"  [map] browser open failed: {e}")

    return path



# %%
# %% Cell 3 — 统一图构建器（最终整洁版）

class PTNGraphBuilder:
    def __init__(self, jf, cn, co, dt=100, city_query=None):
        self.jf = jf
        self.cn = cn                     # 英文显示名
        self.co = co
        self.city_query = city_query if city_query is not None else cn
        self.dt = dt

        # 城市边界
        self.city_boundary = None
        self.boundary_coords = []
        self.boundary_bounds = None
        self.boundary_area_deg2 = None

        # 原始与过滤后的 OSM 数据
        self.nodes = {}
        self.ways = {}
        self.relations = {}

        self.fn = {}   # filtered nodes
        self.fw = {}   # filtered ways
        self.fr = {}   # filtered relations

        # 站点与聚合
        self.bus_stops = {}
        self.csn = set()       # confirmed stop node ids
        self.ga = {}           # aggregated groups
        self.s2a = {}          # stop -> aggregate_id
        self.ac = 0            # aggregate counter

        # 路线与图
        self.bus_routes = {}
        self.sow = {}          # stop-on-way mapping
        self.G = nx.DiGraph()

        # 统计
        self.stats = {
            "nodes_before": 0,
            "edges_before": 0,
            "nodes_after": 0,
            "edges_after": 0,
            "route_quality": {},
            "gwcc": {},
        }

    # =====================================================
    # 边界
    # =====================================================
    def get_city_boundary(self):
        print(f"  [boundary] {self.cn}, {self.co}")
        try:
            cache_name = f"{self.city_query.replace(' ', '_')}_{self.co}_boundary.geojson"
            input_cache_path = Path(os.environ.get("PTN_BOUNDARY_ROOT", CODE_ROOT / "输入缓存" / "城市边界")) / cache_name
            output_cache_path = OUTPUT_ROOT / "threshold_sensitivity_output" / "boundary_cache" / cache_name
            if input_cache_path.exists():
                gdf = gpd.read_file(input_cache_path)
            elif output_cache_path.exists():
                gdf = gpd.read_file(output_cache_path)
            else:
                gdf = ox.geocode_to_gdf(f"{self.city_query}, {self.co}")
            gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.001)

            self.city_boundary = gdf.geometry.iloc[0]
            self.boundary_bounds = self.city_boundary.bounds
            self.boundary_area_deg2 = float(self.city_boundary.area)

            if hasattr(self.city_boundary, "exterior"):
                coords = list(self.city_boundary.exterior.coords)
            elif hasattr(self.city_boundary, "geoms"):
                largest = max(self.city_boundary.geoms, key=lambda p: p.area)
                coords = list(largest.exterior.coords)
            else:
                raise ValueError("Unknown geometry type")

            # shapely coords are (lon, lat), folium wants [lat, lon]
            self.boundary_coords = [[lat, lon] for lon, lat in coords]

        except Exception as e:
            print(f"  [boundary] FAILED: {e}")
            self.city_boundary = None
            self.boundary_coords = []
            self.boundary_bounds = None
            self.boundary_area_deg2 = None

    def _in(self, lat, lon):
        if self.city_boundary is None:
            return True
        p = Point(lon, lat)
        return self.city_boundary.contains(p) or self.city_boundary.touches(p)

    # =====================================================
    # 数据加载与边界过滤
    # =====================================================
    def load_data(self):
        print(f"  [load] {self.jf}")
        with open(self.jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        for e in data["elements"]:
            t = e["type"]
            if t == "node":
                self.nodes[e["id"]] = e
            elif t == "way":
                self.ways[e["id"]] = e
            elif t == "relation":
                self.relations[e["id"]] = e

        print(f"  [load] N={len(self.nodes)} W={len(self.ways)} R={len(self.relations)}")

    def filter_data(self):
        """
        基于城市边界过滤 OSM 元素：
        - 保留边界内节点
        - 仅保留所有 constituent nodes 都在边界内的 way
        - 仅保留引用了这些 way 的 relation
        """
        if self.city_boundary is None:
            self.fn = self.nodes.copy()
            self.fw = self.ways.copy()
            self.fr = self.relations.copy()
            return

        for nid, n in self.nodes.items():
            if self._in(n["lat"], n["lon"]):
                self.fn[nid] = n

        valid_nodes = set(self.fn)

        for wid, w in self.ways.items():
            if all(nid in valid_nodes for nid in w.get("nodes", [])):
                self.fw[wid] = w

        for rid, r in self.relations.items():
            if any(
                m.get("type") == "way" and m.get("ref") in self.fw
                for m in r.get("members", [])
            ):
                self.fr[rid] = r

        print(f"  [filter] N={len(self.fn)} W={len(self.fw)} R={len(self.fr)}")

    @staticmethod
    def _is_bus_route(relation):
        tags = relation.get("tags", {})
        return tags.get("type") == "route" and tags.get("route") in {"bus", "trolleybus"}

    # =====================================================
    # 站点识别与聚合
    # =====================================================
    def identify_stops(self):
        for nid, nd in self.fn.items():
            tg = nd.get("tags", {})
            if tg.get("public_transport") in ["platform", "stop_position"]:
                self.bus_stops[nid] = {
                    "id": nid,
                    "lat": nd["lat"],
                    "lon": nd["lon"],
                    "name": tg.get("name", f"Stop_{nid}"),
                    "type": tg["public_transport"],
                    "tags": tg
                }
                if is_confirmed_bus_stop(tg):
                    self.csn.add(nid)

        print(f"  [stops] total={len(self.bus_stops)} confirmed={len(self.csn)}")

    def _cluster(self, stops):
        """
        基于 BallTree 的空间聚类（connected components 方式）。
        比原来的暴力 O(n³) 快 100~1000 倍。
        """
        if len(stops) <= 1:
            return [stops]

        coords = np.array([[s["lat"], s["lon"]] for s in stops])
        coords_rad = np.radians(coords)

        # BallTree 用 haversine metric，单位是弧度
        # self.dt 是米，转成弧度：meters / 地球半径
        radius_rad = self.dt / 6371000.0

        tree = BallTree(coords_rad, metric="haversine")
        neighbors = tree.query_radius(coords_rad, r=radius_rad)

        # 用 BFS/DFS 找连通分量
        visited = [False] * len(stops)
        clusters = []

        for i in range(len(stops)):
            if visited[i]:
                continue

            cluster = []
            stack = [i]

            while stack:
                idx = stack.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                cluster.append(stops[idx])

                for nb in neighbors[idx]:
                    if not visited[nb]:
                        stack.append(nb)

            clusters.append(cluster)

        return clusters


    def aggregate(self):
        name_groups = defaultdict(list)
        for st in self.bus_stops.values():
            name_groups[st["name"]].append(st)

        for stops in name_groups.values():
            for cluster in self._cluster(stops):
                self.ac += 1
                aid = self.ac

                self.ga[aid] = {
                    "name": cluster[0]["name"],
                    "stops": cluster,
                    "center_lat": np.mean([c["lat"] for c in cluster]),
                    "center_lon": np.mean([c["lon"] for c in cluster]),
                    "cluster_size": len(cluster)
                }

                for c in cluster:
                    self.s2a[c["id"]] = aid

        print(f"  [agg] {len(self.ga)} groups")

    # =====================================================
    # 路线抽取
    # =====================================================
    def _way_agg(self, way_bus_stops):
        if not way_bus_stops:
            return []

        grouped = defaultdict(list)
        for st in way_bus_stops:
            grouped[st["aggregate_id"]].append(st)

        visible = []
        for sts in grouped.values():
            sts.sort(key=lambda x: x["way_node_index"])
            rep = sts[0] if len(sts) == 1 else sts[len(sts) // 2]
            rep["is_representative"] = True
            rep["represented_count"] = len(sts)
            visible.append(rep)

        return visible

    def _proc_way(self, wid, rid, rn, rr, oriented_nodes=None, member_role=""):
        if wid not in self.fw:
            return None

        w = self.fw[wid]
        wn = list(oriented_nodes) if oriented_nodes is not None else list(w.get("nodes", []))
        if len(wn) < 2:
            return None

        poly = []
        way_bus_stops = []

        for idx, nid in enumerate(wn):
            if nid in self.fn:
                nd = self.fn[nid]
                coord = (nd["lat"], nd["lon"])
                poly.append(coord)

                if nid in self.bus_stops:
                    way_bus_stops.append({
                        "node_id": nid,
                        "aggregate_id": self.s2a.get(nid),
                        "name": self.bus_stops[nid]["name"],
                        "type": self.bus_stops[nid]["type"],
                        "lat": coord[0],
                        "lon": coord[1],
                        "way_node_index": idx,
                        "polyline_index": len(poly) - 1
                    })

        if not poly:
            return None

        visible = self._way_agg(way_bus_stops)

        return {
            "way_id": wid,
            "way_name": w.get("tags", {}).get("name", f"Way_{wid}"),
            "route_id": rid,
            "route_name": rn,
            "route_ref": rr,
            "member_role": member_role,
            "complete_polyline_coords": poly,
            "start_node": wn[0],
            "end_node": wn[-1],
            "original_stops": way_bus_stops,
            "visible_stops": visible
        }

    def extract_routes(self):
        quality = Counter({
            "raw_route_relations": sum(self._is_bus_route(rel) for rel in self.relations.values()),
            "filtered_route_relations": sum(self._is_bus_route(rel) for rel in self.fr.values()),
        })
        for rid, rel in self.fr.items():
            tg = rel.get("tags", {})

            if not self._is_bus_route(rel):
                continue

            way_edges = []
            previous_end = None
            way_members = [m for m in rel.get("members", []) if m.get("type") == "way"]
            retained_way_members = [m for m in way_members if m.get("ref") in self.fw]
            quality["way_members_total"] += len(way_members)
            quality["way_members_retained"] += len(retained_way_members)
            quality["missing_or_boundary_filtered_way_members"] += len(way_members) - len(retained_way_members)
            if len(retained_way_members) < len(way_members):
                quality["boundary_truncated_relations"] += 1

            route_gaps = 0
            for member_index, member in enumerate(way_members):
                wid = member.get("ref")
                if wid in self.fw:
                    raw_nodes = list(self.fw[wid].get("nodes", []))
                    role = (member.get("role") or "").strip().lower()
                    reverse = role in {"backward", "reverse", "-1"}
                    explicit_role = role in {"forward", "1", "backward", "reverse", "-1"}
                    if explicit_role:
                        quality["ways_oriented_by_explicit_role"] += 1
                    elif raw_nodes:
                        inferred_from_continuity = False
                        if previous_end is not None:
                            if raw_nodes[-1] == previous_end and raw_nodes[0] != previous_end:
                                reverse = True
                                inferred_from_continuity = True
                            elif raw_nodes[0] == previous_end:
                                reverse = False
                                inferred_from_continuity = True
                            else:
                                route_gaps += 1
                        else:
                            next_nodes = []
                            for next_member in way_members[member_index + 1:]:
                                next_wid = next_member.get("ref")
                                if next_wid in self.fw:
                                    next_nodes = list(self.fw[next_wid].get("nodes", []))
                                    break
                            next_endpoints = {next_nodes[0], next_nodes[-1]} if next_nodes else set()
                            if raw_nodes[0] in next_endpoints and raw_nodes[-1] not in next_endpoints:
                                reverse = True
                                inferred_from_continuity = True
                            elif raw_nodes[-1] in next_endpoints and raw_nodes[0] not in next_endpoints:
                                reverse = False
                                inferred_from_continuity = True
                        if inferred_from_continuity:
                            quality["ways_oriented_by_endpoint_continuity"] += 1
                        else:
                            quality["ways_with_ambiguous_direction"] += 1
                    if reverse:
                        quality["reversed_ways"] += 1
                    oriented_nodes = list(reversed(raw_nodes)) if reverse else raw_nodes
                    e = self._proc_way(
                        wid,
                        rid,
                        tg.get("name", f"Route_{rid}"),
                        tg.get("ref", f"R{rid}"),
                        oriented_nodes=oriented_nodes,
                        member_role=role
                    )
                    if e:
                        way_edges.append(e)
                        previous_end = e["end_node"]

            if way_edges:
                self.bus_routes[rid] = {
                    "id": rid,
                    "name": tg.get("name", f"Route_{rid}"),
                    "ref": tg.get("ref", f"R{rid}"),
                    "way_edges": way_edges
                }
                quality["extracted_route_relations"] += 1
                if route_gaps:
                    quality["discontinuous_route_relations"] += 1
                    quality["route_discontinuities"] += route_gaps
                else:
                    quality["continuous_route_relations"] += 1
            else:
                quality["excluded_route_relations"] += 1

        self.stats["route_quality"] = dict(quality)
        print(f"  [routes] {len(self.bus_routes)}")

    def update_sow(self):
        self.sow = {}
        for r in self.bus_routes.values():
            for e in r["way_edges"]:
                for st in e["visible_stops"]:
                    self.sow[st["node_id"]] = st

    # =====================================================
    # 网络构建
    # =====================================================
    def build_net_pre(self):
        tmp = nx.DiGraph()
        for r in self.bus_routes.values():
            for e in r["way_edges"]:
                tmp.add_edge(e["start_node"], e["end_node"])
                for st in e["original_stops"]:
                    tmp.add_node(st["node_id"])

        self.stats["nodes_before"] = tmp.number_of_nodes()
        self.stats["edges_before"] = tmp.number_of_edges()

    def build_net(self):
        edge_route_ids = defaultdict(set)
        edge_route_refs = defaultdict(set)
        edge_cache = {}

        for rid in sorted(self.bus_routes):
            r = self.bus_routes[rid]
            for e in r["way_edges"]:
                k = (e["start_node"], e["end_node"])
                edge_route_ids[k].add(rid)
                edge_route_refs[k].add(r["ref"])
                edge_cache.setdefault(k, e)

        endpoint_ids = sorted({n for edge in edge_cache for n in edge})
        for n in endpoint_ids:
            if n not in self.fn:
                continue
            nd = self.fn[n]
            stop = self.bus_stops.get(n)
            self.G.add_node(
                n,
                lat=nd["lat"],
                lon=nd["lon"],
                name=stop["name"] if stop else f"node_{n}",
                is_stop=bool(stop)
            )

        for u, v in sorted(edge_cache):
            e = edge_cache[(u, v)]
            poly = e["complete_polyline_coords"][:]
            stop_breakpoints = []
            stop_annotations = []

            for st in e["visible_stops"]:
                idx = inject_stop_to_polyline(poly, (st["lat"], st["lon"]))
                stop_breakpoints.append(idx)
                stop_annotations.append({
                    "node_id": st["node_id"],
                    "aggregate_id": st["aggregate_id"],
                    "name": st["name"],
                    "lat": st["lat"],
                    "lon": st["lon"]
                })

            rc = len(edge_route_ids[(u, v)])
            self.G.add_edge(
                u, v,
                weight=rc,
                route_count=rc,
                route_refs=list(edge_route_refs[(u, v)]),
                way_name=e["way_name"],
                polyline_coords=poly,
                stop_breakpoints=stop_breakpoints,
                stop_annotations=stop_annotations,
                way_id=e["way_id"]
            )

        self.stats["nodes_after"] = self.G.number_of_nodes()
        self.stats["edges_after"] = self.G.number_of_edges()
        self.stats["endpoint_nodes_before_gwcc"] = self.G.number_of_nodes()
        self.stats["endpoint_edges_before_gwcc"] = self.G.number_of_edges()
        print(f"  [net] V={self.G.number_of_nodes()} E={self.G.number_of_edges()}")

    def _ordered_stop_sequences(self):
        """Return relation-ordered stop-group sequences and an explicit quality audit."""
        sequences = []
        audit = Counter()
        for rid, rel in self.fr.items():
            if not self._is_bus_route(rel):
                continue
            audit["route_relations_considered"] += 1
            node_members = [m for m in rel.get("members", []) if m.get("type") == "node"]
            platform_members = [
                m for m in node_members
                if (m.get("role") or "").strip().lower().startswith("platform")
            ]
            stop_members = [
                m for m in node_members
                if (m.get("role") or "").strip().lower().startswith("stop")
            ]
            if platform_members:
                ordered_members = platform_members
                audit["platform_sequence_relations"] += 1
            elif stop_members:
                ordered_members = stop_members
                audit["stop_position_fallback_relations"] += 1
            elif node_members:
                ordered_members = node_members
                audit["untyped_node_fallback_relations"] += 1
            else:
                ordered_members = []
                audit["relations_without_node_members"] += 1

            audit["ordered_members_raw"] += len(ordered_members)
            sequence = []
            unmatched = 0
            duplicates = 0
            for member in ordered_members:
                aggregate_id = self.s2a.get(member.get("ref"))
                if aggregate_id is None:
                    unmatched += 1
                    continue
                if sequence and sequence[-1] == aggregate_id:
                    duplicates += 1
                    continue
                sequence.append(aggregate_id)
            audit["ordered_members_unmatched"] += unmatched
            audit["consecutive_duplicates_removed"] += duplicates
            audit["ordered_stop_groups_retained"] += len(sequence)
            if len(sequence) < 2:
                audit["relations_excluded_short_sequence"] += 1
                continue
            audit["usable_ordered_route_relations"] += 1
            sequences.append((rid, sequence))
        return sequences, dict(audit)

    def _build_stop_representation(self, representation):
        """Build a directed weighted L-space or P-space graph on shared stop groups."""
        if representation not in {"l_space", "p_space"}:
            raise ValueError("representation must be 'l_space' or 'p_space'")
        stop_graph = nx.DiGraph()
        edge_routes = defaultdict(set)
        sequences, audit = self._ordered_stop_sequences()
        for rid, sequence in sequences:
            pairs = pairwise(sequence) if representation == "l_space" else (
                (sequence[i], sequence[j])
                for i in range(len(sequence) - 1)
                for j in range(i + 1, len(sequence))
            )
            for u, v in pairs:
                if u != v:
                    edge_routes[(u, v)].add(rid)

        used_groups = {node for edge in edge_routes for node in edge}
        for aggregate_id in sorted(used_groups):
            group = self.ga[aggregate_id]
            stop_graph.add_node(
                aggregate_id,
                lat=float(group["center_lat"]),
                lon=float(group["center_lon"]),
                name=group["name"],
                is_stop=True,
                member_count=int(group["cluster_size"]),
            )
        for (u, v), route_ids in sorted(edge_routes.items()):
            stop_graph.add_edge(
                u,
                v,
                weight=len(route_ids),
                route_count=len(route_ids),
                route_ids=sorted(route_ids),
            )

        nodes_before_gwcc = stop_graph.number_of_nodes()
        edges_before_gwcc = stop_graph.number_of_edges()
        components_before_gwcc = 0
        if stop_graph.number_of_nodes():
            components = nx.connected_components(stop_graph.to_undirected())
            component_list = list(components)
            components_before_gwcc = len(component_list)
            largest = max(component_list, key=len)
            stop_graph = stop_graph.subgraph(largest).copy()

        stop_graph.graph.update({
            "representation": f"stop_{representation}",
            "ordered_route_relations": audit.get("usable_ordered_route_relations", 0),
            "sequence_audit": audit,
            "stop_group_threshold_m": self.dt,
            "nodes_before_gwcc": nodes_before_gwcc,
            "edges_before_gwcc": edges_before_gwcc,
            "components_before_gwcc": components_before_gwcc,
            "nodes_removed_by_gwcc": nodes_before_gwcc - stop_graph.number_of_nodes(),
            "edges_removed_by_gwcc": edges_before_gwcc - stop_graph.number_of_edges(),
        })
        return stop_graph

    def build_stop_graph(self):
        """Build directed weighted stop-based L-space from consecutive stops."""
        return self._build_stop_representation("l_space")

    def build_p_space_graph(self):
        """Build directed weighted P-space from all ordered direct-ride stop pairs."""
        return self._build_stop_representation("p_space")

    def apply_gwcc(self):
        if self.G.number_of_nodes() == 0:
            return

        cc = list(nx.connected_components(self.G.to_undirected()))
        nodes_before = self.G.number_of_nodes()
        edges_before = self.G.number_of_edges()
        if len(cc) <= 1:
            self.stats["gwcc"] = {
                "components_before": len(cc),
                "nodes_removed": 0,
                "edges_removed": 0,
                "nodes_retained_fraction": 1.0,
            }
            return

        largest = max(cc, key=len)
        removed = self.G.number_of_nodes() - len(largest)

        self.G = self.G.subgraph(largest).copy()
        valid_nodes = set(self.G.nodes())

        self.sow = {k: st for k, st in self.sow.items() if k in valid_nodes}
        self.csn &= valid_nodes

        self.stats["gwcc"] = {
            "components_before": len(cc),
            "nodes_removed": nodes_before - self.G.number_of_nodes(),
            "edges_removed": edges_before - self.G.number_of_edges(),
            "nodes_retained_fraction": self.G.number_of_nodes() / nodes_before if nodes_before else float("nan"),
        }
        self.stats["nodes_after"] = self.G.number_of_nodes()
        self.stats["edges_after"] = self.G.number_of_edges()

        print(f"  [gwcc] kept {self.G.number_of_nodes()}, rm {removed}")

    # =====================================================
    # 主流程
    # =====================================================
    def build(self):
        print(f"\n{'='*60}")
        print(f"  Building: {self.cn}")
        print(f"{'='*60}")

        self.get_city_boundary()
        self.load_data()
        self.filter_data()
        self.identify_stops()
        self.aggregate()
        self.extract_routes()
        self.update_sow()
        self.build_net_pre()
        self.build_net()
        self.apply_gwcc()

        print(f"  [DONE] V={self.G.number_of_nodes()} E={self.G.number_of_edges()}")
        return self.G



# %%
# %% Cell 4 — 统一指标计算（最终整洁版）

def _sampled_global_efficiency(G, sample_size=500, seed=42):
    """
    采样版 global_efficiency。
    对大图（>sample_size 个节点）只取 sample_size 个节点的子图来计算。
    误差通常在 ±1%~3%，但速度快 100 倍以上。
    """
    nodes = list(G.nodes())
    if len(nodes) <= sample_size:
        return nx.global_efficiency(G)

    rng = np.random.RandomState(seed)
    sample = rng.choice(nodes, size=sample_size, replace=False)
    sub = G.subgraph(sample).copy()
    return nx.global_efficiency(sub)


def _exact_global_efficiency(G):
    """Exact NetworkX-equivalent global efficiency, accelerated by NetworKit."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    if not NETWORKIT_OK:
        return nx.global_efficiency(G)

    node_index = {node: i for i, node in enumerate(G.nodes())}
    nk_graph = nk.Graph(n, weighted=False, directed=False)
    for u, v in G.edges():
        if u != v:
            nk_graph.addEdge(node_index[u], node_index[v])
    harmonic = nk.centrality.HarmonicCloseness(
        nk_graph, normalized=False
    ).run().scores()
    return float(sum(harmonic) / (n * (n - 1)))


def weighted_undirected_projection(graph):
    """Aggregate reciprocal directed edges by summing their route-overlap weights."""
    projected = nx.Graph()
    projected.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if projected.has_edge(u, v):
            projected[u][v]["weight"] += weight
        else:
            projected.add_edge(u, v, weight=weight)
    return projected


_METRIC_GRAPH_CACHE = {}


def _evaluation_graph_and_efficiency(graph, sample_efficiency=False):
    """Reuse the city-level projection and PL baseline across algorithms."""
    graph_key = id(graph)
    cached = _METRIC_GRAPH_CACHE.get(graph_key)
    if cached is None or cached[0] is not graph:
        cached = [graph, weighted_undirected_projection(graph), {}]
        _METRIC_GRAPH_CACHE[graph_key] = cached

    evaluation_graph = cached[1]
    efficiency_key = "sampled" if sample_efficiency else "exact"
    if efficiency_key not in cached[2]:
        if sample_efficiency:
            cached[2][efficiency_key] = _sampled_global_efficiency(evaluation_graph)
        else:
            cached[2][efficiency_key] = _exact_global_efficiency(evaluation_graph)
    return evaluation_graph, cached[2][efficiency_key]


def compute_standard_metrics(graph, community_dict, sample_efficiency=False):
    """
    统一后验指标计算（对所有算法使用同一口径）
    
    参数
    ----
    sample_efficiency : bool
        True  -> PL 用采样版 global_efficiency（快，适合 stability 等重复场景）
        False -> PL 用精确版 global_efficiency（慢，适合主结果）

    返回指标：
    - community_count : 社区数
    - Q               : 后验 modularity（undirected, weighted）
    - ACT             : 平均社区内部强度诊断量
    - ACD             : 平均社区边界暴露诊断量
    - ACBC            : 社区层图平均 betweenness centrality
    - CGCC            : 社区层图平均 clustering coefficient
    - PL              : 去除跨社区边后的效率损失比例
    """

    weight_key = "weight"
    nan = float("nan")
    evaluation_graph, overall_efficiency = _evaluation_graph_and_efficiency(
        graph, sample_efficiency=sample_efficiency
    )

    # =====================================================
    # 社区节点集合
    # =====================================================
    communities = defaultdict(set)
    for n, cid in community_dict.items():
        if cid != -1:
            communities[cid].add(n)

    community_sets = list(communities.values())
    if not community_sets:
        return {
            "community_count": 0,
            "Q": nan,
            "Q_directed": nan,
            "ACT": nan,
            "ACT_node_weighted": nan,
            "ACD": nan,
            "ACD_node_weighted": nan,
            "ACBC": nan,
            "CGCC": nan,
            "PL": nan,
            "community_size_min": nan,
            "community_size_median": nan,
            "community_size_max": nan,
            "largest_community_share": nan,
            "singleton_communities": 0,
            "communities_smaller_than_5": 0,
            "modularity_source": "posthoc_networkx_undirected_weighted"
        }

    community_sizes = np.array([len(nodes) for nodes in community_sets], dtype=int)
    assigned_node_count = int(community_sizes.sum())

    # =====================================================
    # 社区内部信息
    # =====================================================
    community_info = {}
    for cid, nodes in communities.items():
        internal_weight = sum(
            d.get(weight_key, 1)
            for u, v, d in graph.edges(data=True)
            if u in nodes and v in nodes
        )
        community_info[cid] = {
            "nodes": nodes,
            "node_count": len(nodes),
            "internal_weight": internal_weight
        }

    # =====================================================
    # 构造社区层图 CG
    # =====================================================
    CG = nx.DiGraph()
    CG.add_nodes_from(communities.keys())

    for u, v, d in graph.edges(data=True):
        cu = community_dict.get(u, -1)
        cv = community_dict.get(v, -1)

        if cu != cv and cu != -1 and cv != -1:
            ew = d.get(weight_key, 1)
            prev = CG.get_edge_data(cu, cv, {}).get(weight_key, 0)
            CG.add_edge(cu, cv, **{weight_key: prev + ew})

    # =====================================================
    # ACT: 平均内部强度
    # =====================================================
    act_values = []
    act_weights = []
    for cid, info in community_info.items():
        if info["node_count"] > 0:
            act_values.append(2 * info["internal_weight"] / info["node_count"])
            act_weights.append(info["node_count"])
    ACT = float(np.mean(act_values)) if act_values else nan
    ACT_node_weighted = float(np.average(act_values, weights=act_weights)) if act_values else nan

    # =====================================================
    # ACD: 平均边界暴露
    # =====================================================
    acd_values = []
    acd_weights = []
    for cid, info in community_info.items():
        Ein = info["internal_weight"]
        Eout = CG.in_degree(cid, weight=weight_key) + CG.out_degree(cid, weight=weight_key)
        denom = 2 * Ein + Eout
        if denom > 0:
            acd_values.append(Eout / denom)
            acd_weights.append(info["node_count"])
    ACD = float(np.mean(acd_values)) if acd_values else nan
    ACD_node_weighted = float(np.average(acd_values, weights=acd_weights)) if acd_values else nan

    # =====================================================
    # 社区层无向图：显式合并两个方向的跨社区强度
    # =====================================================
    CG_u = nx.Graph()
    CG_u.add_nodes_from(CG.nodes())
    for cu, cv, d in CG.edges(data=True):
        strength = d.get(weight_key, 1)
        if CG_u.has_edge(cu, cv):
            CG_u[cu][cv][weight_key] += strength
        else:
            CG_u.add_edge(cu, cv, **{weight_key: strength})
    for cu, cv, d in CG_u.edges(data=True):
        d["distance"] = 1.0 / d[weight_key]

    # =====================================================
    # ACBC: 社区层图平均 betweenness；连接越强，距离越短
    # =====================================================
    ACBC = nan
    if CG_u.number_of_nodes() >= 3:
        bc = nx.betweenness_centrality(
            CG_u,
            weight="distance",
            normalized=True
        )
        ACBC = float(np.mean(list(bc.values()))) if bc else nan

    # =====================================================
    # CGCC: 社区层图平均 clustering
    # =====================================================
    if CG_u.number_of_nodes() >= 3 and CG_u.number_of_edges() > 0:
        CGCC = float(nx.average_clustering(CG_u, weight=weight_key))
    else:
        CGCC = nan

    # Mean directional imbalance among connected community pairs. A value of
    # zero denotes balanced reciprocal flow and one denotes one-way flow.
    asymmetry_numerator = 0.0
    asymmetry_denominator = 0.0
    for cu, cv in CG_u.edges():
        forward = float(CG.get_edge_data(cu, cv, {}).get(weight_key, 0.0))
        backward = float(CG.get_edge_data(cv, cu, {}).get(weight_key, 0.0))
        total_pair_weight = forward + backward
        if total_pair_weight > 0:
            asymmetry_numerator += abs(forward - backward)
            asymmetry_denominator += total_pair_weight
    community_flow_asymmetry = (
        asymmetry_numerator / asymmetry_denominator if asymmetry_denominator > 0 else nan
    )

    # =====================================================
    # PL: 去除跨社区边后的效率损失
    # =====================================================
    try:
        Gu = evaluation_graph

        Gi = Gu.copy()
        inter_edges = [
            (u, v)
            for u, v in Gu.edges()
            if community_dict.get(u, -1) != community_dict.get(v, -1)
        ]
        Gi.remove_edges_from(inter_edges)

        if sample_efficiency:
            intra_efficiency = _sampled_global_efficiency(Gi)
        else:
            intra_efficiency = _exact_global_efficiency(Gi)

        PL = (overall_efficiency - intra_efficiency) / overall_efficiency if overall_efficiency > 0 else nan
    except Exception:
        PL = nan

    # =====================================================
    # Q: 后验统一 modularity
    # =====================================================
    try:
        Q = float(nx.community.modularity(
            evaluation_graph,
            community_sets,
            weight=weight_key
        ))
    except Exception:
        Q = nan

    # Leicht--Newman directed modularity, evaluated on the original directed
    # route-overlap graph rather than on the common undirected projection.
    try:
        total_directed_weight = float(sum(
            d.get(weight_key, 1.0) for _, _, d in graph.edges(data=True)
        ))
        if total_directed_weight <= 0:
            Q_directed = nan
        else:
            community_internal = defaultdict(float)
            community_out = defaultdict(float)
            community_in = defaultdict(float)
            for u, v, data in graph.edges(data=True):
                weight = float(data.get(weight_key, 1.0))
                cu = community_dict.get(u, -1)
                cv = community_dict.get(v, -1)
                if cu != -1:
                    community_out[cu] += weight
                if cv != -1:
                    community_in[cv] += weight
                if cu != -1 and cu == cv:
                    community_internal[cu] += weight
            observed_minus_expected = sum(
                community_internal[cid]
                - community_out[cid] * community_in[cid] / total_directed_weight
                for cid in communities
            )
            Q_directed = float(observed_minus_expected / total_directed_weight)
    except Exception:
        Q_directed = nan

    return {
        "community_count": len(communities),
        "community_size_min": int(community_sizes.min()),
        "community_size_median": float(np.median(community_sizes)),
        "community_size_max": int(community_sizes.max()),
        "largest_community_share": float(community_sizes.max() / assigned_node_count),
        "singleton_communities": int(np.sum(community_sizes == 1)),
        "communities_smaller_than_5": int(np.sum(community_sizes < 5)),
        "Q": Q,
        "Q_directed": Q_directed,
        "ACT": ACT,
        "ACT_node_weighted": ACT_node_weighted,
        "ACD": ACD,
        "ACD_node_weighted": ACD_node_weighted,
        "ACBC": ACBC,
        "CGCC": CGCC,
        "community_flow_asymmetry": community_flow_asymmetry,
        "PL": PL,
        "modularity_source": "posthoc_weight-summed_undirected_projection",
        "efficiency_source": "unweighted_topological_efficiency_on_undirected_projection"
    }



# %%
# =========================================================
# 1) 指标独立性分析
# =========================================================
def analyze_metric_independence(cities=None, algorithms=None):
    """
    分析六个指标间的相关性与可能冗余性。
    可按 cities / algorithms 过滤。
    输出：
    - Pearson 相关热力图
    - Spearman 相关热力图
    - pairwise summary csv
    - metric dimensions table
    """
    df = collect_results()
    if df.empty:
        print("No results found.")
        return None

    latest = df.drop_duplicates(subset=["city", "algorithm"], keep="last").copy()

    # ===== 过滤 =====
    if cities is not None:
        latest = latest[latest["city"].isin(cities)].copy()
    if algorithms is not None:
        latest = latest[latest["algorithm"].isin(algorithms)].copy()

    if latest.empty:
        print("No filtered results for metric independence.")
        return None

    metrics = ["Q", "ACT", "ACD", "ACBC", "CGCC", "PL"]
    metrics = [m for m in metrics if m in latest.columns]

    data = latest[metrics].apply(pd.to_numeric, errors="coerce")
    valid = data.dropna(how="all")
    if valid.shape[0] < 3:
        print("Not enough result rows for correlation.")
        return None

    pearson = valid.corr(method="pearson")
    spearman = valid.corr(method="spearman")

    suffix = ""
    if cities is not None or algorithms is not None:
        suffix = "_filtered"

    # Heatmaps
    for name, corr in [("metric_corr_pearson", pearson), ("metric_corr_spearman", spearman)]:
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        _apply_heatmap_style(
            fig,
            width=800,
            height=700,
            left=105,
            bottom=95,
            x_tickangle=-38,
            x_title="Metric",
            y_title="Metric",
        )
        _save_plotly_summary(fig, f"{name}{suffix}")

    # Pairwise summary
    rows = []
    for a, b in combinations(metrics, 2):
        p = pearson.loc[a, b]
        s = spearman.loc[a, b]
        ap = abs(p) if pd.notna(p) else np.nan
        if pd.isna(ap):
            label = "NA"
        elif ap < 0.30:
            label = "Low correlation"
        elif ap < 0.60:
            label = "Moderate correlation"
        else:
            label = "High correlation"
        rows.append({
            "metric_1": a,
            "metric_2": b,
            "pearson_r": p,
            "spearman_rho": s,
            "abs_pearson": ap,
            "interpretation": label
        })
    pair_df = pd.DataFrame(rows).sort_values("abs_pearson", ascending=True)
    pair_df.to_csv(
        SUMMARY_DIR / f"metric_pairwise_correlation_summary{suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    if not pair_df.empty:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in pair_df.columns],
                fill_color="#2c5a89",
                font=dict(color="white", size=12),
                align="left",
                height=36
            ),
            cells=dict(
                values=[
                    pair_df["metric_1"],
                    pair_df["metric_2"],
                    pair_df["pearson_r"].round(4),
                    pair_df["spearman_rho"].round(4),
                    pair_df["abs_pearson"].round(4),
                    pair_df["interpretation"]
                ],
                fill_color="white",
                align="left",
                height=32,
                font_size=11
            )
        )])
        fig.update_layout(
            title=None,
            width=1100, height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        _save_plotly_summary(fig, f"metric_pairwise_correlation_summary{suffix}")

    # Metric dimension table
    dim_df = pd.DataFrame({
        "Metric": ["Q", "ACT", "ACD", "ACBC", "CGCC", "PL"],
        "Primary dimension": [
            "Overall partition quality",
            "Intra-community cohesion",
            "Boundary separation",
            "Inter-community bridge role",
            "Community graph clustering",
            "Efficiency loss after removing inter-community links"
        ],
        "Level": ["Overall", "Intra", "Boundary", "Inter", "Inter", "Overall/Intra"],
        "Type": ["Diagnostic", "Diagnostic", "Diagnostic", "Diagnostic", "Diagnostic", "Diagnostic"]
    })
    dim_df.to_csv(
        SUMMARY_DIR / "metric_dimension_table.csv",
        index=False, encoding="utf-8-sig"
    )

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in dim_df.columns],
            fill_color="#2c5a89",
            font=dict(color="white", size=12),
            align="left",
            height=36
        ),
        cells=dict(
            values=[dim_df[c] for c in dim_df.columns],
            fill_color="white",
            align="left",
            height=32,
            font_size=11
        )
    )])
    fig.update_layout(
        title=None,
        width=1100, height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    _save_plotly_summary(fig, "metric_dimension_table")

    print("[OK] metric independence analysis saved.")
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pairwise": pair_df,
        "dimensions": dim_df,
        "latest": latest
    }

# =========================================================
# 2) 空间紧凑性分析
# =========================================================
def _community_spatial_stats(graph, community_dict, sample_limit=300, sampling_repeats=20):
    rows = []
    comm_nodes = defaultdict(list)

    for n, cid in community_dict.items():
        if cid == -1 or n not in graph.nodes:
            continue
        nd = graph.nodes[n]
        if "lat" in nd and "lon" in nd:
            comm_nodes[cid].append((n, nd["lat"], nd["lon"]))

    for cid, pts in comm_nodes.items():
        if len(pts) == 0:
            continue

        coords = [(lat, lon) for _, lat, lon in pts]
        n = len(coords)

        # ---------- repeated-sample mean pairwise geodesic distance ----------
        stable_cid = int(hashlib.sha256(str(cid).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(GLOBAL_CFG["random_seed"] + stable_cid)
        repeat_means = []
        repeat_count = sampling_repeats if n > sample_limit else 1
        for _ in range(repeat_count):
            if n > sample_limit:
                idx = rng.choice(np.arange(n), size=sample_limit, replace=False)
                coords_eval = [coords[i] for i in idx]
            else:
                coords_eval = coords
            dists = [
                haversine(coords_eval[i], coords_eval[j])
                for i in range(len(coords_eval))
                for j in range(i + 1, len(coords_eval))
            ]
            repeat_means.append(float(np.mean(dists)) if dists else 0.0)
        avg_intra_dist_m = float(np.mean(repeat_means))
        avg_intra_dist_ci_low_m = float(np.quantile(repeat_means, 0.025))
        avg_intra_dist_ci_high_m = float(np.quantile(repeat_means, 0.975))

        # Diameter is derived from all points through their projected convex
        # hull, avoiding the systematic downward bias of a sampled maximum.
        diameter_m = 0.0

        # ---------- convex hull / compactness in projected CRS ----------
        hull_area_m2 = np.nan
        hull_perimeter_m = np.nan
        compactness = np.nan

        try:
            # 构造点 GeoDataFrame (lon, lat)
            gdf = gpd.GeoDataFrame(
                geometry=[Point(lon, lat) for lat, lon in coords],
                crs="EPSG:4326"
            )
            projected_crs = gdf.estimate_utm_crs()
            if projected_crs is None:
                raise ValueError("Unable to estimate a local projected CRS")
            gdf = gdf.to_crs(projected_crs)

            hull = gdf.unary_union.convex_hull
            hull_area_m2 = float(hull.area)
            hull_perimeter_m = float(hull.length)

            if hull.geom_type == "Polygon":
                hull_points = np.asarray(hull.exterior.coords)
            elif hasattr(hull, "coords"):
                hull_points = np.asarray(hull.coords)
            else:
                hull_points = np.empty((0, 2))
            if len(hull_points) > 1:
                diameter_m = float(np.max(pdist(hull_points)))

            if hull_perimeter_m > 0:
                compactness = float(4 * np.pi * hull_area_m2 / (hull_perimeter_m ** 2))
        except Exception:
            pass

        node_ids = [node for node, _, _ in pts]
        induced = graph.subgraph(node_ids).to_undirected()
        components = list(nx.connected_components(induced)) if induced.number_of_nodes() else []
        component_count = len(components)
        largest_component_share = (
            max(map(len, components)) / len(node_ids) if components and node_ids else float("nan")
        )
        fragmentation_index = 1.0 - largest_component_share if pd.notna(largest_component_share) else float("nan")

        rows.append({
            "community": cid,
            "n_nodes_geo": n,
            "avg_intra_dist_m": avg_intra_dist_m,
            "avg_intra_dist_ci_low_m": avg_intra_dist_ci_low_m,
            "avg_intra_dist_ci_high_m": avg_intra_dist_ci_high_m,
            "distance_sampling_repeats": repeat_count,
            "diameter_m": diameter_m,
            "diameter_source": "all_point_projected_convex_hull",
            "convex_hull_area_m2": hull_area_m2,
            "convex_hull_perimeter_m": hull_perimeter_m,
            "compactness_ratio": compactness,
            "induced_component_count": component_count,
            "largest_component_share": largest_component_share,
            "fragmentation_index": fragmentation_index,
        })

    return pd.DataFrame(rows)


def run_spatial_compactness_comparison(cities=None, algorithms=None):
    """
    跨城市、跨算法对比空间紧凑性（投影版）。
    使用 community cache，避免重复跑算法。
    """
    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    al = ALGORITHMS if algorithms is None else [a for a in ALGORITHMS if a in algorithms]

    detail_rows = []
    summary_rows = []

    for cc in cl:
        city = cc["city"]
        graph, builder = get_graph(cc)
        if graph.number_of_nodes() == 0:
            continue

        for algo in al:
            try:
                community, _ = get_community(cc, algo)   # ===== 用缓存 =====
                cdf = _community_spatial_stats(graph, community)
                if cdf.empty:
                    continue

                cdf["city"] = city
                cdf["algorithm"] = algo
                detail_rows.append(cdf)

                def weighted_valid(column):
                    valid = cdf[column].notna() & cdf["n_nodes_geo"].gt(0)
                    if not valid.any():
                        return float("nan")
                    return float(np.average(cdf.loc[valid, column], weights=cdf.loc[valid, "n_nodes_geo"]))

                summary_rows.append({
                    "city": city,
                    "algorithm": algo,
                    "n_communities": int(cdf.shape[0]),
                    "avg_diameter_m": cdf["diameter_m"].mean(),
                    "median_diameter_m": cdf["diameter_m"].median(),
                    "avg_intra_dist_m": cdf["avg_intra_dist_m"].mean(),
                    "avg_compactness_ratio": cdf["compactness_ratio"].mean(skipna=True),
                    "avg_hull_area_m2": cdf["convex_hull_area_m2"].mean(skipna=True),
                    "avg_fragmentation_index": cdf["fragmentation_index"].mean(skipna=True),
                    "avg_largest_component_share": cdf["largest_component_share"].mean(skipna=True),
                    "node_weighted_diameter_m": weighted_valid("diameter_m"),
                    "node_weighted_intra_dist_m": weighted_valid("avg_intra_dist_m"),
                    "node_weighted_compactness_ratio": weighted_valid("compactness_ratio"),
                    "node_weighted_fragmentation_index": weighted_valid("fragmentation_index"),
                })
                print(f"[spatial] {city} × {algo} done.")

            except Exception as e:
                print(f"[spatial] {city} × {algo} FAILED: {e}")

    if not detail_rows:
        print("No spatial compactness results.")
        return None

    detail_df = pd.concat(detail_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    detail_df.to_csv(
        SUMMARY_DIR / "spatial_compactness_community_level.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        SUMMARY_DIR / "spatial_compactness_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # Heatmaps
    for met in ["avg_diameter_m", "avg_intra_dist_m", "avg_compactness_ratio"]:
        if met not in summary_df.columns:
            continue
        piv = summary_df.pivot(index="city", columns="algorithm", values=met)
        fig = go.Figure(data=go.Heatmap(
            z=piv.values,
            x=piv.columns.tolist(),
            y=piv.index.tolist(),
            colorscale="RdYlGn_r" if met in ["avg_diameter_m", "avg_intra_dist_m"] else "RdYlGn",
            text=np.round(piv.values, 3),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        _apply_heatmap_style(
            fig,
            width=900,
            height=500,
            left=120,
            bottom=105,
            x_title="Algorithm",
            y_title="City",
        )
        _save_plotly_summary(fig, f"spatial_compactness_{met}")

    # Summary table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in summary_df.columns],
            fill_color="#2c5a89",
            font=dict(color="white", size=12),
            align="left",
            height=36
        ),
        cells=dict(
            values=[
                summary_df["city"],
                summary_df["algorithm"],
                summary_df["n_communities"],
                summary_df["avg_diameter_m"].round(2),
                summary_df["median_diameter_m"].round(2),
                summary_df["avg_intra_dist_m"].round(2),
                summary_df["avg_compactness_ratio"].round(4),
                summary_df["avg_hull_area_m2"].round(2)
            ],
            fill_color="white",
            align="left",
            height=32,
            font_size=11
        )
    )])
    fig.update_layout(
        title=None,
        width=1200,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    _save_plotly_summary(fig, "spatial_compactness_summary_table")

    print("[OK] spatial compactness comparison saved.")
    return {
        "detail": detail_df,
        "summary": summary_df
    }

# =========================================================
# 3) 规模敏感性分析
# =========================================================
def analyze_scale_sensitivity(cities=None, algorithms=None):
    """
    分析 graph size 与指标之间的关系：
    - 节点数/边数 vs Q, ACT, ...
    - 按算法分组
    可按 cities / algorithms 过滤。
    """
    df = collect_results()
    if df.empty:
        print("No results found.")
        return None

    latest = df.drop_duplicates(subset=["city", "algorithm"], keep="last").copy()

    # ===== 过滤 =====
    if cities is not None:
        latest = latest[latest["city"].isin(cities)].copy()
    if algorithms is not None:
        latest = latest[latest["algorithm"].isin(algorithms)].copy()

    if latest.empty:
        print("No filtered results for scale sensitivity.")
        return None

    for c in ["graph_nodes", "graph_edges", "Q", "ACT", "ACD", "ACBC", "CGCC", "PL", "community_count"]:
        if c in latest.columns:
            latest[c] = pd.to_numeric(latest[c], errors="coerce")

    metrics = ["Q", "ACT", "ACD", "ACBC", "CGCC", "PL", "community_count"]
    suffix = ""
    if cities is not None or algorithms is not None:
        suffix = "_filtered"

    # Scatter panels: nodes vs each metric
    ncol = 3
    nr = int(np.ceil(len(metrics) / ncol))
    fig = make_subplots(rows=nr, cols=ncol, subplot_titles=metrics)
    color_manager = CommunityColorManager()
    algo_list = sorted(latest["algorithm"].dropna().unique().tolist())
    color_manager.assign_batch(list(range(len(algo_list))))
    algo_color = {a: color_manager.get_color(i) for i, a in enumerate(algo_list)}

    for i, met in enumerate(metrics):
        r = i // ncol + 1
        c = i % ncol + 1
        for algo in algo_list:
            sub = latest[latest["algorithm"] == algo]
            fig.add_trace(
                go.Scatter(
                    x=sub["graph_nodes"],
                    y=sub[met],
                    mode="markers+text",
                    text=sub["city"],
                    textposition="top center",
                    marker=dict(size=10, color=algo_color[algo], line=dict(width=0.5, color="black")),
                    name=algo,
                    legendgroup=algo,
                    showlegend=(i == 0)
                ),
                row=r, col=c
            )
        fig.update_xaxes(title_text="graph_nodes", row=r, col=c)
        fig.update_yaxes(title_text=met, row=r, col=c)

    fig.update_layout(
        title=None,
        width=1300, height=350 * nr,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    _save_plotly_summary(fig, f"scale_sensitivity_nodes_vs_metrics{suffix}")

    # Correlation summary within each algorithm
    rows = []
    for algo in algo_list:
        sub = latest[latest["algorithm"] == algo]
        if sub.shape[0] < 3:
            continue
        for met in metrics:
            x = sub["graph_nodes"]
            y = sub[met]
            ok = x.notna() & y.notna()
            if ok.sum() >= 3:
                r = x[ok].corr(y[ok], method="pearson")
                rows.append({
                    "algorithm": algo,
                    "size_variable": "graph_nodes",
                    "metric": met,
                    "pearson_r": r,
                    "abs_r": abs(r) if pd.notna(r) else np.nan
                })

            x2 = sub["graph_edges"]
            ok2 = x2.notna() & y.notna()
            if ok2.sum() >= 3:
                r2 = x2[ok2].corr(y[ok2], method="pearson")
                rows.append({
                    "algorithm": algo,
                    "size_variable": "graph_edges",
                    "metric": met,
                    "pearson_r": r2,
                    "abs_r": abs(r2) if pd.notna(r2) else np.nan
                })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(
        SUMMARY_DIR / f"scale_sensitivity_correlation_summary{suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    if not corr_df.empty:
        piv = corr_df.pivot_table(index=["algorithm", "size_variable"], columns="metric", values="pearson_r")
        fig = go.Figure(data=go.Heatmap(
            z=piv.values,
            x=piv.columns.tolist(),
            y=[f"{i[0]} | {i[1]}" for i in piv.index.tolist()],
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(piv.values, 3),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        _apply_heatmap_style(
            fig,
            annotation_size=15,
            width=1000,
            height=600,
            left=245,
            bottom=95,
            x_tickangle=-38,
            x_title="Metric",
            y_title="Algorithm | size variable",
        )
        _save_plotly_summary(fig, f"scale_sensitivity_correlation_heatmap{suffix}")

    print("[OK] scale sensitivity analysis saved.")
    return {
        "latest": latest,
        "corr": corr_df
    }

# =========================================================
# 4) 边界文档化
# =========================================================
def document_city_boundaries(cities=None):
    """
    输出各城市边界文档信息，并保存边界总览图。
    
    参数
    ----
    cities : list[str] or None
        若为 None，则处理全部城市；
        否则只处理指定英文城市名。
    """
    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    rows = []

    # Overview map
    overview = folium.Map(
        location=[35, 70], zoom_start=2,
        tiles=OSM_TILE_URL,
        attr="OpenStreetMap contributors"
    )
    cm = CommunityColorManager()
    cm.assign_batch(list(range(len(cl))))

    for i, cc in enumerate(cl):
        city = cc["city"]
        country = cc["country"]
        city_query = cc.get("city_query", city)

        try:
            graph, builder = get_graph(cc)
            geom = builder.city_boundary

            if geom is None:
                rows.append({
                    "city": city,
                    "country": country,
                    "city_query": city_query,
                    "boundary_ok": False,
                    "min_lon": np.nan,
                    "min_lat": np.nan,
                    "max_lon": np.nan,
                    "max_lat": np.nan,
                    "boundary_area_deg2": np.nan,
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges()
                })
                continue

            minx, miny, maxx, maxy = geom.bounds
            area = float(geom.area)

            rows.append({
                "city": city,
                "country": country,
                "city_query": city_query,
                "boundary_ok": True,
                "min_lon": minx,
                "min_lat": miny,
                "max_lon": maxx,
                "max_lat": maxy,
                "boundary_area_deg2": area,
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges()
            })

            if builder.boundary_coords:
                folium.Polygon(
                    locations=builder.boundary_coords,
                    color=cm.get_color(i),
                    weight=2,
                    fill=False,
                    popup=city
                ).add_to(overview)

        except Exception as e:
            print(f"[boundary] {city} FAILED: {e}")
            rows.append({
                "city": city,
                "country": country,
                "city_query": city_query,
                "boundary_ok": False,
                "min_lon": np.nan,
                "min_lat": np.nan,
                "max_lon": np.nan,
                "max_lat": np.nan,
                "boundary_area_deg2": np.nan,
                "graph_nodes": np.nan,
                "graph_edges": np.nan
            })

    bdf = pd.DataFrame(rows)
    suffix = "_filtered" if cities is not None else ""

    # CSV
    csv_path = SUMMARY_DIR / f"boundary_documentation{suffix}.csv"
    bdf.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Table figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in bdf.columns],
            fill_color="#2c5a89",
            font=dict(color="white", size=12),
            align="left",
            height=36
        ),
        cells=dict(
            values=[
                bdf[c].round(6) if pd.api.types.is_numeric_dtype(bdf[c]) else bdf[c]
                for c in bdf.columns
            ],
            fill_color="white",
            align="left",
            height=32,
            font_size=11
        )
    )])
    fig.update_layout(
        title=None,
        width=1400,
        height=420,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    _save_plotly_summary(fig, f"boundary_documentation_table{suffix}")

    # Overview map
    overview_path = SUMMARY_DIR / f"boundary_overview_map{suffix}.html"
    overview.save(str(overview_path))

    print(f"[OK] boundary CSV   : {csv_path}")
    print(f"[OK] boundary map   : {overview_path}")
    print("[OK] boundary documentation saved.")

    return bdf


# %%
# %% Cell 5 — 六种算法（最终整理版）

def run_infomap(graph, markov_time=1.0, seed=42, **kw):
    """
    Infomap on directed weighted graph.
    """
    if not INFOMAP_OK:
        raise ImportError("infomap")

    im = infomap.Infomap(
        silent=True,
        directed=True,
        seed=seed,
        markov_time=markov_time
    )

    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    for u, v, d in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1])):
        if u in node_to_idx and v in node_to_idx:
            im.add_link(node_to_idx[u], node_to_idx[v], d.get("weight", 1))

    im.run()

    community = {}
    for nd in im.tree:
        if nd.is_leaf and nd.node_id in idx_to_node:
            community[idx_to_node[nd.node_id]] = nd.module_id

    for n in node_list:
        community.setdefault(n, -1)

    return community, {
        "algorithm_parameters": {
            "markov_time": markov_time,
            "seed": seed,
            "flow_model": "directed"
        },
        "directed_used": True,
        "weight_used": True,
        "spatial_prior": False,
        "algorithm_group": "main"
    }

def run_leiden_rbc(graph, resolution=0.1, seed=35, **kw):
    """
    Leiden with RBConfigurationVertexPartition on directed weighted graph.
    """
    if not LEIDEN_OK:
        raise ImportError("leidenalg/igraph")

    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    edges = []
    weights = []

    for u, v, d in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1])):
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))
            weights.append(d.get("weight", 1.0))

    g = ig.Graph(n=len(node_list), edges=edges, directed=True)
    g.vs["name"] = node_list
    if weights:
        g.es["weight"] = weights

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
        weights="weight" if weights else None
    )

    community = {}
    for cid, members in enumerate(part):
        for idx in members:
            community[idx_to_node[idx]] = cid

    for n in node_list:
        community.setdefault(n, -1)

    return community, {
        "algorithm_parameters": {
            "resolution": resolution,
            "seed": seed,
            "partition_type": "RBConfigurationVertexPartition"
        },
        "directed_used": True,
        "weight_used": True,
        "spatial_prior": False,
        "algorithm_group": "main"
    }

def run_dcsbm(
    graph,
    seed=42,
    n_restarts=1,
    seed_increment=17,
    checkpoint_dir=None,
    restart_indices=None,
    restart_seed_overrides=None,
    **kw,
):
    """
    Degree-corrected stochastic block model via graph-tool.
    Weighted directed implementation with real-normal edge covariate.
    """
    if not GT_OK:
        raise ImportError("graph-tool")

    g = gt.Graph(directed=True)
    edge_weight = g.new_edge_property("double")
    g.edge_properties["weight"] = edge_weight

    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for i, n in enumerate(node_list)}

    g.add_vertex(len(node_list))

    for u, v, d in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1])):
        e = g.add_edge(g.vertex(node_to_idx[u]), g.vertex(node_to_idx[v]))
        edge_weight[e] = d.get("weight", 1.0)

    restart_records = []
    restart_communities = []
    best_description_length = float("inf")
    best_community = None
    best_seed = None
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    total_restarts = max(1, int(n_restarts))
    if restart_indices is None:
        selected_restart_indices = list(range(total_restarts))
    else:
        selected_restart_indices = sorted({int(index) for index in restart_indices})
        if not selected_restart_indices or selected_restart_indices[0] < 0 or selected_restart_indices[-1] >= total_restarts:
            raise ValueError(
                f"restart_indices must be non-empty and within [0, {total_restarts})"
            )
    seed_overrides = {
        int(index): int(value)
        for index, value in (restart_seed_overrides or {}).items()
    }
    for restart_index in selected_restart_indices:
        restart_seed = seed_overrides.get(
            restart_index, int(seed + restart_index * seed_increment)
        )
        restart_file = checkpoint_path / f"restart_{restart_index:03d}.npz" if checkpoint_path is not None else None
        resumed = False
        if restart_file is not None and restart_file.exists():
            saved = np.load(restart_file, allow_pickle=False)
            labels = saved["labels"]
            saved_nodes = int(saved["node_count"])
            saved_edges = int(saved["edge_count"])
            saved_seed = int(saved["seed"])
            if saved_nodes != len(node_list) or saved_edges != graph.number_of_edges() or saved_seed != restart_seed:
                raise ValueError(f"incompatible DCSBM checkpoint: {restart_file}")
            description_length = float(saved["description_length"])
            candidate = {idx_to_node[i]: int(labels[i]) for i in range(len(node_list))}
            resumed = True
        else:
            gt.seed_rng(restart_seed)
            state = gt.minimize_blockmodel_dl(
                g,
                state_args={
                    "deg_corr": True,
                    "recs": [edge_weight],
                    "rec_types": ["real-normal"]
                }
            )
            description_length = float(state.entropy())
            blocks = state.get_blocks()
            labels = np.asarray(
                [int(blocks[g.vertex(i)]) for i in range(g.num_vertices())],
                dtype=np.int32,
            )
            candidate = {idx_to_node[i]: int(labels[i]) for i in range(len(node_list))}
            if restart_file is not None:
                temporary_file = restart_file.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary_file,
                    labels=labels,
                    node_count=len(node_list),
                    edge_count=graph.number_of_edges(),
                    seed=restart_seed,
                    description_length=description_length,
                )
                os.replace(temporary_file, restart_file)
        block_count = len(set(candidate.values()))
        restart_records.append({
            "restart": restart_index,
            "seed": restart_seed,
            "description_length": description_length,
            "block_count": block_count,
        })
        restart_communities.append(candidate)
        if description_length < best_description_length:
            best_description_length = description_length
            best_community = candidate
            best_seed = restart_seed
        print(
            f"  [DCSBM] restart {restart_index + 1}/{total_restarts} "
            f"{'resumed' if resumed else 'saved'}: K={block_count}, DL={description_length:.3f}",
            flush=True,
        )

    pairwise_nmi = []
    if len(restart_communities) > 1:
        ordered_nodes = node_list
        label_vectors = [
            [candidate[node] for node in ordered_nodes]
            for candidate in restart_communities
        ]
        pairwise_nmi = [
            float(normalized_mutual_info_score(label_vectors[i], label_vectors[j]))
            for i in range(len(label_vectors))
            for j in range(i + 1, len(label_vectors))
        ]

    community = best_community or {}

    return community, {
        "algorithm_parameters": {
            "seed": seed,
            "selected_seed": best_seed,
            "n_restarts": total_restarts,
            "computed_restart_indices": selected_restart_indices,
            "seed_increment": seed_increment,
            "restart_seed_overrides": seed_overrides,
            "deg_corr": True,
            "edge_covariate": "weight",
            "rec_type": "real-normal",
            "model_type": "non_nested_directed_weighted_degree_corrected_sbm",
            "selection_rule": "minimum_description_length",
            "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
            "selected_description_length": best_description_length,
            "restart_records": restart_records,
            "restart_pairwise_nmi_mean": float(np.mean(pairwise_nmi)) if pairwise_nmi else float("nan"),
            "restart_pairwise_nmi_sd": float(np.std(pairwise_nmi, ddof=1)) if len(pairwise_nmi) > 1 else float("nan"),
            "restart_pairwise_nmi_min": float(np.min(pairwise_nmi)) if pairwise_nmi else float("nan"),
            "restart_pairwise_nmi_max": float(np.max(pairwise_nmi)) if pairwise_nmi else float("nan"),
        },
        "directed_used": True,
        "weight_used": True,
        "spatial_prior": False,
        "algorithm_group": "main"
    }

def run_geolp(
    graph,
    geographic_weight=0.7,
    bandwidth_m=5000.0,
    initial_communities=25,
    min_cluster_size=0.0,
    max_iter=100,
    seed=42,
    **kw,
):
    """Geographically weighted label propagation on a metric coordinate plane.

    The v1 implementation mixed angular coordinate differences with network
    weights, imposed a 15-community ceiling, and used a convergence test that
    could never succeed.  This implementation projects coordinates to a local
    equirectangular metric plane, exposes the spatial bandwidth and initial
    granularity, normalizes edge strength, and stops when labels no longer
    change.  Direction is intentionally collapsed by pooling predecessors and
    successors; this is reported in the returned metadata.
    """
    if not SKLEARN_OK:
        raise ImportError("sklearn")

    np.random.seed(seed)
    random.seed(seed)

    node_coords = {}
    for n in graph.nodes():
        nd = graph.nodes[n]
        lat, lon = nd.get("lat"), nd.get("lon")
        if lat is not None and lon is not None:
            try:
                lat, lon = float(lat), float(lon)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    node_coords[n] = (lat, lon)
            except Exception:
                pass

    if not 0.0 <= geographic_weight <= 1.0:
        raise ValueError("geographic_weight must be in [0, 1]")
    if bandwidth_m <= 0:
        raise ValueError("bandwidth_m must be positive")

    node_list = list(graph.nodes())
    if not node_list:
        return {}, {}

    lat0 = np.radians(np.mean([lat for lat, _ in node_coords.values()]))
    radius_m = 6371000.0
    metric_coords = {
        n: np.array([
            radius_m * np.radians(lon) * np.cos(lat0),
            radius_m * np.radians(lat),
        ])
        for n, (lat, lon) in node_coords.items()
    }
    coord_nodes = [n for n in node_list if n in metric_coords]
    n_clusters = max(1, min(int(initial_communities), len(coord_nodes)))
    labels = {}
    if coord_nodes:
        coord_array = np.vstack([metric_coords[n] for n in coord_nodes])
        initial = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(coord_array)
        labels.update({n: int(initial[i]) for i, n in enumerate(coord_nodes)})
    for n in node_list:
        labels.setdefault(n, random.randrange(max(n_clusters, 1)))

    reciprocal_strengths = {}
    for focal in graph.nodes():
        for neighbor in set(graph.predecessors(focal)) | set(graph.successors(focal)):
            pair = frozenset((focal, neighbor))
            if len(pair) < 2 or pair in reciprocal_strengths:
                continue
            strength = 0.0
            if graph.has_edge(focal, neighbor):
                strength += float(graph[focal][neighbor].get("weight", 1.0))
            if graph.has_edge(neighbor, focal):
                strength += float(graph[neighbor][focal].get("weight", 1.0))
            reciprocal_strengths[pair] = strength
    max_reciprocal_strength = max(reciprocal_strengths.values(), default=1.0)
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        changed = 0
        random.shuffle(node_list)
        for focal in node_list:
            neighbor_strength = defaultdict(float)
            for neighbor in set(graph.predecessors(focal)) | set(graph.successors(focal)):
                edge_strength = 0.0
                if graph.has_edge(focal, neighbor):
                    edge_strength += float(graph[focal][neighbor].get("weight", 1.0))
                if graph.has_edge(neighbor, focal):
                    edge_strength += float(graph[neighbor][focal].get("weight", 1.0))
                topology_term = edge_strength / max_reciprocal_strength
                if focal in metric_coords and neighbor in metric_coords:
                    distance_m = float(np.linalg.norm(metric_coords[focal] - metric_coords[neighbor]))
                    spatial_term = float(np.exp(-distance_m / bandwidth_m))
                else:
                    spatial_term = 0.0
                score = (1.0 - geographic_weight) * topology_term + geographic_weight * spatial_term
                neighbor_strength[labels[neighbor]] += score

            if not neighbor_strength:
                continue
            best_score = max(neighbor_strength.values())
            best_labels = sorted(label for label, score in neighbor_strength.items() if np.isclose(score, best_score))
            new_label = random.choice(best_labels)
            if new_label != labels[focal]:
                labels[focal] = new_label
                changed += 1
        if changed == 0:
            break

    node_label = labels

    groups = defaultdict(set)
    for n, l in node_label.items():
        groups[l].add(n)

    communities = list(groups.values())
    min_size = max(1, int(len(node_label) * min_cluster_size))

    large = [c for c in communities if len(c) >= min_size]
    small = [c for c in communities if len(c) < min_size]

    for sc in small:
        if not large:
            large.append(sc)
            continue

        sc_coords = [metric_coords[n] for n in sc if n in metric_coords]
        if not sc_coords:
            max(large, key=len).update(sc)
            continue

        sc_center = np.mean(sc_coords, axis=0)

        best_large = None
        best_dist = float("inf")
        for lc in large:
            lc_coords = [metric_coords[n] for n in lc if n in metric_coords]
            if not lc_coords:
                continue
            d = float(np.linalg.norm(sc_center - np.mean(lc_coords, axis=0)))
            if d < best_dist:
                best_dist = d
                best_large = lc

        (best_large or max(large, key=len)).update(sc)

    community = {}
    for cid, nodes in enumerate(large):
        for n in nodes:
            community[n] = cid

    for n in graph.nodes():
        community.setdefault(n, -1)

    return community, {
        "algorithm_parameters": {
            "geographic_weight": geographic_weight,
            "bandwidth_m": bandwidth_m,
            "initial_communities": initial_communities,
            "min_cluster_size": min_cluster_size,
            "max_iter": max_iter,
            "iterations_run": iterations,
            "converged": iterations < max_iter or changed == 0,
            "initial_cluster_count_realized": n_clusters,
            "final_community_count": len(large),
            "edge_strength_normalization": "reciprocal_sum_divided_by_global_max_reciprocal_sum",
            "seed": seed
        },
        "directed_used": False,
        "weight_used": True,
        "spatial_prior": True,
        "algorithm_group": "baseline",
        "note": "custom spatially informed baseline"
    }

def run_spinglass(graph, spins=25, seed=42, **kw):
    """
    Spinglass on undirected weighted projection.
    """
    if not IGRAPH_OK:
        raise ImportError("igraph")

    np.random.seed(seed)
    if hasattr(ig, "set_random_instance"):
        ig.set_random_instance(np.random.RandomState(seed))

    Gu = weighted_undirected_projection(graph)
    g = ig.Graph.from_networkx(Gu)

    part = g.community_spinglass(weights="weight", spins=spins)
    community = {g.vs[i]["_nx_name"]: part.membership[i] for i in range(g.vcount())}

    return community, {
        "algorithm_parameters": {
            "spins": spins,
            "seed": seed
        },
        "directed_used": False,
        "weight_used": True,
        "spatial_prior": False,
        "algorithm_group": "baseline",
        "note": "applied on the same reciprocal-weight-summed undirected projection used for post hoc evaluation"
    }

def directed_modularity(graph, community, resolution=1.0, weight="weight"):
    """Leicht--Newman directed modularity with an explicit resolution parameter."""
    total_weight = float(sum(data.get(weight, 1.0) for _, _, data in graph.edges(data=True)))
    if total_weight <= 0:
        return float("nan")
    community_internal = defaultdict(float)
    community_out = defaultdict(float)
    community_in = defaultdict(float)
    valid_communities = set()
    for node, label in community.items():
        if label != -1:
            valid_communities.add(label)
    for u, v, data in graph.edges(data=True):
        edge_weight = float(data.get(weight, 1.0))
        cu = community.get(u, -1)
        cv = community.get(v, -1)
        if cu != -1:
            community_out[cu] += edge_weight
        if cv != -1:
            community_in[cv] += edge_weight
        if cu != -1 and cu == cv:
            community_internal[cu] += edge_weight
    return float(sum(
        community_internal[label] / total_weight
        - resolution * community_out[label] * community_in[label] / (total_weight ** 2)
        for label in valid_communities
    ))


def run_dirlou(graph, resolution=0.1, seed=40, max_levels=5, max_iter=50, **kw):
    """
    Directed Louvain-style heuristic based on directed modularity gain
    (Leicht-Newman directed modularity with gamma/resolution).

    Experimental / exploratory only.
    """

    random.seed(seed)
    np.random.seed(seed)

    G = graph.copy()
    orig_to_cur = {n: n for n in G.nodes()}
    objective_trace = []
    level_trace = []

    def _precompute(g):
        total_weight = sum(d.get("weight", 1.0) for _, _, d in g.edges(data=True))

        in_deg = defaultdict(float)
        out_deg = defaultdict(float)

        for u, v, d in g.edges(data=True):
            w = d.get("weight", 1.0)
            out_deg[u] += w
            in_deg[v] += w

        for n in g.nodes():
            in_deg.setdefault(n, 0.0)
            out_deg.setdefault(n, 0.0)

        adj = {}
        for n in g.nodes():
            adj[n] = {
                "succ_w": {nb: g[n][nb].get("weight", 1.0) for nb in g.successors(n)},
                "pred_w": {nb: g[nb][n].get("weight", 1.0) for nb in g.predecessors(n)}
            }

        return total_weight, in_deg, out_deg, adj

    for level in range(max_levels):
        # 初始：当前层每个节点独立成社区
        comms = {n: n for n in G.nodes()}

        total_weight, in_deg, out_deg, adj = _precompute(G)
        if total_weight <= 0:
            break

        # 社区总入/出强度
        comm_in = defaultdict(float)
        comm_out = defaultdict(float)
        for n in G.nodes():
            c = comms[n]
            comm_in[c] += in_deg[n]
            comm_out[c] += out_deg[n]

        for _ in range(max_iter):
            improved = False
            nodes = list(G.nodes())
            random.shuffle(nodes)

            for n in nodes:
                current = comms[n]

                # 只考虑邻居所在社区
                candidate_comms = set()
                for nb in adj[n]["succ_w"]:
                    candidate_comms.add(comms[nb])
                for nb in adj[n]["pred_w"]:
                    candidate_comms.add(comms[nb])
                candidate_comms.add(current)

                if not candidate_comms:
                    continue

                # 先把 n 从当前社区临时移除
                comm_in[current] -= in_deg[n]
                comm_out[current] -= out_deg[n]

                best_comm = current
                best_gain = -float("inf")

                for cm in candidate_comms:
                    w_i_to_C = sum(
                        w for nb, w in adj[n]["succ_w"].items()
                        if nb != n and comms[nb] == cm
                    )
                    w_C_to_i = sum(
                        w for nb, w in adj[n]["pred_w"].items()
                        if nb != n and comms[nb] == cm
                    )

                    reward = (w_i_to_C + w_C_to_i) / total_weight
                    penalty = resolution * (
                        (out_deg[n] * comm_in[cm] + in_deg[n] * comm_out[cm]) / (total_weight ** 2)
                    )
                    gain = reward - penalty

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = cm

                # 恢复 / 或移动到最佳社区
                if best_comm != current:
                    comms[n] = best_comm
                    comm_in[best_comm] += in_deg[n]
                    comm_out[best_comm] += out_deg[n]
                    improved = True
                else:
                    comm_in[current] += in_deg[n]
                    comm_out[current] += out_deg[n]

            if not improved:
                break

        # 把原始节点映射到当前层社区
        orig_to_cur = {orig: comms.get(cur, cur) for orig, cur in orig_to_cur.items()}
        objective_value = directed_modularity(graph, orig_to_cur, resolution=resolution)
        objective_trace.append(objective_value)
        level_trace.append({
            "level": level,
            "input_nodes": G.number_of_nodes(),
            "input_edges": G.number_of_edges(),
            "community_count": len(set(orig_to_cur.values())),
            "directed_modularity": objective_value,
        })

        unique_comms = set(comms.values())
        if len(unique_comms) <= 1:
            break

        # 聚合生成下一层图
        newG = nx.DiGraph()
        newG.add_nodes_from(unique_comms)

        comm_weights = defaultdict(float)
        for u, v, d in G.edges(data=True):
            cu, cv = comms[u], comms[v]
            comm_weights[(cu, cv)] += d.get("weight", 1.0)

        for (cu, cv), w in comm_weights.items():
            newG.add_edge(cu, cv, weight=w)

        # 如果压缩幅度太小，就停止
        if newG.number_of_nodes() >= G.number_of_nodes() * 0.95:
            break

        G = newG

    community = orig_to_cur
    for n in graph.nodes():
        community.setdefault(n, -1)

    n_comms = len(set(v for v in community.values() if v != -1))
    if n_comms <= 1:
        print("[DirLou warning] trivial single-community partition detected.")

    return community, {
        "algorithm_parameters": {
            "resolution": resolution,
            "seed": seed,
            "max_levels": max_levels,
            "max_iter": max_iter,
            "objective": "directed_modularity_gamma",
            "objective_trace": objective_trace,
            "level_trace": level_trace,
            "objective_monotonic_non_decreasing": all(
                objective_trace[index] >= objective_trace[index - 1] - 1e-12
                for index in range(1, len(objective_trace))
            ),
            "note": "custom directed Louvain-style heuristic with directed modularity gain"
        },
        "directed_used": True,
        "weight_used": True,
        "spatial_prior": False,
        "algorithm_group": "experimental"
    }


ALGORITHM_REGISTRY = {
    "Infomap": run_infomap,
    "LeidenRBC": run_leiden_rbc,
    "DCSBM": run_dcsbm,
    "GeoLP": run_geolp,
    "Spinglass": run_spinglass,
    "DirLou": run_dirlou
}


# %%
# %% Cell 6 — 可视化函数（更新版）

def _map_center(g):
    la2 = [d['lat'] for _, d in g.nodes(data=True) if 'lat' in d]
    lo2 = [d['lon'] for _, d in g.nodes(data=True) if 'lon' in d]
    return [np.mean(la2), np.mean(lo2)] if la2 else [39.9, 116.4]

def _comm_info(g, cd):
    comms = defaultdict(set)
    for n, c in cd.items():
        if c != -1:
            comms[c].add(n)
    tw = sum(d.get('weight', 1) for _, _, d in g.edges(data=True)) or 1
    info = {}
    for cid, nodes in comms.items():
        iw = sum(d.get('weight', 1) for u, v, d in g.edges(data=True) if u in nodes and v in nodes)
        ow = sum(d.get('weight', 1) for u, v, d in g.edges(data=True) if u in nodes and v not in nodes)
        inw = sum(d.get('weight', 1) for u, v, d in g.edges(data=True) if u not in nodes and v in nodes)
        tc = iw + ow + inw
        info[cid] = {'nodes': nodes, 'size': len(nodes), 'iw': iw, 'ow': ow, 'inw': inw, 'tw': tc, 'pct': tc / tw * 100}
    sc = sorted(info.items(), key=lambda x: x[1]['tw'], reverse=True)
    o2n = {}
    ni = {}
    for nid, (oid, ci) in enumerate(sc, 1):
        o2n[oid] = nid
        ni[nid] = ci
    return ni, o2n

# ============================================
# 基础网络地图
# ============================================
def create_base_map(g, bc=None, stats=None):
    m = folium.Map(
        _map_center(g), zoom_start=11,
        tiles=OSM_TILE_URL,
        attr="OpenStreetMap contributors"
    )
    if bc:
        folium.Polygon(bc, color='red', weight=2.5, fill=False, dash_array='5').add_to(m)
    for u, v, d in g.edges(data=True):
        poly = d.get('polyline_coords', [])
        if not poly:
            continue
        folium.PolyLine(poly, color='green', weight=3, opacity=0.7).add_to(m)
        for idx in d.get('stop_breakpoints', []):
            if 0 <= idx < len(poly):
                folium.CircleMarker(poly[idx], radius=4, color='blue',
                                     fill=True, fill_opacity=0.9).add_to(m)
    return m

# ============================================
# 社区着色地图（核心：你要的效果）
# ============================================
def create_community_map(g, cd, bc=None, city="", algo=""):
    """
    社区着色地图：
    - 每条边按其起点所在社区着色
    - 每个站点按所在社区着色，大圆点
    - 城市边界红色
    - 不显示黑色跨社区边
    """
    center = _map_center(g)
    m = folium.Map(
        center, zoom_start=11,
        tiles=OSM_TILE_URL,
        attr="OpenStreetMap contributors"
    )

    # 城市边界
    if bc:
        folium.Polygon(bc, color='red', weight=2.5, fill=False, dash_array='5').add_to(m)

    # 社区信息 + 颜色
    ni, o2n = _comm_info(g, cd)
    cm = CommunityColorManager()
    cm.assign_batch(sorted(ni.keys()))

    # 建立节点坐标索引（用于站点着色）
    node_coords = {}
    for nid in g.nodes():
        nd = g.nodes[nid]
        if 'lat' in nd and 'lon' in nd:
            node_coords[nid] = (nd['lat'], nd['lon'])

    # ---- 按社区分组画边，同一社区的边放在同一 FeatureGroup ----
    # 先收集每个社区的边
    comm_edges = defaultdict(list)  # new_comm_id -> [(polyline, weight)]

    for u, v, d in g.edges(data=True):
        poly = d.get('polyline_coords', [])
        if not poly:
            continue

        cu = cd.get(u, -1)
        cv = cd.get(v, -1)

        # 按起点社区着色（即使跨社区也用起点颜色，不画黑色）
        if cu != -1:
            new_cu = o2n.get(cu, -1)
        elif cv != -1:
            new_cu = o2n.get(cv, -1)
        else:
            new_cu = -1

        comm_edges[new_cu].append({
            'poly': poly,
            'weight': d.get('weight', 1),
            'stops': d.get('stop_breakpoints', []),
        })

    # 画边
    for new_cid in sorted(comm_edges.keys()):
        color = cm.get_color(new_cid) if new_cid > 0 else '#808080'
        edges = comm_edges[new_cid]

        fg = folium.FeatureGroup(name=f"Community {new_cid}")

        for edge in edges:
            poly = edge['poly']
            w = edge['weight']
            line_weight = min(2 + w, 6)  # 根据权重调整线宽

            folium.PolyLine(
                poly, color=color, weight=line_weight,
                opacity=0.85
            ).add_to(fg)

        fg.add_to(m)

    # ---- 画站点（大圆点，按社区着色）----
    # 收集所有站点位置
    stop_coords_drawn = set()  # 去重

    for u, v, d in g.edges(data=True):
        poly = d.get('polyline_coords', [])
        if not poly:
            continue

        for idx in d.get('stop_breakpoints', []):
            if idx < 0 or idx >= len(poly):
                continue

            lat, lon = poly[idx]
            key = (round(lat, 5), round(lon, 5))  # 去重 key
            if key in stop_coords_drawn:
                continue
            stop_coords_drawn.add(key)

            # 找最近节点确定社区
            best_node = None
            best_dist = float('inf')
            for nid, (nlat, nlon) in node_coords.items():
                dd = (lat - nlat) ** 2 + (lon - nlon) ** 2
                if dd < best_dist:
                    best_dist = dd
                    best_node = nid

            if best_node is not None:
                raw_cid = cd.get(best_node, -1)
                new_cid = o2n.get(raw_cid, -1)
                stop_color = cm.get_color(new_cid) if new_cid > 0 else '#808080'
            else:
                stop_color = '#808080'
                new_cid = -1

            folium.CircleMarker(
                [lat, lon],
                radius=5,
                color=stop_color,
                fill=True,
                fill_color=stop_color,
                fill_opacity=0.95,
                weight=1,
            ).add_to(m)

    # ---- 图例 ----
    legend_html = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:12px 16px;border:2px solid #333;'
        'border-radius:8px;font-size:13px;max-height:400px;overflow-y:auto;">'
        f'<b>{city} - {algo}</b><br>'
        f'<b>{len(ni)} Communities</b><br><hr style="margin:4px 0;">'
    )
    for new_cid in sorted(ni.keys()):
        color = cm.get_color(new_cid)
        size = ni[new_cid]['size']
        pct = ni[new_cid]['pct']
        legend_html += (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};margin-right:6px;border:1px solid #333;'
            f'vertical-align:middle;border-radius:2px;"></span>'
            f'C{new_cid}: {size} nodes ({pct:.1f}%)<br>'
        )
    legend_html += (
        '<hr style="margin:4px 0;">'
        '<span style="color:red;font-weight:bold;">━━</span> City boundary'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # 添加图层控制
    folium.LayerControl().add_to(m)

    return m

# ============================================
# 社区网络结构图 (plotly)
# ============================================
def create_community_network_plot(g, cd, city="", algo=""):
    """
    零重影版社区网络结构图（Plotly）

    特点：
    - 节点代表社区
    - 节点大小 ~ 社区规模
    - 边表示社区间连接
    - 节点编号使用单层白字，无重影
    - 节点保留黑色边框以增强对比
    """

    ni, o2n = _comm_info(g, cd)
    if not ni:
        return None

    cm = CommunityColorManager()
    cm.assign_batch(sorted(ni.keys()))

    # ===== 构建社区层网络 =====
    CG = nx.DiGraph()
    CG.add_nodes_from(ni.keys())

    for u, v, d in g.edges(data=True):
        cu, cv = cd.get(u, -1), cd.get(v, -1)
        if cu != cv and cu != -1 and cv != -1:
            ncu, ncv = o2n.get(cu, -1), o2n.get(cv, -1)
            if ncu > 0 and ncv > 0:
                prev = CG.get_edge_data(ncu, ncv, {}).get("weight", 0)
                CG.add_edge(ncu, ncv, weight=prev + d.get("weight", 1))

    if CG.number_of_nodes() == 0:
        return None

    # ===== 布局 =====
    pos = nx.spring_layout(
        CG,
        k=2 / np.sqrt(max(len(CG), 1)),
        iterations=120,
        seed=30
    )

    # ===== 边宽缩放 =====
    wts = [d["weight"] for _, _, d in CG.edges(data=True)] if CG.number_of_edges() > 0 else [1]
    wmin, wmax = min(wts), max(wts)
    wr = wmax - wmin if wmax > wmin else 1

    traces = []
    annotations = []

    # ============================================
    # 画边
    # ============================================
    for u, v, d in CG.edges(data=True):
        if u not in pos or v not in pos:
            continue

        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = d["weight"]
        wd = 1.5 + 7 * (w - wmin) / wr if wr > 0 else 2.5

        # 边主线
        traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=wd, color="rgba(120,120,120,0.75)"),
            hoverinfo="text",
            text=f"C{u} → C{v}<br>Weight: {w:.0f}",
            showlegend=False
        ))

        # 箭头
        ax = x0 + 0.72 * (x1 - x0)
        ay = y0 + 0.72 * (y1 - y0)
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))

        traces.append(go.Scatter(
            x=[ax],
            y=[ay],
            mode="markers",
            marker=dict(
                symbol="triangle-right",
                size=max(10 + wd, 10),
                angle=ang,
                color="rgba(100,100,100,0.9)"
            ),
            hoverinfo="none",
            showlegend=False
        ))

    # ============================================
    # 画节点
    # ============================================
    for nid, info in ni.items():
        if nid not in pos:
            continue

        x, y = pos[nid]
        node_color = cm.get_color(nid)

        hover_text = (
            f"<b>Community {nid}</b><br>"
            f"Nodes: {info['size']}<br>"
            f"Internal weight: {info['iw']:.0f}<br>"
            f"Outgoing weight: {info['ow']:.0f}<br>"
            f"Incoming weight: {info['inw']:.0f}<br>"
            f"Total share: {info['pct']:.2f}%"
        )

        node_size = 40 + np.log1p(info["size"]) * 6

        traces.append(go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2.4, color="black")
            ),
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=False
        ))

        # ===== 单层白字，无描边，无重影 =====
        annotations.append(dict(
            x=x,
            y=y,
            text=f"<b>{nid}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(
                family="Arial Black",
                size=20,
                color="white"
            )
        ))

    # ============================================
    # 组装图
    # ============================================
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=None,
        showlegend=False,
        width=1200,
        height=800,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations
    )

    return fig


# ============================================
# 指标表格 (plotly)
# ============================================
def create_metrics_table(city, algo, metrics):
    h = ['<b>Category</b>', '<b>Metric</b>', '<b>Value</b>', '<b>Interpretation</b>']
    cats = ['Intra', 'Intra', 'Inter', 'Inter', 'Inter', 'Overall']
    ms = ['ACT', 'PL', 'ACD', 'ACBC', 'CGCC', 'Q']
    vs = [f"{metrics.get(m, float('nan')):.4f}" for m in ms]
    ips = ['Internal density (diagnostic)', 'Internal contribution (diagnostic)',
           'Boundary clarity (diagnostic)', 'Hub importance (diagnostic)',
           'Clustering effect (diagnostic)', 'Partition quality (diagnostic)']
    fig = go.Figure(data=[go.Table(
        header=dict(values=h, fill_color='#2c5a89',
                    font=dict(color='white', size=14), align='left', height=40),
        cells=dict(values=[cats, ms, vs, ips],
                   fill_color='white', align='left', height=38, font_size=13)
    )])
    fig.update_layout(
        title=None,
        width=1100, height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig



# %%
# %% Cell 6.5 — 论文静态导图模块（OSM + 公交站点圆点版 + 无轮廓线，PNG/PDF）
# ============================================
# 颜色工具（与社区颜色保持一致）
# ============================================
def _static_get_color(cid):
    if cid == -1:
        return "#808080"
    return CB_HEX[(cid - 1) % len(CB_HEX)]

# ============================================
# 把边界坐标转成 GeoDataFrame
# boundary_coords: [[lat, lon], ...]
# ============================================
def _boundary_to_gdf(boundary_coords, city_name="city"):
    if boundary_coords is None or len(boundary_coords) < 3:
        return None

    poly = Polygon([(lon, lat) for lat, lon in boundary_coords])
    gdf = gpd.GeoDataFrame({"city": [city_name]}, geometry=[poly], crs="EPSG:4326")
    return gdf

# ============================================
# 从 graph + community 提取线路 GeoDataFrame
# 线路按“起点社区”着色
# ============================================
def _graph_edges_to_gdf(graph, community_dict, o2n=None, color_mgr=None):
    """
    从 graph + community 提取线路 GeoDataFrame
    线路按“起点社区”着色，并可使用 o2n 重编号保持与 community_map 一致。
    """
    rows = []

    for u, v, d in graph.edges(data=True):
        poly = d.get("polyline_coords", [])
        if not poly or len(poly) < 2:
            continue

        cu = community_dict.get(u, -1)
        cv = community_dict.get(v, -1)

        # 与 community_map 保持一致：优先取起点社区；若起点无社区则用终点
        raw_cid = cu if cu != -1 else cv

        if raw_cid == -1:
            cid = -1
        else:
            cid = o2n.get(raw_cid, -1) if o2n is not None else raw_cid

        if cid == -1:
            color = "#808080"
        elif color_mgr is not None:
            color = color_mgr.get_color(cid)
        else:
            color = _static_get_color(cid)

        try:
            geom = LineString([(lon, lat) for lat, lon in poly])
            rows.append({
                "u": u,
                "v": v,
                "community": cid,
                "color": color,
                "weight": d.get("weight", 1),
                "geometry": geom
            })
        except Exception:
            continue

    if not rows:
        return gpd.GeoDataFrame(
            columns=["community", "color", "weight", "geometry"],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ============================================
# 从 graph + community 提取节点 GeoDataFrame
# 默认只取公交站点（is_stop=True）
# ============================================
def _graph_nodes_to_gdf(graph, community_dict, stops_only=True, o2n=None, color_mgr=None):
    """
    从 graph + community 提取节点 GeoDataFrame
    默认只取公交站点（is_stop=True）
    并可使用 o2n 重编号保持与 community_map 一致。
    """
    rows = []

    for n, raw_cid in community_dict.items():
        if n not in graph.nodes:
            continue

        nd = graph.nodes[n]

        if "lat" not in nd or "lon" not in nd:
            continue

        if stops_only and not nd.get("is_stop", False):
            continue

        if raw_cid == -1:
            cid = -1
        else:
            cid = o2n.get(raw_cid, -1) if o2n is not None else raw_cid

        if cid == -1:
            color = "#808080"
        elif color_mgr is not None:
            color = color_mgr.get_color(cid)
        else:
            color = _static_get_color(cid)

        try:
            geom = Point(nd["lon"], nd["lat"])
            rows.append({
                "node": n,
                "community": cid,
                "color": color,
                "is_stop": nd.get("is_stop", False),
                "geometry": geom
            })
        except Exception:
            continue

    if not rows:
        return gpd.GeoDataFrame(
            columns=["node", "community", "color", "is_stop", "geometry"],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

def _graph_breakpoints_to_gdf(graph, community_dict, o2n=None, color_mgr=None):
    """
    从每条边的 stop_breakpoints 生成站点 GeoDataFrame。
    这样静态图中的站点来源与 community_map 完全一致。

    颜色逻辑：
    - 与 community_map 保持一致
    - 优先按边起点社区着色
    """
    rows = []
    seen = set()

    for u, v, d in graph.edges(data=True):
        poly = d.get("polyline_coords", [])
        if not poly:
            continue

        cu = community_dict.get(u, -1)
        cv = community_dict.get(v, -1)

        # 与 community_map 保持一致：优先起点社区
        raw_cid = cu if cu != -1 else cv

        if raw_cid == -1:
            cid = -1
        else:
            cid = o2n.get(raw_cid, -1) if o2n is not None else raw_cid

        if cid == -1:
            color = "#808080"
        elif color_mgr is not None:
            color = color_mgr.get_color(cid)
        else:
            color = _static_get_color(cid)

        for idx in d.get("stop_breakpoints", []):
            if idx < 0 or idx >= len(poly):
                continue

            lat, lon = poly[idx]

            # 去重：避免多个边重复画同一个站点
            key = (round(lat, 6), round(lon, 6), cid)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "community": cid,
                "color": color,
                "geometry": Point(lon, lat)
            })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["community", "color", "geometry"],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ============================================
# 给城市边界外区域加遮罩
# 让图只保留城市轮廓以内
# ============================================
def _add_polygon_mask(ax, polygon_3857, facecolor="white"):
    if polygon_3857 is None:
        return

    minx, miny, maxx, maxy = polygon_3857.bounds

    outer = [
        (minx - 1e6, miny - 1e6),
        (maxx + 1e6, miny - 1e6),
        (maxx + 1e6, maxy + 1e6),
        (minx - 1e6, maxy + 1e6),
        (minx - 1e6, miny - 1e6),
    ]

    if polygon_3857.geom_type == "Polygon":
        inner = list(polygon_3857.exterior.coords)
    else:
        largest = max(polygon_3857.geoms, key=lambda g: g.area)
        inner = list(largest.exterior.coords)

    vertices = outer + inner[::-1]
    codes = (
        [MplPath.MOVETO] +
        [MplPath.LINETO] * (len(outer) - 2) +
        [MplPath.CLOSEPOLY] +
        [MplPath.MOVETO] +
        [MplPath.LINETO] * (len(inner) - 2) +
        [MplPath.CLOSEPOLY]
    )

    path = MplPath(vertices, codes)
    patch = PathPatch(path, facecolor=facecolor, edgecolor="none", zorder=2.5)
    ax.add_patch(patch)

# ============================================
# 保存静态论文图（OSM底图 + 公交站点圆点版 + 默认无轮廓线）
# ============================================
def save_static_boundary_map(
    graph,
    community_dict,
    boundary_coords,
    city,
    algorithm,
    output_root=RESULTS_DIR,
    dpi=None,
    width_mm=None,                              # 85=单栏, 120=中等, 180=双栏
    add_basemap=True,
    basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
    clip_to_boundary=True,
    show_nodes=True,
    stops_only=True,
    node_size=6,
    line_alpha=0.92,
    show_boundary=False,                       # 默认不显示轮廓线
    boundary_color="red",
    boundary_linewidth=1.2,
    line_width_base=1.0,
    line_width_scale=0.25,
    mask_outside=True,
    facecolor="white",
    transparent=False
):
    if width_mm is None:
        width_mm = get_figure_width_mm("map")
    if dpi is None:
        dpi = FIGURE_CFG.get("dpi", 600)

    # =====================================================
    # 与 community_map 保持一致的社区重编号和颜色映射
    # =====================================================
    ni, o2n = _comm_info(graph, community_dict)
    color_mgr = CommunityColorManager()
    color_mgr.assign_batch(sorted(ni.keys()))

    # ---------- 输出目录 ----------
    fig_dir = Path(output_root) / city / algorithm / "static_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 边界 ----------
    boundary_gdf = _boundary_to_gdf(boundary_coords, city_name=city)
    if boundary_gdf is None:
        print(f"[static map] boundary invalid for {city}-{algorithm}")
        return None

    # ---------- 边 ----------
    edges_gdf = _graph_edges_to_gdf(
        graph,
        community_dict,
        o2n=o2n,
        color_mgr=color_mgr
    )
    if edges_gdf.empty:
        print(f"[static map] no edges for {city}-{algorithm}")
        return None

    # ---------- 点（用 stop_breakpoints，与 community_map 一致） 
    nodes_gdf = _graph_breakpoints_to_gdf(
        graph,
        community_dict,
        o2n=o2n,
        color_mgr=color_mgr
    ) if show_nodes else None

    # ---------- 投影到 Web Mercator ----------
    boundary_3857 = boundary_gdf.to_crs(epsg=3857)
    edges_3857 = edges_gdf.to_crs(epsg=3857)
    nodes_3857 = nodes_gdf.to_crs(epsg=3857) if nodes_gdf is not None and not nodes_gdf.empty else None

    boundary_geom = boundary_3857.geometry.iloc[0]

    # ---------- 裁切 ----------
    if clip_to_boundary:
        try:
            edges_3857 = gpd.clip(edges_3857, boundary_3857)
        except Exception as e:
            print(f"[static map] edge clip failed: {e}")
        if nodes_3857 is not None:
            try:
                nodes_3857 = gpd.clip(nodes_3857, boundary_3857)
            except Exception as e:
                print(f"[static map] node clip failed: {e}")

    # ---------- 图尺寸 ----------
    minx, miny, maxx, maxy = boundary_geom.bounds
    width_in = width_mm / 25.4
    aspect = (maxy - miny) / (maxx - minx) if (maxx - minx) > 0 else 1.0
    height_in = width_in * aspect

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor=facecolor)

    # 预设显示范围
    pad_x = (maxx - minx) * 0.01
    pad_y = (maxy - miny) * 0.01
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # ---------- OSM 底图 ----------
    if add_basemap:
        try:
            ctx.add_basemap(
                ax,
                source=basemap_provider,
                headers=OSM_TILE_HEADERS,
                attribution=False
            )
        except Exception as e:
            print(f"[static map] basemap failed: {e}")

    # ---------- 边界外遮罩 ----------
    if mask_outside:
        _add_polygon_mask(ax, boundary_geom, facecolor=facecolor)

    # ---------- 画线路 ----------
    if not edges_3857.empty:
        for color in edges_3857["color"].dropna().unique():
            sub = edges_3857[edges_3857["color"] == color]
            if sub.empty:
                continue

            avg_weight = sub["weight"].mean() if "weight" in sub.columns else 1
            lw = line_width_base + line_width_scale * avg_weight
            lw = min(max(lw, 0.8), 4.0)

            sub.plot(
                ax=ax,
                color=color,
                linewidth=lw,
                alpha=line_alpha,
                zorder=3
            )

    # ---------- 画公交站点圆点 ----------
    if show_nodes and nodes_3857 is not None and not nodes_3857.empty:
        for color in nodes_3857["color"].dropna().unique():
            sub = nodes_3857[nodes_3857["color"] == color]
            if sub.empty:
                continue
            sub.plot(
                ax=ax,
                color=color,
                markersize=node_size,
                marker='o',
                alpha=0.95,
                edgecolor='none',
                linewidth=0,
                zorder=4
            )

    # ---------- 边界线（可选） ----------
    if show_boundary:
        boundary_3857.boundary.plot(
            ax=ax,
            color=boundary_color,
            linewidth=boundary_linewidth,
            zorder=5
        )

    # ---------- 去边框/坐标 ----------
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # ---------- 保存 ----------
    base_name = f"{city}_{algorithm}_boundary_map"
    png_path = fig_dir / f"{base_name}.png"
    pdf_path = fig_dir / f"{base_name}.pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=facecolor,
        transparent=transparent
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=facecolor,
        transparent=transparent
    )

    plt.close(fig)

    print(f"[OK] saved static map: {png_path}")
    print(f"[OK] saved static map: {pdf_path}")

    return {
        "png": png_path,
        "pdf": pdf_path
    }


# ============================================
# 单独导出一个城市×算法的静态图
# ============================================
def export_static_map_for_one(city_cfg, algo_name, show_nodes=True, node_size=6, show_boundary=False):
    city = city_cfg["city"]
    graph, builder = get_graph(city_cfg)

    community, _ = get_community(city_cfg, algo_name)   # ===== 用缓存 =====

    return save_static_boundary_map(
        graph=graph,
        community_dict=community,
        boundary_coords=builder.boundary_coords,
        city=city,
        algorithm=algo_name,
        output_root=RESULTS_DIR,
        dpi=FIGURE_CFG.get("dpi", 600),
        width_mm=get_figure_width_mm("map"),
        add_basemap=True,
        basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
        clip_to_boundary=True,
        show_nodes=show_nodes,
        stops_only=True,
        node_size=node_size,
        show_boundary=show_boundary,
        transparent=False
    )


# ============================================
# 批量导出静态图
# ============================================
def export_static_maps_batch(cities=None, algorithms=None, show_nodes=True, node_size=6, show_boundary=False):
    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    al = ALGORITHMS if algorithms is None else [a for a in ALGORITHMS if a in algorithms]

    print(f"\n{'#'*60}")
    print(f"  Static Map Export (OSM + Stops + No Boundary): {len(cl)} cities × {len(al)} algorithms")
    print(f"{'#'*60}")

    ok, fail = 0, 0
    for cc in cl:
        for algo in al:
            try:
                print(f"\n  Exporting static map: {cc['city']} × {algo}")
                export_static_map_for_one(
                    cc,
                    algo,
                    show_nodes=show_nodes,
                    node_size=node_size,
                    show_boundary=show_boundary
                )
                ok += 1
            except Exception as e:
                print(f"  FAILED: {cc['city']} × {algo}: {e}")
                traceback.print_exc()
                fail += 1

    print(f"\n[Static export done] ok={ok}, fail={fail}")


# %%
# %% Cell 7 — POI
NOISE_TYPES={"toilets","waste_basket","parking","parking_space","parking_entrance","bench","shelter",
"telephone","drinking_water","fountain","fire_hydrant","post_box","vending_machine","clock",
"bbq","hunting_stand","loading_dock","give_box","watering_place","compressed_air","vacuum_cleaner"}

CAT_MAP={
"restaurant":"Food","fast_food":"Food","cafe":"Food","pub":"Food","bar":"Food","bakery":"Food",
"ice_cream":"Food","butcher":"Food","beverages":"Food","confectionery":"Food","deli":"Food",
"greengrocer":"Food","seafood":"Food","wine":"Food","alcohol":"Food","tea":"Food","coffee":"Food",
"supermarket":"Shopping","convenience":"Shopping","department_store":"Shopping","mall":"Shopping",
"clothes":"Shopping","shoes":"Shopping","jewelry":"Shopping","electronics":"Shopping",
"mobile_phone":"Shopping","hardware":"Shopping","furniture":"Shopping","gift":"Shopping",
"stationery":"Shopping","books":"Shopping","toys":"Shopping","sports":"Shopping","cosmetics":"Shopping",
"beauty":"Shopping","optician":"Shopping","florist":"Shopping","pet":"Shopping","marketplace":"Shopping",
"variety_store":"Shopping","general":"Shopping","kiosk":"Shopping",
"hospital":"Healthcare","clinic":"Healthcare","pharmacy":"Healthcare","dentist":"Healthcare",
"doctors":"Healthcare","veterinary":"Healthcare","doctor":"Healthcare","medical_supply":"Healthcare",
"school":"Education","university":"Education","college":"Education","kindergarten":"Education",
"library":"Education","museum":"Education","theatre":"Education","cinema":"Education",
"arts_centre":"Education","community_centre":"Education","gallery":"Education",
"bank":"Finance","atm":"Finance","bureau_de_change":"Finance","post_office":"Finance",
"insurance":"Finance","estate_agent":"Finance","travel_agent":"Finance","lawyer":"Finance",
"it":"Finance","company":"Finance","government":"Finance",
"park":"Recreation","playground":"Recreation","sports_centre":"Recreation","swimming_pool":"Recreation",
"fitness_centre":"Recreation","stadium":"Recreation","pitch":"Recreation","golf_course":"Recreation",
"garden":"Recreation","nature_reserve":"Recreation","marina":"Recreation","dance":"Recreation",
"nightclub":"Recreation","sauna":"Recreation","massage":"Recreation","spa":"Recreation",
"hotel":"Tourism","guest_house":"Tourism","hostel":"Tourism","motel":"Tourism",
"attraction":"Tourism","viewpoint":"Tourism","information":"Tourism",
"place_of_worship":"Religion","monastery":"Religion",
"fuel":"Transport","car_repair":"Transport","car_wash":"Transport","car":"Transport",
"bicycle":"Transport","charging_station":"Transport","bus_station":"Transport",
"taxi":"Transport","ferry_terminal":"Transport",
"laundry":"Residential","dry_cleaning":"Residential","hairdresser":"Residential",
"tailor":"Residential","copyshop":"Residential","locksmith":"Residential","repair":"Residential",
"plumber":"Residential","electrician":"Residential","carpenter":"Residential","photographer":"Residential"}

# Correct broad classes that were conflated in v1 (for example, an unknown
# amenity was previously labelled "Finance" and a craft "Residential").
for _raw in ["library", "museum", "theatre", "cinema", "arts_centre", "community_centre", "gallery"]:
    CAT_MAP[_raw] = "Culture"
for _raw in ["post_office", "government"]:
    CAT_MAP[_raw] = "Public services"
for _raw in ["estate_agent", "travel_agent", "lawyer", "it", "company"]:
    CAT_MAP[_raw] = "Professional services"
for _raw in ["laundry", "dry_cleaning", "hairdresser", "tailor", "copyshop", "locksmith", "repair",
             "plumber", "electrician", "carpenter", "photographer"]:
    CAT_MAP[_raw] = "Personal services"

FIELD_FALLBACK={"amenity":"Other services","shop":"Shopping","healthcare":"Healthcare",
                "leisure":"Recreation","office":"Professional services","craft":"Personal services"}

def _poi_cat(row):
    for f in ["amenity","shop","healthcare","leisure","office","craft"]:
        v=row.get(f)
        if v and isinstance(v,str) and v.strip():
            v=v.strip().lower()
            if v in NOISE_TYPES:return None
            if v in CAT_MAP:return CAT_MAP[v]
            return FIELD_FALLBACK.get(f)
    return None

_POI_CACHE={}
def load_poi(city):
    if city in _POI_CACHE:return _POI_CACHE[city]
    pf=CITY_POI_FILES.get(city)
    if not pf or not Path(pf).exists():
        print(f"  [poi] No file for {city}");_POI_CACHE[city]=pd.DataFrame();return _POI_CACHE[city]
    with open(pf,'r',encoding='utf-8') as f:raw=json.load(f)
    rows=[]
    for it in raw:
        la2,lo2=it.get("lat"),it.get("lon")
        if la2 is None or lo2 is None:continue
        cat=_poi_cat(it)
        if cat is None:continue
        rt=None
        for ff in ["amenity","shop","healthcare","leisure","office","craft"]:
            vv=it.get(ff)
            if vv and isinstance(vv,str) and vv.strip():rt=vv.strip().lower();break
        rows.append({"lat":float(la2),"lon":float(lo2),"category":cat,"raw_type":rt,"name":it.get("name","")})
    df=pd.DataFrame(rows)
    print(f"  [poi] {city}: {len(raw)} raw → {len(df)} valid, {df['category'].nunique() if not df.empty else 0} cats")
    _POI_CACHE[city]=df;return df

_POI_NEAREST_CACHE = {}


def match_pois_to_nodes(g, pdf):
    """Return graph-independent nearest-node matches, cached per graph/POI frame."""
    cache_key = (id(g), id(pdf))
    cached = _POI_NEAREST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    nids=[];ncoords=[]
    for n in g.nodes():
        nd=g.nodes[n]
        if 'lat' in nd and 'lon' in nd:nids.append(n);ncoords.append([nd['lat'],nd['lon']])
    if not ncoords:
        r=pdf.copy();r['nearest_node']=pd.NA;r['nearest_distance_m']=float('inf')
        _POI_NEAREST_CACHE[cache_key]=r
        return r
    tree=BallTree(np.radians(np.asarray(ncoords)),metric='haversine')
    pc=np.radians(pdf[['lat','lon']].values);dists,idxs=tree.query(pc,k=1)
    idxs=idxs[:,0];dists_m=dists[:,0]*6371000.0
    r=pdf.copy();r['nearest_node']=[nids[i] for i in idxs];r['nearest_distance_m']=dists_m
    _POI_NEAREST_CACHE[cache_key]=r
    return r


def apply_poi_distance_threshold(assigned, max_distance_m):
    """Apply a distance threshold to preassigned POIs without another tree query."""
    r=assigned.copy()
    if max_distance_m is not None:
        r.loc[r['nearest_distance_m'] > float(max_distance_m), 'community'] = -1
    r['matched'] = r['community'] != -1
    return r


def assign_pois(g,cd,pdf,max_distance_m=None,nearest_matches=None):
    nearest = match_pois_to_nodes(g,pdf) if nearest_matches is None else nearest_matches
    r=nearest.copy()
    r['community']=[cd.get(node,-1) if pd.notna(node) else -1 for node in r['nearest_node']]
    return apply_poi_distance_threshold(r,max_distance_m)

def compute_poi_metrics(pa,cd):
    df=pa[pa['community']!=-1].copy()
    if df.empty:return pd.DataFrame(),{}
    csz=Counter(v for v in cd.values() if v!=-1);K=df['category'].nunique()
    rows=[]
    for cid in sorted(df['community'].unique()):
        cp=df[df['community']==cid];n=len(cp)
        if n==0:continue
        cc=cp['category'].value_counts();pr=cc.values/cc.values.sum()
        H=float(scipy_entropy(pr,base=2));Hmax=np.log2(K) if K>1 else 1.0
        Hn=H/Hmax if Hmax>0 else 0.0;simp=float(np.sum(pr**2))
        rows.append({"community":cid,"n_pois":n,"n_cats":len(cc),"shannon":H,"norm_entropy":Hn,
            "simpson":simp,"dominant":cc.index[0],"dom_share":cc.values[0]/n,"specialization":1-Hn,
            "density":n/csz.get(cid,1)})
    cdf=pd.DataFrame(rows)
    if cdf.empty:return cdf,{}
    tp=cdf['n_pois'].sum();w=cdf['n_pois']/tp
    s={"total_pois":int(tp),"n_comms":len(cdf),"K":K,
       "entropy_w":float(np.average(cdf['norm_entropy'],weights=w)),
       "entropy_uw":float(cdf['norm_entropy'].mean()),
       "std_entropy":float(cdf['norm_entropy'].std()),
       "spec_w":float(np.average(cdf['specialization'],weights=w)),
       "simpson_w":float(np.average(cdf['simpson'],weights=w)),
       "dom_share":float(cdf['dom_share'].mean()),"density":float(cdf['density'].mean())}
    return cdf,s


def poi_entropy_permutation_test(
    pa,
    community_dict,
    n_permutations=9999,
    seed=42,
    graph=None,
    spatial_block_m=None,
    permutation_batch_size=64,
):
    """Global or spatially blocked node-label null for POI entropy.

    The null preserves the number of graph nodes assigned to every community,
    the POI locations/categories, and the POI-to-node matches, while removing
    the observed association between community labels and geographical nodes.
    When ``spatial_block_m`` is provided, labels are permuted only among nodes
    in the same projected grid cell, retaining broad spatial autocorrelation.
    """
    matched = pa[(pa["community"] != -1) & pa["nearest_node"].notna()].copy()
    node_ids = [node for node, label in community_dict.items() if label != -1]
    if matched.empty or len(node_ids) < 2 or n_permutations < 1:
        return {"null_mean": float("nan"), "null_sd": float("nan"), "z_score": float("nan"), "p_value": float("nan")}

    node_index = {node: index for index, node in enumerate(node_ids)}
    matched = matched[matched["nearest_node"].isin(node_index)]
    if matched.empty:
        return {"null_mean": float("nan"), "null_sd": float("nan"), "z_score": float("nan"), "p_value": float("nan")}

    label_values = sorted({community_dict[node] for node in node_ids})
    label_index = {label: index for index, label in enumerate(label_values)}
    base_labels = np.array([label_index[community_dict[node]] for node in node_ids], dtype=int)
    poi_node_indices = np.array([node_index[node] for node in matched["nearest_node"]], dtype=int)
    categories, category_codes = np.unique(matched["category"].to_numpy(), return_inverse=True)
    n_categories = len(categories)
    # Aggregate repeated POIs once at node × category level. Every permutation
    # then accumulates node rows rather than revisiting individual POI records;
    # this is algebraically identical to the original POI-level np.add.at.
    node_category_counts = np.zeros((len(node_ids), n_categories), dtype=np.int64)
    np.add.at(node_category_counts, (poi_node_indices, category_codes), 1)
    poi_active_nodes = np.flatnonzero(node_category_counts.sum(axis=1) > 0)

    active_counts = node_category_counts[poi_active_nodes]
    entry_node_positions, entry_categories = np.nonzero(active_counts)
    entry_weights = active_counts[entry_node_positions, entry_categories].astype(float)
    n_labels = len(label_values)

    def weighted_entropy_batch(node_label_rows):
        """Evaluate one or more label vectors with algebraically exact counts."""
        rows = np.asarray(node_label_rows, dtype=int)
        if rows.ndim == 1:
            rows = rows[None, :]
        batch_count = rows.shape[0]
        entry_labels = rows[:, poi_active_nodes][:, entry_node_positions]
        flat_indices = (
            (
                np.arange(batch_count, dtype=np.int64)[:, None] * n_labels
                + entry_labels
            )
            * n_categories
            + entry_categories[None, :]
        )
        weights = np.broadcast_to(entry_weights, flat_indices.shape).ravel()
        counts = np.bincount(
            flat_indices.ravel(),
            weights=weights,
            minlength=batch_count * n_labels * n_categories,
        ).reshape(batch_count, n_labels, n_categories)
        totals = counts.sum(axis=2)
        probabilities = np.divide(
            counts,
            totals[:, :, None],
            out=np.zeros_like(counts),
            where=totals[:, :, None] > 0,
        )
        terms = np.zeros_like(probabilities)
        positive = probabilities > 0
        terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
        if n_categories > 1:
            entropies = -terms.sum(axis=2) / np.log2(n_categories)
        else:
            entropies = np.zeros_like(totals)
        total_weights = totals.sum(axis=1)
        return np.divide(
            (entropies * totals).sum(axis=1),
            total_weights,
            out=np.full(batch_count, np.nan),
            where=total_weights > 0,
        )

    def weighted_entropy(node_labels):
        return float(weighted_entropy_batch(node_labels)[0])

    observed = weighted_entropy(base_labels)
    rng = np.random.default_rng(seed)
    block_indices = None
    spatial_block_count = None
    active_spatial_block_count = None
    if spatial_block_m is not None:
        if graph is None:
            raise ValueError("graph is required for a spatially constrained POI null")
        valid_coords = [
            (float(graph.nodes[node]["lat"]), float(graph.nodes[node]["lon"]))
            for node in node_ids
            if node in graph and "lat" in graph.nodes[node] and "lon" in graph.nodes[node]
        ]
        if len(valid_coords) != len(node_ids):
            raise ValueError("all permuted graph nodes require coordinates")
        lat0 = np.radians(np.mean([lat for lat, _ in valid_coords]))
        radius_m = 6371000.0
        xy = np.array([
            [
                radius_m * np.radians(lon) * np.cos(lat0),
                radius_m * np.radians(lat),
            ]
            for lat, lon in valid_coords
        ])
        cell_ids = np.floor(xy / float(spatial_block_m)).astype(np.int64)
        grouped_indices = defaultdict(list)
        for index, cell in enumerate(map(tuple, cell_ids)):
            grouped_indices[cell].append(index)
        spatial_block_count = len(grouped_indices)
        # Singleton blocks and blocks containing only one community label are
        # invariant under within-block permutation. Skipping them changes no
        # possible null realization and removes most small-grid overhead.
        block_indices = []
        for indices in grouped_indices.values():
            indices = np.asarray(indices, dtype=int)
            if len(indices) > 1 and len(np.unique(base_labels[indices])) > 1:
                block_indices.append(indices)
        active_spatial_block_count = len(block_indices)

    def permuted_labels():
        if block_indices is None:
            return rng.permutation(base_labels)
        permuted = base_labels.copy()
        for indices_in_block in block_indices:
            if len(indices_in_block) > 1:
                permuted[indices_in_block] = rng.permutation(base_labels[indices_in_block])
        return permuted

    # Keep RNG calls in their original order while evaluating entropy in
    # bounded batches. Counts are identical integer node-category totals.
    requested_batch_size = max(1, int(permutation_batch_size))
    nonzero_entries = max(1, len(entry_weights))
    memory_bounded_batch_size = max(1, 4_000_000 // nonzero_entries)
    effective_batch_size = min(requested_batch_size, memory_bounded_batch_size)
    null = np.empty(n_permutations, dtype=float)
    for start in range(0, n_permutations, effective_batch_size):
        stop = min(start + effective_batch_size, n_permutations)
        label_rows = np.stack([permuted_labels() for _ in range(stop - start)])
        null[start:stop] = weighted_entropy_batch(label_rows)
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    z_score = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")
    p_value = float((1 + np.sum(np.abs(null - null_mean) >= abs(observed - null_mean))) / (len(null) + 1))
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_ci_low": float(np.quantile(null, 0.025)),
        "null_ci_high": float(np.quantile(null, 0.975)),
        "z_score": z_score,
        "p_value": p_value,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "null_type": "spatial_block" if spatial_block_m is not None else "global",
        "spatial_block_m": spatial_block_m,
        "spatial_block_count": spatial_block_count,
        "active_spatial_block_count": active_spatial_block_count,
        "permutation_batch_size": int(effective_batch_size),
    }


# %%
# %% Cell 8 — 保存 + 缓存 + run_one
def save_result(city,algo,config,metrics,community):
    rd=RESULTS_DIR/city/algo/datetime.now().strftime("%Y%m%d_%H%M%S")
    rd.mkdir(parents=True,exist_ok=True)
    with open(rd/"config.json","w",encoding="utf-8") as f:json.dump(_json_safe(config),f,ensure_ascii=False,indent=2)
    with open(rd/"metrics.json","w",encoding="utf-8") as f:json.dump(_json_safe(metrics),f,ensure_ascii=False,indent=2)
    with open(rd/"communities.csv","w",encoding="utf-8",newline="") as f:
        wr=csv.writer(f);wr.writerow(["node_id","community_id"])
        for n,c in community.items():wr.writerow([n,c])
    print(f"  [saved] {rd}")
    return rd

_GRAPH_CACHE={}
def get_graph(city_cfg):
    city=city_cfg["city"]
    threshold=GLOBAL_CFG["distance_threshold_m"]
    key=(city,threshold,GLOBAL_CFG["direction_method"])
    if key in _GRAPH_CACHE:return _GRAPH_CACHE[key]
    b = PTNGraphBuilder(
        city_cfg["json_file"],
        city,
        city_cfg["country"],
        threshold,
        city_query=city_cfg.get("city_query", city)
    )
    g=b.build();_GRAPH_CACHE[key]=(g,b);return g,b

# ======== 社区结果缓存 ========
_COMMUNITY_CACHE = {}

def get_community(city_cfg, algo_name, **algo_kwargs):
    """
    带缓存地获取社区划分结果。

    缓存键：
    (city, algo_name, sorted(algo_kwargs.items()))
    """
    city = city_cfg["city"]
    if algo_name == "DCSBM" and "n_restarts" not in algo_kwargs:
        algo_kwargs["n_restarts"] = GLOBAL_CFG.get("dcsbm_restarts", 50)
    if algo_name == "DCSBM" and "checkpoint_dir" not in algo_kwargs:
        algo_kwargs["checkpoint_dir"] = str(
            OUTPUT_ROOT / "dcsbm_checkpoints" / PIPELINE_VERSION / city
        )
    key = (
        PIPELINE_VERSION,
        city,
        GLOBAL_CFG["distance_threshold_m"],
        algo_name,
        tuple(sorted(algo_kwargs.items()))
    )

    if key in _COMMUNITY_CACHE:
        return _COMMUNITY_CACHE[key]

    graph, _ = get_graph(city_cfg)
    runner = ALGORITHM_REGISTRY.get(algo_name)
    if runner is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    community, meta = runner(graph, **algo_kwargs)
    _COMMUNITY_CACHE[key] = (community, meta)
    return community, meta

def clear_community_cache():
    _COMMUNITY_CACHE.clear()
    print("[OK] community cache cleared.")


def run_one(city_cfg, algo_name, show_base=False, show_comm_map=True, show_comm_plot=True,
            show_table=True, show_poi=True, save_static_map=False):
    city = city_cfg["city"]
    print(f"\n{'='*60}\n  {city} × {algo_name}\n{'='*60}")

    try:
        graph, builder = get_graph(city_cfg)
    except Exception as e:
        print(f"  [ERROR] Build: {e}")
        traceback.print_exc()
        return None, None

    if graph.number_of_nodes() == 0:
        print("  [SKIP] empty")
        return None, None

    if show_base:
        try:
            bm = create_base_map(graph, builder.boundary_coords, builder.stats)
            _save_folium(bm, city, algo_name, "base_map", auto_open=False, use_iframe=False)
        except Exception as e:
            print(f"  [viz] base map: {e}")

    # ===== 带缓存获取 community =====
    try:
        community, ameta = get_community(city_cfg, algo_name)
        nc = len(set(v for v in community.values() if v != -1))
        print(f"  [algo] {nc} communities")
    except Exception as e:
        print(f"  [ERROR] algo: {e}")
        traceback.print_exc()
        return None, None

    # ===== 静态论文图导出（默认可关掉，加速）=====
    if save_static_map:
        try:
            save_static_boundary_map(
                graph=graph,
                community_dict=community,
                boundary_coords=builder.boundary_coords,
                city=city,
                algorithm=algo_name,
                output_root=RESULTS_DIR,
                dpi=FIGURE_CFG.get("dpi", 600),
                width_mm=get_figure_width_mm("map"),
                add_basemap=True,
                basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
                clip_to_boundary=True,
                show_nodes=True,
                stops_only=True,
                node_size=6,
                line_alpha=0.92,
                show_boundary=False,
                line_width_base=1.0,
                line_width_scale=0.25,
                transparent=False
            )
        except Exception as e:
            print(f"  [static map] failed: {e}")

    try:
        metrics = compute_standard_metrics(graph, community)
        print(f"  [met] Q={metrics['Q']:.4f} ACT={metrics['ACT']:.4f} ACD={metrics['ACD']:.4f} "
              f"ACBC={metrics['ACBC']:.4f} CGCC={metrics['CGCC']:.4f} PL={metrics['PL']:.4f}")
    except Exception as e:
        print(f"  [ERROR] metrics: {e}")
        metrics = {
            "community_count": 0,
            "Q": float("nan"),
            "ACT": float("nan"),
            "ACD": float("nan"),
            "ACBC": float("nan"),
            "CGCC": float("nan"),
            "PL": float("nan"),
            "modularity_source": "failed"
        }

    config = {
        "city": city,
        "country": city_cfg["country"],
        "algorithm": algo_name,
        "script_version": PIPELINE_VERSION,
        "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "distance_threshold_m": GLOBAL_CFG["distance_threshold_m"],
        "direction_method": GLOBAL_CFG["direction_method"],
        "connectivity_filter": GLOBAL_CFG["connectivity_filter"],
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges()
    }
    config.update(ameta)
    save_result(city, algo_name, config, metrics, community)

    if show_comm_plot:
        try:
            fig = create_community_network_plot(graph, community, city, algo_name)
            if fig:
                _save_plotly(fig, city, algo_name, "community_structure")
                if DISPLAY_OUTPUT:
                    fig.show()

        except Exception as e:
            print(f"  [viz] plot: {e}")

    if show_table:
        try:
            fig = create_metrics_table(city, algo_name, metrics)
            _save_plotly(fig, city, algo_name, "metrics_table")
            if DISPLAY_OUTPUT:
                fig.show()
        except Exception as e:
            print(f"  [viz] table: {e}")

    if show_comm_map:
        try:
            cm2 = create_community_map(graph, community, builder.boundary_coords, city, algo_name)
            _save_folium(cm2, city, algo_name, "community_map", auto_open=False, use_iframe=False)
        except Exception as e:
            print(f"  [viz] comm map: {e}")

    poi_metrics = {}
    if show_poi:
        try:
            pdf = load_poi(city)
            if not pdf.empty:
                pa = assign_pois(
                    graph,
                    community,
                    pdf,
                    max_distance_m=GLOBAL_CFG["poi_max_distance_m"],
                )
                cdf, ps = compute_poi_metrics(pa, community)

                if ps:
                    poi_metrics = ps
                    print(f"  [poi] entropy_w={ps.get('entropy_w',0):.4f} spec_w={ps.get('spec_w',0):.4f}")

                    ph = ['<b>Metric</b>', '<b>Value</b>']
                    pm = [
                        ('Total POIs', f"{ps.get('total_pois',0):,}"),
                        ('Categories', str(ps.get('K',0))),
                        ('Entropy (w)', f"{ps.get('entropy_w',0):.4f}"),
                        ('Specialization (w)', f"{ps.get('spec_w',0):.4f}"),
                        ('Simpson (w)', f"{ps.get('simpson_w',0):.4f}"),
                        ('Dominant share', f"{ps.get('dom_share',0):.4f}"),
                        ('POI density', f"{ps.get('density',0):.2f}")
                    ]

                    pfig = go.Figure(data=[go.Table(
                        header=dict(
                            values=ph,
                            fill_color='#2c5a89',
                            font=dict(color='white', size=14),
                            align='left',
                            height=40
                        ),
                        cells=dict(
                            values=[
                                [m[0] for m in pm],
                                [m[1] for m in pm]
                            ],
                            fill_color='white',
                            align='left',
                            height=35,
                            font_size=13
                        )
                    )])

                    pfig.update_layout(
                        title=None,
                        width=850,
                        height=380,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )

                    _save_plotly(pfig, city, algo_name, "poi_table")
                    if DISPLAY_OUTPUT:
                        pfig.show()

                    sd = cdf.sort_values("n_pois", ascending=False).head(20)
                    pfig2 = go.Figure()
                    pfig2.add_trace(go.Bar(
                        x=[f"C{c}" for c in sd["community"]],
                        y=sd["norm_entropy"],
                        name="Entropy"
                    ))
                    pfig2.add_trace(go.Bar(
                        x=[f"C{c}" for c in sd["community"]],
                        y=sd["specialization"],
                        name="Specialization"
                    ))
                    pfig2.update_layout(
                        title=None,
                        barmode='group',
                        width=1100,
                        height=450,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    _save_plotly(pfig2, city, algo_name, "poi_diversity")
                    if DISPLAY_OUTPUT:
                        pfig2.show()

                    pod = RESULTS_DIR / city / algo_name / "poi"
                    pod.mkdir(parents=True, exist_ok=True)
                    cdf.to_csv(pod / "community_poi.csv", index=False, encoding="utf-8-sig")
                    with open(pod / "poi_summary.json", "w", encoding="utf-8") as f:
                        json.dump(_json_safe(ps), f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  [poi] failed: {e}")
            traceback.print_exc()

    print(f"  [OK] {city} × {algo_name} ✓")
    return community, {**metrics, **{f"poi_{k}": v for k, v in poi_metrics.items()}}


# %%
# %% Cell 9 — 汇总/对比/敏感性/稳定性
def collect_results():
    rows=[]
    for cp in RESULTS_DIR.rglob("config.json"):
        rd=cp.parent
        with open(cp,"r",encoding="utf-8") as f:cfg=json.load(f)
        if cfg.get("script_version") != PIPELINE_VERSION:
            continue
        mp=rd/"metrics.json";met={}
        if mp.exists():
            with open(mp,"r",encoding="utf-8") as f:met=json.load(f)
        rows.append({"run_dir":str(rd),"script_version":cfg.get("script_version"),
            "city":cfg.get("city"),"algorithm":cfg.get("algorithm"),
            "date_run":cfg.get("date_run"),"graph_nodes":cfg.get("graph_nodes"),
            "graph_edges":cfg.get("graph_edges"),"directed_used":cfg.get("directed_used"),
            "weight_used":cfg.get("weight_used"),"spatial_prior":cfg.get("spatial_prior"),
            **{metric: met.get(metric) for metric in (
                "community_count", "community_size_min", "community_size_median",
                "community_size_max", "largest_community_share", "singleton_communities",
                "communities_smaller_than_5", "Q", "Q_directed", "ACT",
                "ACT_node_weighted", "ACD", "ACD_node_weighted", "ACBC", "CGCC",
                "community_flow_asymmetry", "PL"
            )}})
    df=pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values(["city", "algorithm", "date_run"])
            .drop_duplicates(subset=["city", "algorithm"], keep="last")
            .reset_index(drop=True)
        )
    return df

def show_summary():
    df=collect_results()
    if df.empty:print("No results.");return None
    print(f"\nFound {len(df)} runs")
    df.to_csv(SUMMARY_DIR/"master_results.csv",index=False,encoding="utf-8-sig")
    pivots={}
    for m in [
        "Q", "Q_directed", "ACT", "ACT_node_weighted", "ACD", "ACD_node_weighted",
        "ACBC", "CGCC", "community_flow_asymmetry", "PL", "community_count",
        "largest_community_share"
    ]:
        if m in df.columns:
            lt=df.drop_duplicates(subset=["city","algorithm"],keep="last")
            try:pivots[m]=lt.pivot(index="city",columns="algorithm",values=m)
            except:pass
    try:
        with pd.ExcelWriter(SUMMARY_DIR/"master_results.xlsx",engine="openpyxl") as wr:
            df.to_excel(wr,sheet_name="all",index=False)
            for n,p in pivots.items():p.to_excel(wr,sheet_name=f"pivot_{n}")
        print(f"[OK] master_results.xlsx")
    except ImportError:
        print("⚠ openpyxl missing, skip xlsx")
    for n,p in pivots.items():
        fig=go.Figure(data=go.Heatmap(z=p.values,x=p.columns.tolist(),y=p.index.tolist(),
            colorscale='RdYlGn' if n!='ACD' else 'RdYlGn_r',
            text=np.round(p.values,4),
            texttemplate="%{text}", hoverinfo='z'))
        _apply_heatmap_style(
            fig,
            annotation_size=17,
            width=900,
            height=500,
            left=120,
            bottom=105,
            x_title="Algorithm",
            y_title="City",
        )
        _save_plotly_summary(fig,f"heatmap_{n}")
        if DISPLAY_OUTPUT:
            fig.show()
            print(f"\n=== {n} ===")
            display(p.round(4))
    print(f"\n[OK] {SUMMARY_DIR}/master_results.csv")
    return df

def run_sensitivity(city_cfg, algo, param, values):
    city = city_cfg["city"]
    print(f"\n  Sensitivity: {city}×{algo}×{param}")

    graph, _ = get_graph(city_cfg)
    runner = ALGORITHM_REGISTRY.get(algo)
    if not runner:
        return pd.DataFrame()

    rows = []
    for v in values:
        try:
            c, _ = runner(graph, **{param: v})
            m = compute_standard_metrics(graph, c)
            rows.append({
                "city": city,
                "algorithm": algo,
                "parameter": param,
                "param_value": v,
                **m,
            })
            print(f"  {param}={v}: Q={m['Q']:.4f} n={m['community_count']}")
        except Exception as e:
            print(f"  {param}={v} FAILED:{e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ===== 统一保留列顺序 =====
    cols = [
        "city", "algorithm", "parameter", "param_value", "community_count",
        "community_size_min", "community_size_median", "community_size_max",
        "largest_community_share", "Q", "Q_directed", "ACT", "ACT_node_weighted",
        "ACD", "ACD_node_weighted", "ACBC", "CGCC", "community_flow_asymmetry", "PL",
    ]
    cols = [c for c in cols if c in df.columns]
    out_df = df[cols].copy()

    # 数值格式整理
    for c in out_df.columns:
        if c not in {"city", "algorithm", "parameter", "param_value"}:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").round(4)

    # ===== 保存目录 =====
    out_dir = RESULTS_DIR / city / algo / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / f"sensitivity_{param}.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [table] {csv_path}")

    # HTML 表格
    html_path = out_dir / f"sensitivity_{param}.html"
    out_df.to_html(html_path, index=False, border=0)
    print(f"  [table] {html_path}")

    if DISPLAY_OUTPUT:
    # notebook 中显示
        display(out_df)

    return out_df


def graph_fingerprint(graph):
    payload = {
        "nodes": sorted(graph.nodes()),
        "edges": [
            [u, v, float(data.get("weight", 1.0))]
            for u, v, data in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1]))
        ]
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_threshold_sensitivity(cities=None, thresholds=(50, 75, 100, 125, 150),
                              algorithms=("LeidenRBC", "Infomap")):
    selected = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    rows = []

    for city_cfg in selected:
        for threshold in thresholds:
            builder = PTNGraphBuilder(
                city_cfg["json_file"],
                city_cfg["city"],
                city_cfg["country"],
                dt=threshold,
                city_query=city_cfg.get("city_query", city_cfg["city"])
            )
            graph = builder.build()
            fingerprint = graph_fingerprint(graph)

            for algorithm in algorithms:
                community, _ = ALGORITHM_REGISTRY[algorithm](graph)
                metrics = compute_standard_metrics(graph, community)
                rows.append({
                    "city": city_cfg["city"],
                    "country": city_cfg["country"],
                    "threshold_m": threshold,
                    "algorithm": algorithm,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "stop_groups": len(builder.ga),
                    "endpoint_stops": sum(bool(graph.nodes[n].get("is_stop")) for n in graph.nodes()),
                    "graph_fingerprint": fingerprint,
                    **metrics
                })

    raw = pd.DataFrame(rows)
    out_dir = OUTPUT_ROOT / "threshold_sensitivity_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "threshold_sensitivity_raw.csv", index=False, encoding="utf-8-sig")

    summary = raw.groupby(["city", "country", "algorithm"], as_index=False).agg(
        node_range=("nodes", lambda s: int(s.max() - s.min())),
        edge_range=("edges", lambda s: int(s.max() - s.min())),
        topology_versions=("graph_fingerprint", "nunique"),
        community_min=("community_count", "min"),
        community_max=("community_count", "max"),
        Q_min=("Q", "min"),
        Q_max=("Q", "max")
    )
    summary.to_csv(out_dir / "threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return raw, summary


def _partition_agreement(labels_a, labels_b):
    """Return NMI, ARI, and variation of information for two label vectors."""
    if not labels_a or len(labels_a) != len(labels_b):
        return {"nmi": float("nan"), "ari": float("nan"), "vi": float("nan")}
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    mutual_information = mutual_info_score(labels_a, labels_b)
    _, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)
    probabilities_a = counts_a / counts_a.sum()
    probabilities_b = counts_b / counts_b.sum()
    entropy_a = -float(np.sum(probabilities_a * np.log(probabilities_a)))
    entropy_b = -float(np.sum(probabilities_b * np.log(probabilities_b)))
    return {
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "vi": float(entropy_a + entropy_b - 2.0 * mutual_information),
    }


def run_representation_sensitivity(
    cities=None,
    algorithms=None,
    output_subdir="representation_sensitivity_output",
):
    """Compare endpoint, L-space, and P-space partitions on explicit common supports.

    L-space and P-space share stop-group identifiers, so their agreement isolates
    adjacency-construction sensitivity. Endpoint comparisons transfer labels to
    stop centroids and are repeated under distance limits to expose uncertainty
    induced by changing both observational units and adjacency.
    """
    if not SKLEARN_OK:
        raise ImportError("scikit-learn is required for representation NMI")
    selected = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    algorithms = tuple(ALGORITHMS if algorithms is None else algorithms)
    unknown = sorted(set(algorithms) - set(ALGORITHM_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown representation-sensitivity algorithms: {unknown}")
    pairwise_rows = []
    legacy_rows = []
    thresholds = [None] + list(GLOBAL_CFG.get(
        "representation_transfer_thresholds_m", [100, 250, 500, 1000]
    ))

    for city_cfg in selected:
        endpoint_graph, builder = get_graph(city_cfg)
        l_space_graph = builder.build_stop_graph()
        p_space_graph = builder.build_p_space_graph()
        if min(l_space_graph.number_of_nodes(), p_space_graph.number_of_nodes()) < 2:
            print(f"  [representation] {city_cfg['city']}: insufficient ordered stop members")
            continue

        endpoint_nodes = list(endpoint_graph.nodes())
        endpoint_coords = np.radians(np.array([
            [endpoint_graph.nodes[n]["lat"], endpoint_graph.nodes[n]["lon"]]
            for n in endpoint_nodes
        ]))
        endpoint_tree = BallTree(endpoint_coords, metric="haversine")

        for algorithm in algorithms:
            endpoint_partition, _ = get_community(city_cfg, algorithm)
            # Keep the formal algorithm settings identical across representations.
            # In particular, DCSBM must retain the 50-start selection rule; each
            # representation receives an isolated checkpoint directory because its
            # node/edge support differs from Endpoint.
            def _run_representation(graph, representation_name):
                kwargs = {}
                if algorithm == "DCSBM":
                    kwargs.update({
                        "n_restarts": int(GLOBAL_CFG.get("dcsbm_restarts", 50)),
                        "checkpoint_dir": str(
                            OUTPUT_ROOT / "dcsbm_checkpoints" / PIPELINE_VERSION
                            / "representation" / representation_name / city_cfg["city"]
                        ),
                    })
                return ALGORITHM_REGISTRY[algorithm](graph, **kwargs)

            l_partition, _ = _run_representation(l_space_graph, "l_space")
            p_partition, _ = _run_representation(p_space_graph, "p_space")
            endpoint_metrics = compute_standard_metrics(endpoint_graph, endpoint_partition)
            l_metrics = compute_standard_metrics(l_space_graph, l_partition)
            p_metrics = compute_standard_metrics(p_space_graph, p_partition)

            common_nodes = sorted(set(l_space_graph) & set(p_space_graph))
            l_labels = [l_partition.get(node, -1) for node in common_nodes]
            p_labels = [p_partition.get(node, -1) for node in common_nodes]
            valid_lp = [i for i, (a, b) in enumerate(zip(l_labels, p_labels)) if a != -1 and b != -1]
            agreement_lp = _partition_agreement(
                [l_labels[i] for i in valid_lp],
                [p_labels[i] for i in valid_lp],
            )
            pairwise_rows.append({
                "city": city_cfg["city"],
                "country": city_cfg["country"],
                "algorithm": algorithm,
                "comparison": "l_space__p_space",
                "transfer_threshold_m": float("nan"),
                "common_nodes": len(valid_lp),
                "common_support_rate": len(valid_lp) / len(set(l_space_graph) | set(p_space_graph)),
                "transfer_distance_median": float("nan"),
                "transfer_distance_p95": float("nan"),
                "many_to_one_rate": float("nan"),
                **agreement_lp,
                "left_nodes": l_space_graph.number_of_nodes(),
                "left_edges": l_space_graph.number_of_edges(),
                "left_community_count": l_metrics.get("community_count"),
                "left_Q": l_metrics.get("Q"),
                "right_nodes": p_space_graph.number_of_nodes(),
                "right_edges": p_space_graph.number_of_edges(),
                "right_community_count": p_metrics.get("community_count"),
                "right_Q": p_metrics.get("Q"),
            })

            for target_name, target_graph, target_partition, target_metrics in (
                ("l_space", l_space_graph, l_partition, l_metrics),
                ("p_space", p_space_graph, p_partition, p_metrics),
            ):
                target_nodes = list(target_graph.nodes())
                target_coords = np.radians(np.array([
                    [target_graph.nodes[n]["lat"], target_graph.nodes[n]["lon"]]
                    for n in target_nodes
                ]))
                distances, indices = endpoint_tree.query(target_coords, k=1)
                transfer_distance_m = distances[:, 0] * 6371000.0
                transferred_labels = [
                    endpoint_partition.get(endpoint_nodes[index], -1)
                    for index in indices[:, 0]
                ]
                target_labels = [target_partition.get(node, -1) for node in target_nodes]

                for threshold in thresholds:
                    valid = [
                        i for i, (left, right) in enumerate(zip(transferred_labels, target_labels))
                        if left != -1 and right != -1
                        and (threshold is None or transfer_distance_m[i] <= threshold)
                    ]
                    agreement = _partition_agreement(
                        [transferred_labels[i] for i in valid],
                        [target_labels[i] for i in valid],
                    )
                    retained_distances = transfer_distance_m[valid]
                    retained_endpoint_indices = [int(indices[i, 0]) for i in valid]
                    many_to_one_rate = (
                        1.0 - len(set(retained_endpoint_indices)) / len(retained_endpoint_indices)
                        if retained_endpoint_indices else float("nan")
                    )
                    pairwise_rows.append({
                        "city": city_cfg["city"],
                        "country": city_cfg["country"],
                        "algorithm": algorithm,
                        "comparison": f"endpoint__{target_name}",
                        "transfer_threshold_m": float("nan") if threshold is None else threshold,
                        "common_nodes": len(valid),
                        "common_support_rate": len(valid) / len(target_nodes) if target_nodes else float("nan"),
                        "transfer_distance_median": float(np.median(retained_distances)) if len(retained_distances) else float("nan"),
                        "transfer_distance_p95": float(np.quantile(retained_distances, 0.95)) if len(retained_distances) else float("nan"),
                        "many_to_one_rate": many_to_one_rate,
                        **agreement,
                        "left_nodes": endpoint_graph.number_of_nodes(),
                        "left_edges": endpoint_graph.number_of_edges(),
                        "left_community_count": endpoint_metrics.get("community_count"),
                        "left_Q": endpoint_metrics.get("Q"),
                        "right_nodes": target_graph.number_of_nodes(),
                        "right_edges": target_graph.number_of_edges(),
                        "right_community_count": target_metrics.get("community_count"),
                        "right_Q": target_metrics.get("Q"),
                    })

                    if target_name == "l_space" and threshold is None:
                        legacy_row = {
                            "city": city_cfg["city"],
                            "country": city_cfg["country"],
                            "algorithm": algorithm,
                            "endpoint_nodes": endpoint_graph.number_of_nodes(),
                            "endpoint_edges": endpoint_graph.number_of_edges(),
                            "stop_nodes": l_space_graph.number_of_nodes(),
                            "stop_edges": l_space_graph.number_of_edges(),
                            "ordered_route_relations": l_space_graph.graph.get("ordered_route_relations", 0),
                            "common_stop_groups": len(valid),
                            "transfer_distance_median": float(np.median(retained_distances)),
                            "transfer_distance_p95": float(np.quantile(retained_distances, 0.95)),
                            "partition_nmi": agreement["nmi"],
                            "partition_ari": agreement["ari"],
                            "partition_vi": agreement["vi"],
                        }
                        for prefix, metrics in (("endpoint", endpoint_metrics), ("stop", l_metrics)):
                            for metric in ("community_count", "Q", "ACT", "ACD", "ACBC", "CGCC", "PL"):
                                legacy_row[f"{prefix}_{metric}"] = metrics.get(metric)
                        legacy_rows.append(legacy_row)

            print(
                f"  [representation] {city_cfg['city']}×{algorithm}: "
                f"L--P NMI={agreement_lp['nmi']:.4f}, "
                f"K(endpoint/L/P)={endpoint_metrics['community_count']}/"
                f"{l_metrics['community_count']}/{p_metrics['community_count']}"
            )

    results = pd.DataFrame(pairwise_rows)
    legacy_results = pd.DataFrame(legacy_rows)
    out_dir = OUTPUT_ROOT / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "representation_pairwise_summary.csv", index=False, encoding="utf-8-sig")
    legacy_results.to_csv(out_dir / "representation_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    if not legacy_results.empty:
        pivot = legacy_results.pivot(index="city", columns="algorithm", values="partition_nmi")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
        ))
        _apply_heatmap_style(
            fig,
            annotation_size=17,
            width=700,
            height=500,
            left=125,
            bottom=90,
            x_title="Algorithm",
            y_title="City",
        )
        try:
            fig.write_html(str(out_dir / "representation_nmi.html"))
        except Exception as exc:
            print(f"  [representation figure] export failed: {exc}")
    return results



def run_stability(city_cfg, algo, n_runs=10):
    city = city_cfg["city"]
    print(f"\n  Stability: {city}×{algo}×{n_runs} runs")

    graph, _ = get_graph(city_cfg)
    runner = ALGORITHM_REGISTRY.get(algo)
    if not runner:
        return pd.DataFrame()

    sig = inspect.signature(runner)
    has_seed = "seed" in sig.parameters

    rows = []
    all_c = []

    if algo == "DCSBM":
        checkpoint_dir = OUTPUT_ROOT / "dcsbm_checkpoints" / PIPELINE_VERSION / city
        # Complete or resume the same starts used for main-result selection.
        # Stability must reuse these partitions rather than spend another full
        # DCSBM budget on an identical seed sequence.
        run_dcsbm(
            graph,
            seed=GLOBAL_CFG["random_seed"],
            n_restarts=n_runs,
            checkpoint_dir=str(checkpoint_dir),
        )
        ordered_nodes = sorted(graph.nodes())
        for i in range(n_runs):
            try:
                saved = np.load(checkpoint_dir / f"restart_{i:03d}.npz", allow_pickle=False)
                labels = saved["labels"]
                c = {node: int(labels[index]) for index, node in enumerate(ordered_nodes)}
                m = compute_standard_metrics(graph, c, sample_efficiency=True)
                rows.append({
                    "run": i + 1,
                    "seed": int(saved["seed"]),
                    "description_length": float(saved["description_length"]),
                    **m,
                })
                all_c.append(c)
                print(f"  run {i+1}: Q={m['Q']:.4f} n={m['community_count']} [checkpoint]")
            except Exception as e:
                print(f"  run {i+1} FAILED: {e}")
    else:
        for i in range(n_runs):
            try:
                kw = {}
                if has_seed:
                    kw["seed"] = GLOBAL_CFG["random_seed"] + i * 17

                c, _ = runner(graph, **kw)
                m = compute_standard_metrics(graph, c, sample_efficiency=True)

                rows.append({"run": i + 1, **m})
                all_c.append(c)

                print(f"  run {i+1}: Q={m['Q']:.4f} n={m['community_count']}")
            except Exception as e:
                print(f"  run {i+1} FAILED: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ===== 指标列 =====
    metric_cols = ["Q", "ACT", "ACD", "ACBC", "CGCC", "PL", "community_count"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # ===== 保存目录 =====
    out_dir = RESULTS_DIR / city / algo / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1) 每次运行结果表
    # =====================================================
    provenance_cols = [column for column in ["seed", "description_length"] if column in df]
    run_df = df[["run"] + provenance_cols + metric_cols].copy()
    for c in metric_cols:
        run_df[c] = pd.to_numeric(run_df[c], errors="coerce").round(4)

    run_csv = out_dir / "stability_runs.csv"
    run_html = out_dir / "stability_runs.html"

    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    run_df.to_html(run_html, index=False, border=0)

    print(f"  [table] {run_csv}")
    print(f"  [table] {run_html}")

    # =====================================================
    # 2) 汇总统计表
    # =====================================================
    srows = []
    for met in metric_cols:
        v = pd.to_numeric(df[met], errors="coerce").dropna()
        if len(v) == 0:
            continue
        srows.append({
            "metric": met,
            "mean": v.mean(),
            "std": v.std(),
            "min": v.min(),
            "max": v.max(),
            "cv": v.std() / v.mean() if v.mean() != 0 else float("nan")
        })

    sdf = pd.DataFrame(srows)
    if not sdf.empty:
        for c in ["mean", "std", "min", "max", "cv"]:
            if c in sdf.columns:
                sdf[c] = pd.to_numeric(sdf[c], errors="coerce").round(4)

        summary_csv = out_dir / "stability_summary.csv"
        summary_html = out_dir / "stability_summary.html"

        sdf.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        sdf.to_html(summary_html, index=False, border=0)

        print(f"  [table] {summary_csv}")
        print(f"  [table] {summary_html}")

    # =====================================================
    # 3) NMI 汇总（可选）
    # =====================================================
    nmi_df = pd.DataFrame()
    if len(all_c) >= 2 and SKLEARN_OK:
        ns = list(graph.nodes())
        label_arrays = [
            np.fromiter((community.get(node, -1) for node in ns), dtype=np.int64, count=len(ns))
            for community in all_c
        ]
        nmis = []
        for i in range(len(label_arrays)):
            for j in range(i + 1, len(label_arrays)):
                nmis.append(normalized_mutual_info_score(label_arrays[i], label_arrays[j]))

        if len(nmis) > 0:
            nmi_df = pd.DataFrame([{
                "n_pairs": len(nmis),
                "nmi_mean": round(float(np.mean(nmis)), 4),
                "nmi_std": round(float(np.std(nmis)), 4),
                "nmi_min": round(float(np.min(nmis)), 4),
                "nmi_max": round(float(np.max(nmis)), 4)
            }])

            nmi_csv = out_dir / "stability_nmi_summary.csv"
            nmi_html = out_dir / "stability_nmi_summary.html"

            nmi_df.to_csv(nmi_csv, index=False, encoding="utf-8-sig")
            nmi_df.to_html(nmi_html, index=False, border=0)

            print(f"  [table] {nmi_csv}")
            print(f"  [table] {nmi_html}")
            print(f"  NMI: mean={nmi_df.loc[0, 'nmi_mean']:.4f} std={nmi_df.loc[0, 'nmi_std']:.4f}")

    # notebook 显示
    if DISPLAY_OUTPUT:
        print("\n  --- Stability run table ---")
        display(run_df)

        if not sdf.empty:
            print("\n  --- Stability summary table ---")
            display(sdf)

        if not nmi_df.empty:
            print("\n  --- Stability NMI summary ---")
            display(nmi_df)

    return {
        "runs": run_df,
        "summary": sdf,
        "nmi": nmi_df
    }


def create_algorithm_fairness_table():
    data = {
        "Algorithm": ["Infomap", "Leiden(RBC)", "DCSBM", "GeoLP", "Spinglass", "DirLou"],
        "Family": ["Flow", "Modularity", "Generative", "Spatial baseline", "Physics", "Heuristic"],
        "Direction": ["✓", "✓", "✓", "✗", "✗", "✓"],
        "Weight": ["✓", "✓", "✓", "✓", "✓", "✓"],
        "Spatial": ["✗", "✗", "✗", "✓", "✗", "✗"],
        "Projection": ["Directed", "Directed", "Directed", "Symmetrized neighbourhood", "Reciprocal weights summed", "Directed"],
        "Key Param": ["markov_time=1", "resolution=0.1", "degree-corrected MDL", "geo_weight=.7; bandwidth=5 km; K0=25", "spins=25 (explicit cap)", "resolution=0.1"],
        "Status": ["Main", "Main", "Main", "Baseline", "Baseline", "Experimental"]
    }

    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color="#2c5a89",
            font=dict(color="white", size=13),
            align="left",
            height=40
        ),
        cells=dict(
            values=[df[c] for c in df.columns],
            fill_color="white",
            align="left",
            height=35,
            font_size=12
        )
    )])

    fig.update_layout(
        title=None,
        width=1200,
        height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    _save_plotly_summary(fig, "algorithm_fairness")
    if DISPLAY_OUTPUT:
        fig.show()
    return df


def _benjamini_hochberg(p_values):
    """Return BH-adjusted p-values while preserving NaN positions and input order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted
    order = valid_positions[np.argsort(values[valid_positions])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def run_poi_all(
    cities=None,
    algorithms=None,
    max_distance_m=None,
    n_permutations=None,
    match_distances_m=None,
    spatial_blocks_m=None,
    result_tag=None,
    render_outputs=True,
):
    """Evaluate POI functional characterization across matching and null scales.

    ``poi_master.csv`` remains the 1000 m/global-null main table used by the
    reported analysis.  The complete matching-distance sensitivity and all global and
    spatially blocked null tests are written to separate long-format tables.
    BH correction is applied across all city--algorithm tests within each
    matching-distance and null-model specification.
    """
    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    al = ALGORITHMS if algorithms is None else [a for a in ALGORITHMS if a in algorithms]

    if match_distances_m is None:
        match_distances_m = (
            [max_distance_m]
            if max_distance_m is not None
            else GLOBAL_CFG.get("poi_match_distances_m", [GLOBAL_CFG["poi_max_distance_m"]])
        )
    match_distances_m = sorted({int(value) for value in match_distances_m})
    if spatial_blocks_m is None:
        spatial_blocks_m = GLOBAL_CFG.get("poi_spatial_null_blocks_m", [1000, 2000, 5000])
    spatial_blocks_m = sorted({int(value) for value in spatial_blocks_m})
    if n_permutations is None:
        n_permutations = GLOBAL_CFG.get("poi_permutations", 9999)
    metric_rows = []
    null_rows = []
    checkpoint_dir = SUMMARY_DIR / "poi_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for cc in cl:
        city = cc["city"]
        pdf = load_poi(city)
        if pdf.empty:
            continue

        graph, _ = get_graph(cc)
        if graph.number_of_nodes() == 0:
            continue

        nearest_matches = match_pois_to_nodes(graph, pdf)
        for algo in al:
            try:
                c, _ = get_community(cc, algo)   # ===== 用缓存 =====
                sm = compute_standard_metrics(graph, c)
                assigned_pois = assign_pois(
                    graph,
                    c,
                    pdf,
                    max_distance_m=None,
                    nearest_matches=nearest_matches,
                )
                for distance_index, distance_m in enumerate(match_distances_m):
                    pa = apply_poi_distance_threshold(assigned_pois, distance_m)
                    _, s = compute_poi_metrics(pa, c)
                    matched_distances = pa.loc[pa["matched"], "nearest_distance_m"]
                    metric_row = {
                        "city": city,
                        "algorithm": algo,
                        "n_comms": sm.get("community_count", 0),
                        "Q": sm.get("Q"),
                        "poi_raw": len(pa),
                        "poi_matched": s.get("total_pois", 0),
                        "poi_match_rate": s.get("total_pois", 0) / len(pa) if len(pa) else float("nan"),
                        "poi_max_distance_m": distance_m,
                        "poi_distance_median": matched_distances.median() if len(matched_distances) else float("nan"),
                        "poi_distance_p95": matched_distances.quantile(0.95) if len(matched_distances) else float("nan"),
                        "poi_entropy_w": s.get("entropy_w", float("nan")),
                        "poi_spec_w": s.get("spec_w", float("nan")),
                        "poi_simpson_w": s.get("simpson_w", float("nan")),
                    }
                    metric_rows.append(metric_row)
                    metric_checkpoint = checkpoint_dir / f"{city}__{algo}__{distance_m}m__metrics.json"
                    metric_checkpoint.write_text(
                        json.dumps(_json_safe(metric_row), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    null_specs = [("global", None)] + [("spatial_block", block) for block in spatial_blocks_m]
                    task_seed_offset = int.from_bytes(
                        hashlib.sha256(f"{city}|{algo}".encode("utf-8")).digest()[:4],
                        byteorder="big",
                    ) % 1_000_000
                    for null_index, (null_type, block_m) in enumerate(null_specs):
                        null_label = "global" if block_m is None else f"block_{block_m}m"
                        null_checkpoint = checkpoint_dir / (
                            f"{city}__{algo}__{distance_m}m__{null_label}__n{n_permutations}.json"
                        )
                        expected_seed = (
                            GLOBAL_CFG["random_seed"]
                            + task_seed_offset
                            + 101 * distance_index
                            + 17 * null_index
                        )
                        null_row = None
                        if null_checkpoint.exists():
                            candidate_row = json.loads(null_checkpoint.read_text(encoding="utf-8"))
                            if (
                                candidate_row.get("n_permutations") == int(n_permutations)
                                and candidate_row.get("seed") == int(expected_seed)
                            ):
                                null_row = candidate_row
                                print(f"  [POI] resumed {null_checkpoint.name}")
                        if null_row is None:
                            null = poi_entropy_permutation_test(
                                pa,
                                c,
                                n_permutations=n_permutations,
                                seed=expected_seed,
                                graph=graph,
                                spatial_block_m=block_m,
                            )
                            null_row = {
                                "city": city,
                                "algorithm": algo,
                                "poi_max_distance_m": distance_m,
                                **null,
                            }
                            temporary_checkpoint = null_checkpoint.with_suffix(".tmp")
                            temporary_checkpoint.write_text(
                                json.dumps(_json_safe(null_row), ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            os.replace(temporary_checkpoint, null_checkpoint)
                        null_rows.append(null_row)
                    print(
                        f"  {city}×{algo} @ {distance_m} m: "
                        f"Q={sm['Q']:.4f} ent={s.get('entropy_w', float('nan')):.4f}"
                    )

            except Exception as e:
                print(f"  {city}×{algo} FAILED: {e}")

    sensitivity_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if sensitivity_df.empty:
        return sensitivity_df

    if not null_df.empty:
        null_df["p_value_bh"] = np.nan
        family_columns = ["poi_max_distance_m", "null_type", "spatial_block_m"]
        for _, indices in null_df.groupby(family_columns, dropna=False).groups.items():
            null_df.loc[indices, "p_value_bh"] = _benjamini_hochberg(null_df.loc[indices, "p_value"])
        null_df["significant_bh_0_05"] = null_df["p_value_bh"] <= 0.05

    suffix = f"__{result_tag}" if result_tag else ""
    sensitivity_df.to_csv(
        SUMMARY_DIR / f"poi_matching_distance_sensitivity{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null_df.to_csv(
        SUMMARY_DIR / f"poi_null_model_tests{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_distance = int(GLOBAL_CFG["poi_max_distance_m"])
    if primary_distance not in set(sensitivity_df["poi_max_distance_m"]):
        primary_distance = int(max(sensitivity_df["poi_max_distance_m"]))
    mdf = sensitivity_df[sensitivity_df["poi_max_distance_m"].eq(primary_distance)].copy()
    if not null_df.empty:
        primary_null = null_df[
            null_df["poi_max_distance_m"].eq(primary_distance)
            & null_df["null_type"].eq("global")
        ][["city", "algorithm", "null_mean", "null_sd", "null_ci_low", "null_ci_high", "z_score", "p_value", "p_value_bh"]]
        primary_null = primary_null.rename(columns={
            "null_mean": "poi_entropy_null_mean",
            "null_sd": "poi_entropy_null_sd",
            "null_ci_low": "poi_entropy_null_ci_low",
            "null_ci_high": "poi_entropy_null_ci_high",
            "z_score": "poi_entropy_z",
            "p_value": "poi_entropy_p",
            "p_value_bh": "poi_entropy_p_bh",
        })
        mdf = mdf.merge(primary_null, on=["city", "algorithm"], how="left")

    mdf.to_csv(SUMMARY_DIR / f"poi_master{suffix}.csv", index=False, encoding="utf-8-sig")

    if not render_outputs:
        return mdf

    # 保存总表
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in mdf.columns],
            fill_color="#2c5a89",
            font=dict(color="white", size=12),
            align="left",
            height=36
        ),
        cells=dict(
            values=[mdf[c].round(4) if pd.api.types.is_numeric_dtype(mdf[c]) else mdf[c] for c in mdf.columns],
            fill_color="white",
            align="left",
            height=32,
            font_size=11
        )
    )])
    fig.update_layout(
        title=None,
        width=1200,
        height=420,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    _save_plotly_summary(fig, "poi_master_table")

    # Heatmaps
    for met in ["poi_entropy_w", "poi_spec_w", "poi_simpson_w"]:
        if met not in mdf.columns:
            continue
        try:
            piv = mdf.pivot(index="city", columns="algorithm", values=met)
            fig = go.Figure(data=go.Heatmap(
                z=piv.values,
                x=piv.columns.tolist(),
                y=piv.index.tolist(),
                colorscale="RdYlGn_r" if "entropy" in met else "RdYlGn",
                text=np.round(piv.values, 4),
                texttemplate="%{text}"
            ))
            _apply_heatmap_style(
                fig,
                annotation_size=17,
                width=900,
                height=500,
                left=120,
                bottom=105,
                x_title="Algorithm",
                y_title="City",
            )
            _save_plotly_summary(fig, f"poi_heatmap_{met}")
            if DISPLAY_OUTPUT:
                fig.show()
        except Exception as e:
            print(f"  [poi heatmap] {met} FAILED: {e}")

    if DISPLAY_OUTPUT:
        display(mdf.round(4))
    return mdf


# %%
def run_batch(cities=None, algorithms=None, show_base=False, show_comm_map=True,
              show_comm_plot=True, show_table=True, show_poi=True, save_static_map=False):
    """
    批量运行城市 × 算法组合。
    
    参数
    ----
    cities : list[str] or None
    algorithms : list[str] or None
    show_base / show_comm_map / show_comm_plot / show_table / show_poi : bool
    save_static_map : bool
        是否在 run_one 中导出静态论文图。
        调试阶段建议 False，可明显提速。
    """
    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    al = ALGORITHMS if algorithms is None else [a for a in ALGORITHMS if a in algorithms]

    total = len(cl) * len(al)
    ok = 0
    fail = 0

    print(f"\n{'#'*60}")
    print(f"  Batch: {len(cl)} × {len(al)} = {total}")
    print(f"{'#'*60}")

    t0 = time.time()

    for cc in cl:
        for algo in al:
            try:
                r = run_one(
                    cc,
                    algo,
                    show_base=show_base,
                    show_comm_map=show_comm_map,
                    show_comm_plot=show_comm_plot,
                    show_table=show_table,
                    show_poi=show_poi,
                    save_static_map=save_static_map
                )

                if r[0] is not None:
                    ok += 1
                else:
                    fail += 1

            except Exception as e:
                print(f"[FATAL] {cc['city']} × {algo}: {e}")
                traceback.print_exc()
                fail += 1

    elapsed = time.time() - t0

    print(f"\n{'#'*60}")
    print(f"  Done! ok={ok} fail={fail} time={elapsed:.0f}s")
    print(f"{'#'*60}")


# %%
def run_full(cities=None, algorithms=None, sens_cfg=None, stab_runs=5,
             show_maps=False, run_reviewer_modules=True):
    """
    最终完整流水线（可提交版）

    STEP 1  Batch
    STEP 2  Sensitivity (table output)
    STEP 3  Stability (table output)
    STEP 4  Fairness
    STEP 5  POI
    STEP 6  Summary
    STEP 7  Metric Independence
    STEP 8  Spatial Compactness
    STEP 9  Scale Sensitivity
    STEP 10 Boundary Documentation
    """

    cl = CITIES if cities is None else [c for c in CITIES if c["city"] in cities]
    al = ALGORITHMS if algorithms is None else [a for a in ALGORITHMS if a in algorithms]

    print(f"\n{'#'*60}")
    print("  FULL PIPELINE")
    print(f"  Cities     : {[c['city'] for c in cl]}")
    print(f"  Algorithms : {al}")
    print(f"{'#'*60}")

    t0 = time.time()

    # =====================================================
    # STEP 1: Batch
    # =====================================================
    print("\n===== STEP 1: Batch =====")
    run_batch(
        cities=[c["city"] for c in cl],
        algorithms=al,
        show_base=False,
        show_comm_map=show_maps,
        show_comm_plot=show_maps,
        show_table=show_maps,
        show_poi=True,           # 保存POI表和POI图
        save_static_map=show_maps
    )

    # =====================================================
    # STEP 2: Sensitivity
    # =====================================================
    if sens_cfg:
        print("\n===== STEP 2: Sensitivity (table output) =====")
        for cc in cl:
            for algo, (param, vals) in sens_cfg.items():
                if algo in al:
                    try:
                        print(f"  [sensitivity] {cc['city']} × {algo} × {param}")
                        run_sensitivity(cc, algo, param, vals)
                    except Exception as e:
                        print(f"  [sensitivity] {cc['city']} × {algo} FAILED: {e}")
                        traceback.print_exc()
    else:
        print("\n===== STEP 2: Sensitivity =====")
        print("  skipped (sens_cfg is None)")

    # =====================================================
    # STEP 3: Stability
    # =====================================================
    if stab_runs > 1:
        print("\n===== STEP 3: Stability (table output) =====")
        for cc in cl:
            for algo in [a for a in al if a in ["Infomap", "LeidenRBC", "Spinglass", "GeoLP", "DirLou", "DCSBM"]]:
                try:
                    print(f"  [stability] {cc['city']} × {algo} × {stab_runs} runs")
                    run_stability(cc, algo, n_runs=stab_runs)
                except Exception as e:
                    print(f"  [stability] {cc['city']} × {algo} FAILED: {e}")
                    traceback.print_exc()
    else:
        print("\n===== STEP 3: Stability =====")
        print("  skipped (stab_runs <= 1)")

    # =====================================================
    # STEP 4: Fairness
    # =====================================================
    print("\n===== STEP 4: Fairness =====")
    try:
        create_algorithm_fairness_table()
    except Exception as e:
        print(f"  [fairness] FAILED: {e}")
        traceback.print_exc()

    # =====================================================
    # STEP 5: POI
    # =====================================================
    print("\n===== STEP 5: POI =====")
    try:
        run_poi_all(
            cities=[c["city"] for c in cl],
            algorithms=al
        )
    except Exception as e:
        print(f"  [POI] FAILED: {e}")
        traceback.print_exc()

    # =====================================================
    # STEP 6: Summary
    # =====================================================
    print("\n===== STEP 6: Summary =====")
    try:
        show_summary()
    except Exception as e:
        print(f"  [summary] FAILED: {e}")
        traceback.print_exc()

    # =====================================================
    # STEP 7–10: Reviewer-response modules
    # =====================================================
    if run_reviewer_modules:
        print("\n===== STEP 7: Metric Independence =====")
        try:
            analyze_metric_independence(
                cities=[c["city"] for c in cl],
                algorithms=al
            )
        except Exception as e:
            print(f"  [metric independence] FAILED: {e}")
            traceback.print_exc()

        print("\n===== STEP 8: Spatial Compactness =====")
        try:
            run_spatial_compactness_comparison(
                cities=[c["city"] for c in cl],
                algorithms=al
            )
        except Exception as e:
            print(f"  [spatial compactness] FAILED: {e}")
            traceback.print_exc()

        print("\n===== STEP 9: Scale Sensitivity =====")
        try:
            analyze_scale_sensitivity(
                cities=[c["city"] for c in cl],
                algorithms=al
            )
        except Exception as e:
            print(f"  [scale sensitivity] FAILED: {e}")
            traceback.print_exc()

        print("\n===== STEP 10: Boundary Documentation =====")
        try:
            document_city_boundaries(
                cities=[c["city"] for c in cl]
            )
        except Exception as e:
            print(f"  [boundary documentation] FAILED: {e}")
            traceback.print_exc()
    else:
        print("\n===== STEP 7–10: Reviewer-response modules =====")
        print("  skipped (run_reviewer_modules=False)")

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")


# %%
# %% Cell 11 — 执行（最终模板）

# =========================================================
# 运行模式
# 可选：
#   "test"   -> 单城市 × 单算法，快速检查
#   "fast"   -> 小规模轻量流程，优先用于调试和提速
#   "review" -> 小规模完整流程，适合审稿补充分析
#   "full"   -> 全部城市 × 全部算法，正式运行
# =========================================================

if __name__ == "__main__":

    RUN_MODE = "full"

    # =========================================================
    # 参数配置
    # =========================================================
    TEST_CITY = CITIES[0]
    TEST_ALGO = "Infomap"

    REVIEW_CITIES = ["BeiJing", "Moscow"]
    REVIEW_ALGOS = ["Infomap", "LeidenRBC", "Spinglass"]

    FAST_CITIES = ["BeiJing", "Moscow"]
    FAST_ALGOS = ["Infomap", "LeidenRBC"]   # fast 模式先不跑 Spinglass

    FULL_CITIES = [c["city"] for c in CITIES]
    FULL_ALGOS = ALGORITHMS

    SENS_CFG_REVIEW = {
        "LeidenRBC": ("resolution", [0.05, 0.1, 0.5, 1.0]),
        "Infomap": ("markov_time", [0.5, 1.0, 1.5])
    }

    SENS_CFG_FULL = {
        "LeidenRBC": ("resolution", [0.05, 0.1, 0.5, 1.0, 2.0]),
        "Infomap": ("markov_time", [0.5, 1.0, 1.5, 2.0])
    }

    # =========================================================
    # 执行入口
    # =========================================================
    if RUN_MODE == "test":
        print("\n[RUN MODE] test")
        run_one(
            TEST_CITY,
            TEST_ALGO,
            show_base=False,
            show_comm_map=False,
            show_comm_plot=True,
            show_table=True,
            show_poi=True,
            save_static_map=False
        )

    elif RUN_MODE == "fast":
        print("\n[RUN MODE] fast")
        run_full(
            cities=FAST_CITIES,
            algorithms=FAST_ALGOS,
            sens_cfg=None,                  # 关闭敏感性分析
            stab_runs=1,                    # 关闭稳定性分析
            show_maps=False,
            run_reviewer_modules=False      # 关闭 reviewer 模块
        )

    elif RUN_MODE == "review":
        print("\n[RUN MODE] review")
        run_full(
            cities=REVIEW_CITIES,
            algorithms=REVIEW_ALGOS,
            sens_cfg=SENS_CFG_REVIEW,
            stab_runs=3,
            show_maps=False,
            run_reviewer_modules=True
        )

    elif RUN_MODE == "full":
        print("\n[RUN MODE] full")
        run_full(
            cities=FULL_CITIES,
            algorithms=FULL_ALGOS,
            sens_cfg=SENS_CFG_FULL,
            stab_runs=5,
            show_maps=False,
            run_reviewer_modules=True
        )

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
