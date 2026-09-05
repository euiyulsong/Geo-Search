#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REAL GEO SEARCH BENCHMARK - FIXED METRICS VERSION

Compare:
1. Elasticsearch native geo distance
2. Elasticsearch adaptive radius
3. H3 incremental candidate generation + local exact rerank
4. H3 candidate generation + Elasticsearch exact rerank
5. Elasticsearch polygon query

Metrics:
- Top1 Accuracy
- Recall@1
- Recall@5
- Recall@20
- Candidate Recall@1
- Candidate Recall@5
- Candidate Recall@20
- Radius Recall

Important:
- Ground truth is brute-force Haversine / point-in-polygon.
- GT does NOT depend on ES or H3.
- Empty GT => NaN, NOT 1.0.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from elasticsearch import Elasticsearch, helpers

try:
    import h3
except ImportError:
    raise RuntimeError("pip install h3")


# ======================================================================================
# CONFIG
# ======================================================================================

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX = "geo_real_benchmark_metrics"

CSV_PATH = "gangnam_osm_pois.csv"
RESULT_PATH = "geo_benchmark_results_metrics.csv"
SUMMARY_PATH = "geo_benchmark_summary_metrics.csv"
SANITY_PATH = "h3_ring0_sanity_metrics.csv"

TOP_K = 20

H3_RES = 9
MAX_H3_RING = 20

WARMUP = 3
REPEAT = 20

INITIAL_RADIUS_M = 100
MAX_RADIUS_M = 20_000
RADIUS_GROWTH = 2.0

BBOX = {
    "south": 37.42,
    "west": 126.80,
    "north": 37.65,
    "east": 127.18,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "geo-search-benchmark/1.0"


# ======================================================================================
# QUERIES
# ======================================================================================

QUERIES = [

    # ------------------------------------------------------------------
    # NEAR ME
    # ------------------------------------------------------------------

    {
        "qid": "near_gangnam_restaurant",
        "query_text": "내 주변 음식점",
        "query_type": "near_me",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "near_gangnam_cafe",
        "query_text": "내 주변 카페",
        "query_type": "near_me",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["cafe"],
        "radius_m": None,
    },

    {
        "qid": "near_jamsil_restaurant",
        "query_text": "내 주변 음식점",
        "query_type": "near_me",
        "lat": 37.5133,
        "lon": 127.1002,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "near_hongdae_cafe",
        "query_text": "내 주변 카페",
        "query_type": "near_me",
        "lat": 37.5572,
        "lon": 126.9254,
        "categories": ["cafe"],
        "radius_m": None,
    },


    # ------------------------------------------------------------------
    # POINT ANCHOR
    # ------------------------------------------------------------------

    {
        "qid": "gangnam_station_restaurant",
        "query_text": "강남역 맛집",
        "query_type": "point_anchor",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "jamsil_station_restaurant",
        "query_text": "잠실역 맛집",
        "query_type": "point_anchor",
        "lat": 37.5133,
        "lon": 127.1002,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "hongdae_station_cafe",
        "query_text": "홍대입구역 카페",
        "query_type": "point_anchor",
        "lat": 37.5572,
        "lon": 126.9254,
        "categories": ["cafe"],
        "radius_m": None,
    },

    {
        "qid": "seoul_station_restaurant",
        "query_text": "서울역 맛집",
        "query_type": "point_anchor",
        "lat": 37.5547,
        "lon": 126.9706,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "yeouido_station_cafe",
        "query_text": "여의도역 카페",
        "query_type": "point_anchor",
        "lat": 37.5216,
        "lon": 126.9242,
        "categories": ["cafe"],
        "radius_m": None,
    },


    # ------------------------------------------------------------------
    # LANDMARK
    # ------------------------------------------------------------------

    {
        "qid": "coex_cafe",
        "query_text": "코엑스 카페",
        "query_type": "landmark",
        "lat": 37.5125,
        "lon": 127.0588,
        "categories": ["cafe"],
        "radius_m": None,
    },

    {
        "qid": "lotte_world_restaurant",
        "query_text": "롯데월드 맛집",
        "query_type": "landmark",
        "lat": 37.5111,
        "lon": 127.0982,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "gyeongbokgung_restaurant",
        "query_text": "경복궁 맛집",
        "query_type": "landmark",
        "lat": 37.5796,
        "lon": 126.9770,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": None,
    },

    {
        "qid": "seoul_forest_cafe",
        "query_text": "서울숲 카페",
        "query_type": "landmark",
        "lat": 37.5444,
        "lon": 127.0374,
        "categories": ["cafe"],
        "radius_m": None,
    },


    # ------------------------------------------------------------------
    # EXPLICIT RADIUS
    # ------------------------------------------------------------------

    {
        "qid": "gangnam_200m",
        "query_text": "강남역 200m 맛집",
        "query_type": "explicit_radius",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": 200.0,
    },

    {
        "qid": "gangnam_500m",
        "query_text": "강남역 500m 맛집",
        "query_type": "explicit_radius",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": 500.0,
    },

    {
        "qid": "gangnam_1km",
        "query_text": "강남역 1km 맛집",
        "query_type": "explicit_radius",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": 1000.0,
    },

    {
        "qid": "gangnam_3km",
        "query_text": "강남역 3km 맛집",
        "query_type": "explicit_radius",
        "lat": 37.4979,
        "lon": 127.0276,
        "categories": ["restaurant", "fast_food", "food_court"],
        "radius_m": 3000.0,
    },

    {
        "qid": "hongdae_500m_cafe",
        "query_text": "홍대입구역 500m 카페",
        "query_type": "explicit_radius",
        "lat": 37.5572,
        "lon": 126.9254,
        "categories": ["cafe"],
        "radius_m": 500.0,
    },

    {
        "qid": "coex_1km_cafe",
        "query_text": "코엑스 1km 카페",
        "query_type": "explicit_radius",
        "lat": 37.5125,
        "lon": 127.0588,
        "categories": ["cafe"],
        "radius_m": 1000.0,
    },
]


POLYGON_QUERIES = [

    {
        "qid": "gangnam_gu_restaurant",
        "query_text": "강남구 맛집",
        "query_type": "admin_area",
        "place": "강남구, 서울특별시, 대한민국",
        "categories": ["restaurant", "fast_food", "food_court"],
    },

    {
        "qid": "songpa_gu_restaurant",
        "query_text": "송파구 맛집",
        "query_type": "admin_area",
        "place": "송파구, 서울특별시, 대한민국",
        "categories": ["restaurant", "fast_food", "food_court"],
    },

    {
        "qid": "mapo_gu_cafe",
        "query_text": "마포구 카페",
        "query_type": "admin_area",
        "place": "마포구, 서울특별시, 대한민국",
        "categories": ["cafe"],
    },

    {
        "qid": "jongno_gu_restaurant",
        "query_text": "종로구 맛집",
        "query_type": "admin_area",
        "place": "종로구, 서울특별시, 대한민국",
        "categories": ["restaurant", "fast_food", "food_court"],
    },

    {
        "qid": "yeongdeungpo_gu_cafe",
        "query_text": "영등포구 카페",
        "query_type": "admin_area",
        "place": "영등포구, 서울특별시, 대한민국",
        "categories": ["cafe"],
    },

    {
        "qid": "yeoksam_dong",
        "query_text": "역삼동 맛집",
        "query_type": "neighborhood",
        "place": "역삼동, 강남구, 서울특별시, 대한민국",
        "categories": ["restaurant", "fast_food", "food_court"],
    },

    {
        "qid": "samseong_dong",
        "query_text": "삼성동 카페",
        "query_type": "neighborhood",
        "place": "삼성동, 강남구, 서울특별시, 대한민국",
        "categories": ["cafe"],
    },

    {
        "qid": "seogyo_dong",
        "query_text": "서교동 카페",
        "query_type": "neighborhood",
        "place": "서교동, 마포구, 서울특별시, 대한민국",
        "categories": ["cafe"],
    },

    {
        "qid": "jamsil_dong",
        "query_text": "잠실동 맛집",
        "query_type": "neighborhood",
        "place": "잠실동, 송파구, 서울특별시, 대한민국",
        "categories": ["restaurant", "fast_food", "food_court"],
    },
]


# ======================================================================================
# H3 VERSION COMPAT
# ======================================================================================

def h3_cell(lat, lon, res):
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)

    return h3.geo_to_h3(lat, lon, res)


def h3_disk(cell, ring):
    if hasattr(h3, "grid_disk"):
        return list(h3.grid_disk(cell, ring))

    return list(h3.k_ring(cell, ring))


def h3_exact_ring(cell, ring):

    if ring == 0:
        return [cell]

    if hasattr(h3, "grid_ring"):
        try:
            return list(h3.grid_ring(cell, ring))
        except Exception:
            pass

    outer = set(h3_disk(cell, ring))
    inner = set(h3_disk(cell, ring - 1))

    return list(outer - inner)


# ======================================================================================
# DISTANCE
# ======================================================================================

EARTH_RADIUS_M = 6_371_008.8


def haversine_vec(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(
        np.asarray(lat2, dtype=np.float64)
    )

    lon2 = np.radians(
        np.asarray(lon2, dtype=np.float64)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(a))
    )


# ======================================================================================
# DATA
# ======================================================================================

def download_osm():

    print()
    print("=" * 100)
    print("DOWNLOADING REAL OSM POIs")
    print("=" * 100)

    s = BBOX["south"]
    w = BBOX["west"]
    n = BBOX["north"]
    e = BBOX["east"]

    query = f"""
    [out:json][timeout:180];

    (
      node["amenity"~"restaurant|cafe|fast_food|food_court"]({s},{w},{n},{e});
      way["amenity"~"restaurant|cafe|fast_food|food_court"]({s},{w},{n},{e});
      relation["amenity"~"restaurant|cafe|fast_food|food_court"]({s},{w},{n},{e});
    );

    out center tags;
    """

    r = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=240,
    )

    r.raise_for_status()

    elements = r.json()["elements"]

    print(
        "raw OSM features:",
        len(elements),
    )

    rows = []

    for x in elements:

        tags = x.get(
            "tags",
            {},
        )

        lat = x.get("lat")
        lon = x.get("lon")

        if lat is None or lon is None:

            center = x.get(
                "center",
                {},
            )

            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        category = tags.get(
            "amenity"
        )

        if category not in {
            "restaurant",
            "cafe",
            "fast_food",
            "food_court",
        }:
            continue

        rows.append(
            {
                "poi_id": f'{x["type"]}_{x["id"]}',
                "osm_type": x["type"],
                "osm_id": x["id"],
                "name": tags.get(
                    "name",
                    "",
                ),
                "category": category,
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates("poi_id")
        .dropna(
            subset=[
                "lat",
                "lon",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "usable POIs:",
        len(df),
    )

    print()

    print(
        df["category"]
        .value_counts()
    )

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    return df


def load_data():

    if os.path.exists(CSV_PATH):

        print(
            f"[load] {CSV_PATH}"
        )

        df = pd.read_csv(
            CSV_PATH
        )

        if "poi_id" not in df.columns:

            if "id" in df.columns:
                df["poi_id"] = (
                    df["id"]
                    .astype(str)
                )

            else:
                df["poi_id"] = [
                    f"poi_{i}"
                    for i in range(len(df))
                ]

        return df

    return download_osm()


# ======================================================================================
# METRICS
# ======================================================================================

def recall_at_k(
    pred_ids,
    gt_ids,
    k,
):

    gt = list(
        gt_ids[:k]
    )

    if len(gt) == 0:
        return np.nan

    pred = list(
        pred_ids[:k]
    )

    return (
        len(
            set(pred)
            & set(gt)
        )
        / len(gt)
    )


def top1_accuracy(
    pred_ids,
    gt_ids,
):

    if len(gt_ids) == 0:
        return np.nan

    if len(pred_ids) == 0:
        return 0.0

    return float(
        pred_ids[0]
        == gt_ids[0]
    )


def hit_at_k(
    pred_ids,
    gt_ids,
    k,
):

    gt = set(
        gt_ids[:k]
    )

    if len(gt) == 0:
        return np.nan

    pred = set(
        pred_ids[:k]
    )

    return float(
        len(pred & gt) > 0
    )


def candidate_recall_at_k(
    candidate_ids,
    gt_ids,
    k,
):

    gt = set(
        gt_ids[:k]
    )

    if len(gt) == 0:
        return np.nan

    candidates = set(
        candidate_ids
    )

    return (
        len(
            candidates
            & gt
        )
        / len(gt)
    )


def set_recall(
    pred_ids,
    gt_ids,
):

    gt = set(gt_ids)

    if len(gt) == 0:
        return np.nan

    return (
        len(
            set(pred_ids)
            & gt
        )
        / len(gt)
    )


# ======================================================================================
# GROUND TRUTH
# ======================================================================================

def category_filter(
    df,
    categories,
):

    return df[
        df["category"]
        .isin(categories)
    ].copy()


def brute_force_ground_truth(
    df,
    q,
    k=TOP_K,
):

    x = category_filter(
        df,
        q["categories"],
    )

    x = x.copy()

    x["distance_m"] = haversine_vec(
        q["lat"],
        q["lon"],
        x["lat"].values,
        x["lon"].values,
    )

    if (
        q.get("radius_m")
        is not None
    ):

        x = x[
            x["distance_m"]
            <= q["radius_m"]
        ].copy()

    x = x.sort_values(
        [
            "distance_m",
            "poi_id",
        ]
    )

    topk = x.head(k)

    return {
        "topk_ids": (
            topk["poi_id"]
            .astype(str)
            .tolist()
        ),

        "topk_df": topk,

        "all_ids": (
            x["poi_id"]
            .astype(str)
            .tolist()
        ),

        "all_df": x,

        "total_matches": len(x),
    }


# ======================================================================================
# H3 INDEX
# ======================================================================================

def build_h3_lookup(df):

    print()
    print("=" * 100)
    print("BUILD H3 LOOKUP")
    print("=" * 100)

    lookup = defaultdict(list)
    cells = []

    for idx, row in df.iterrows():

        cell = h3_cell(
            float(row["lat"]),
            float(row["lon"]),
            H3_RES,
        )

        cells.append(cell)
        lookup[cell].append(idx)

    df = df.copy()

    df["h3"] = cells

    print(
        "H3 resolution:",
        H3_RES,
    )

    print(
        "unique cells:",
        len(lookup),
    )

    return (
        df,
        lookup,
    )


# ======================================================================================
# ELASTICSEARCH
# ======================================================================================

def connect_es():

    es = Elasticsearch(
        ES_URL
    )

    info = es.info()

    print(
        "[ES]",
        info["version"]["number"],
        ES_URL,
    )

    return es


def create_es_index(
    es,
    df,
):

    print()
    print("=" * 100)
    print("BUILD ELASTICSEARCH INDEX")
    print("=" * 100)

    if es.indices.exists(
        index=INDEX
    ):
        es.indices.delete(
            index=INDEX
        )

    es.indices.create(
        index=INDEX,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        mappings={
            "properties": {
                "poi_id": {
                    "type": "keyword"
                },
                "name": {
                    "type": "text"
                },
                "category": {
                    "type": "keyword"
                },
                "h3": {
                    "type": "keyword"
                },
                "location": {
                    "type": "geo_point"
                },
            }
        },
    )

    actions = []

    for _, row in df.iterrows():

        actions.append(
            {
                "_index": INDEX,
                "_id": str(
                    row["poi_id"]
                ),
                "_source": {
                    "poi_id": str(
                        row["poi_id"]
                    ),

                    "name": str(
                        row.get(
                            "name",
                            "",
                        )
                    ),

                    "category": str(
                        row["category"]
                    ),

                    "h3": str(
                        row["h3"]
                    ),

                    "location": {
                        "lat": float(
                            row["lat"]
                        ),
                        "lon": float(
                            row["lon"]
                        ),
                    },
                },
            }
        )

    helpers.bulk(
        es,
        actions,
        chunk_size=2000,
        request_timeout=120,
    )

    es.indices.refresh(
        index=INDEX
    )

    print(
        "indexed docs:",
        es.count(
            index=INDEX
        )["count"],
    )


# ======================================================================================
# ES NATIVE
# ======================================================================================

def es_native_distance(
    es,
    q,
    k=TOP_K,
):

    filters = [
        {
            "terms": {
                "category":
                q["categories"]
            }
        }
    ]

    if (
        q.get("radius_m")
        is not None
    ):

        filters.append(
            {
                "geo_distance": {
                    "distance":
                    f'{q["radius_m"]}m',

                    "location": {
                        "lat": q["lat"],
                        "lon": q["lon"],
                    },
                }
            }
        )

    body = {

        "query": {
            "bool": {
                "filter": filters
            }
        },

        "sort": [

            {
                "_geo_distance": {

                    "location": {
                        "lat": q["lat"],
                        "lon": q["lon"],
                    },

                    "order": "asc",
                    "unit": "m",
                    "distance_type": "arc",
                }
            },

            {
                "poi_id": {
                    "order": "asc"
                }
            },
        ],

        "size": k,

        "track_total_hits": False,
    }

    r = es.search(
        index=INDEX,
        body=body,
    )

    ids = [
        hit["_source"]["poi_id"]
        for hit in r["hits"]["hits"]
    ]

    return {
        "ids": ids,
        "candidate_ids": None,
        "candidate_count": None,
        "rings": None,
        "es_queries": 1,
    }


# ======================================================================================
# ADAPTIVE RADIUS
# ======================================================================================

def es_count_radius(
    es,
    q,
    radius_m,
):

    filters = [

        {
            "terms": {
                "category":
                q["categories"]
            }
        },

        {
            "geo_distance": {

                "distance":
                f"{radius_m}m",

                "location": {
                    "lat": q["lat"],
                    "lon": q["lon"],
                },
            }
        },
    ]

    r = es.count(
        index=INDEX,
        query={
            "bool": {
                "filter": filters
            }
        },
    )

    return r["count"]


def es_adaptive_radius(
    es,
    q,
    k=TOP_K,
):

    if (
        q.get("radius_m")
        is not None
    ):

        radius = q["radius_m"]

        count = es_count_radius(
            es,
            q,
            radius,
        )

        result = es_native_distance(
            es,
            q,
            k,
        )

        result[
            "candidate_count"
        ] = count

        result[
            "es_queries"
        ] = 2

        return result

    radius = INITIAL_RADIUS_M
    queries = 0

    while True:

        count = es_count_radius(
            es,
            q,
            radius,
        )

        queries += 1

        if count >= k:
            break

        if radius >= MAX_RADIUS_M:
            break

        radius = min(
            radius
            * RADIUS_GROWTH,
            MAX_RADIUS_M,
        )

    filters = [

        {
            "terms": {
                "category":
                q["categories"]
            }
        },

        {
            "geo_distance": {

                "distance":
                f"{radius}m",

                "location": {
                    "lat": q["lat"],
                    "lon": q["lon"],
                },
            }
        },
    ]

    body = {

        "query": {
            "bool": {
                "filter": filters
            }
        },

        "sort": [

            {
                "_geo_distance": {

                    "location": {
                        "lat": q["lat"],
                        "lon": q["lon"],
                    },

                    "order": "asc",
                    "unit": "m",
                    "distance_type": "arc",
                }
            },

            {
                "poi_id": {
                    "order": "asc"
                }
            },
        ],

        "size": k,

        "track_total_hits": False,
    }

    r = es.search(
        index=INDEX,
        body=body,
    )

    queries += 1

    ids = [
        hit["_source"]["poi_id"]
        for hit in r["hits"]["hits"]
    ]

    return {
        "ids": ids,
        "candidate_ids": None,
        "candidate_count": count,
        "rings": None,
        "es_queries": queries,
    }


# ======================================================================================
# H3 CANDIDATES
# ======================================================================================

def get_rows_for_cells(
    df,
    lookup,
    cells,
    categories,
):

    idxs = []

    for cell in cells:
        idxs.extend(
            lookup.get(
                cell,
                [],
            )
        )

    if not idxs:
        return df.iloc[
            0:0
        ].copy()

    x = df.loc[
        list(
            set(idxs)
        )
    ]

    return x[
        x["category"]
        .isin(categories)
    ].copy()


def h3_incremental_candidates(
    df,
    lookup,
    q,
    k=TOP_K,
    max_ring=MAX_H3_RING,
    force_ring=None,
):

    center = h3_cell(
        q["lat"],
        q["lon"],
        H3_RES,
    )

    candidate_cells = set()

    # Explicit radius
    if (
        q.get("radius_m")
        is not None
        and force_ring is None
    ):

        target_radius = (
            q["radius_m"]
        )

        final_ring = 0

        for ring in range(
            max_ring + 1
        ):

            candidate_cells.update(
                h3_exact_ring(
                    center,
                    ring,
                )
            )

            candidates = get_rows_for_cells(
                df,
                lookup,
                candidate_cells,
                q["categories"],
            )

            final_ring = ring

            if len(candidates) == 0:
                continue

            distances = haversine_vec(
                q["lat"],
                q["lon"],
                candidates["lat"].values,
                candidates["lon"].values,
            )

            # conservative-ish stop
            if (
                ring > 0
                and distances.max()
                > target_radius * 1.5
            ):
                break

        return (
            candidates,
            final_ring,
        )

    final_ring = 0

    for ring in range(
        max_ring + 1
    ):

        candidate_cells.update(
            h3_exact_ring(
                center,
                ring,
            )
        )

        candidates = get_rows_for_cells(
            df,
            lookup,
            candidate_cells,
            q["categories"],
        )

        final_ring = ring

        if (
            force_ring
            is not None
        ):

            if ring >= force_ring:
                break

        else:

            if len(candidates) >= k:
                break

    return (
        candidates,
        final_ring,
    )


# ======================================================================================
# H3 LOCAL SEARCH
# ======================================================================================

def h3_incremental_search(
    df,
    lookup,
    q,
    k=TOP_K,
    force_ring=None,
):

    candidates, ring = (
        h3_incremental_candidates(
            df,
            lookup,
            q,
            k,
            force_ring=force_ring,
        )
    )

    if len(candidates) == 0:

        return {
            "ids": [],
            "candidate_ids": [],
            "candidate_count": 0,
            "rings": ring,
            "es_queries": 0,
        }

    candidates = candidates.copy()

    candidates[
        "distance_m"
    ] = haversine_vec(
        q["lat"],
        q["lon"],
        candidates["lat"].values,
        candidates["lon"].values,
    )

    if (
        q.get("radius_m")
        is not None
    ):

        candidates = candidates[
            candidates["distance_m"]
            <= q["radius_m"]
        ].copy()

    candidate_ids = (
        candidates["poi_id"]
        .astype(str)
        .tolist()
    )

    ranked = candidates.sort_values(
        [
            "distance_m",
            "poi_id",
        ]
    )

    ids = (
        ranked
        .head(k)["poi_id"]
        .astype(str)
        .tolist()
    )

    return {
        "ids": ids,
        "candidate_ids": candidate_ids,
        "candidate_count": len(
            candidate_ids
        ),
        "rings": ring,
        "es_queries": 0,
    }


# ======================================================================================
# H3 + ES
# ======================================================================================

def h3_es_hybrid(
    es,
    df,
    lookup,
    q,
    k=TOP_K,
):

    candidates, ring = (
        h3_incremental_candidates(
            df,
            lookup,
            q,
            k,
        )
    )

    if len(candidates) == 0:

        return {
            "ids": [],
            "candidate_ids": [],
            "candidate_count": 0,
            "rings": ring,
            "es_queries": 0,
        }

    candidate_ids = (
        candidates["poi_id"]
        .astype(str)
        .tolist()
    )

    filters = [
        {
            "ids": {
                "values":
                candidate_ids
            }
        }
    ]

    if (
        q.get("radius_m")
        is not None
    ):

        filters.append(
            {
                "geo_distance": {

                    "distance":
                    f'{q["radius_m"]}m',

                    "location": {
                        "lat": q["lat"],
                        "lon": q["lon"],
                    },
                }
            }
        )

    body = {

        "query": {
            "bool": {
                "filter":
                filters
            }
        },

        "sort": [

            {
                "_geo_distance": {

                    "location": {
                        "lat": q["lat"],
                        "lon": q["lon"],
                    },

                    "order": "asc",
                    "unit": "m",
                    "distance_type": "arc",
                }
            },

            {
                "poi_id": {
                    "order": "asc"
                }
            },
        ],

        "size": k,

        "track_total_hits": False,
    }

    r = es.search(
        index=INDEX,
        body=body,
    )

    ids = [
        hit["_source"]["poi_id"]
        for hit in r["hits"]["hits"]
    ]

    return {
        "ids": ids,
        "candidate_ids": candidate_ids,
        "candidate_count": len(
            candidate_ids
        ),
        "rings": ring,
        "es_queries": 1,
    }


# ======================================================================================
# POLYGON
# ======================================================================================

def fetch_polygons(
    place,
):

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {
        "q": place,
        "format": "json",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "limit": 10,
    }

    r = requests.get(
        url,
        params=params,
        headers={
            "User-Agent":
            USER_AGENT
        },
        timeout=60,
    )

    r.raise_for_status()

    results = r.json()

    polygons = []

    print(
        f"[Nominatim] {place}"
    )

    for i, item in enumerate(
        results
    ):

        geo = item.get(
            "geojson"
        )

        geo_type = (
            geo.get("type")
            if geo
            else None
        )

        print(
            f" [{i}]",
            item.get(
                "display_name"
            ),
            "|",
            geo_type,
        )

        if geo_type in (
            "Polygon",
            "MultiPolygon",
        ):
            polygons.append(
                geo
            )

    if not polygons:

        raise RuntimeError(
            f"No Polygon/MultiPolygon: {place}"
        )

    return polygons


def point_in_ring(
    lon,
    lat,
    ring,
):

    inside = False

    j = len(ring) - 1

    for i in range(
        len(ring)
    ):

        xi, yi = ring[i]
        xj, yj = ring[j]

        intersects = (
            (yi > lat)
            != (yj > lat)
        ) and (
            lon
            <
            (
                (xj - xi)
                * (lat - yi)
                / ((yj - yi) + 1e-15)
                + xi
            )
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def point_in_polygon_geojson(
    lon,
    lat,
    geo,
):

    if geo["type"] == "Polygon":

        polygons = [
            geo["coordinates"]
        ]

    elif geo["type"] == "MultiPolygon":

        polygons = (
            geo["coordinates"]
        )

    else:

        return False

    for poly in polygons:

        outer = poly[0]

        if not point_in_ring(
            lon,
            lat,
            outer,
        ):
            continue

        in_hole = False

        for hole in poly[1:]:

            if point_in_ring(
                lon,
                lat,
                hole,
            ):

                in_hole = True
                break

        if not in_hole:
            return True

    return False


def point_in_any_polygon(
    lon,
    lat,
    geos,
):

    for geo in geos:

        if point_in_polygon_geojson(
            lon,
            lat,
            geo,
        ):
            return True

    return False


def polygons_center(
    geos,
):

    coords = []

    for geo in geos:

        if geo["type"] == "Polygon":

            coords.extend(
                geo[
                    "coordinates"
                ][0]
            )

        else:

            for p in geo[
                "coordinates"
            ]:

                coords.extend(
                    p[0]
                )

    arr = np.asarray(
        coords
    )

    return (
        float(
            arr[:, 1].mean()
        ),
        float(
            arr[:, 0].mean()
        ),
    )


def brute_force_polygon_ground_truth(
    df,
    q,
    geos,
    k=TOP_K,
):

    x = category_filter(
        df,
        q["categories"],
    )

    mask = []

    for _, row in x.iterrows():

        mask.append(
            point_in_any_polygon(
                float(
                    row["lon"]
                ),
                float(
                    row["lat"]
                ),
                geos,
            )
        )

    x = x[
        np.asarray(mask)
    ].copy()

    center_lat, center_lon = (
        polygons_center(
            geos
        )
    )

    if len(x):

        x["distance_m"] = (
            haversine_vec(
                center_lat,
                center_lon,
                x["lat"].values,
                x["lon"].values,
            )
        )

        x = x.sort_values(
            [
                "distance_m",
                "poi_id",
            ]
        )

    return {

        "topk_ids": (
            x.head(k)[
                "poi_id"
            ]
            .astype(str)
            .tolist()
        ),

        "all_ids": (
            x["poi_id"]
            .astype(str)
            .tolist()
        ),

        "total_matches":
        len(x),

        "center_lat":
        center_lat,

        "center_lon":
        center_lon,
    }


def es_polygon_search(
    es,
    q,
    geos,
    center_lat,
    center_lon,
    k=TOP_K,
):

    should = []

    for geo in geos:

        should.append(
            {
                "geo_shape": {

                    "location": {

                        "shape":
                        geo,

                        "relation":
                        "intersects",
                    }
                }
            }
        )

    body = {

        "query": {

            "bool": {

                "filter": [

                    {
                        "terms": {
                            "category":
                            q[
                                "categories"
                            ]
                        }
                    },

                    {
                        "bool": {

                            "should":
                            should,

                            "minimum_should_match":
                            1,
                        }
                    },
                ]
            }
        },

        "sort": [

            {
                "_geo_distance": {

                    "location": {
                        "lat":
                        center_lat,
                        "lon":
                        center_lon,
                    },

                    "order":
                    "asc",

                    "unit":
                    "m",

                    "distance_type":
                    "arc",
                }
            },

            {
                "poi_id": {
                    "order":
                    "asc"
                }
            },
        ],

        "size":
        k,

        "track_total_hits":
        False,
    }

    r = es.search(
        index=INDEX,
        body=body,
    )

    ids = [
        h["_source"]["poi_id"]
        for h in r["hits"]["hits"]
    ]

    return {

        "ids":
        ids,

        "candidate_ids":
        None,

        "candidate_count":
        None,

        "rings":
        None,

        "es_queries":
        1,
    }


# ======================================================================================
# BENCHMARK
# ======================================================================================

def percentile(
    xs,
    p,
):

    return float(
        np.percentile(
            np.asarray(xs),
            p,
        )
    )


def benchmark_call(
    fn,
):

    result = None

    for _ in range(
        WARMUP
    ):

        result = fn()

    times = []

    for _ in range(
        REPEAT
    ):

        t0 = (
            time.perf_counter()
        )

        result = fn()

        times.append(
            (
                time.perf_counter()
                - t0
            )
            * 1000.0
        )

    return {

        "result":
        result,

        "p50_ms":
        percentile(
            times,
            50,
        ),

        "p95_ms":
        percentile(
            times,
            95,
        ),

        "p99_ms":
        percentile(
            times,
            99,
        ),

        "mean_ms":
        float(
            np.mean(times)
        ),
    }


# ======================================================================================
# POINT QUERIES
# ======================================================================================

def run_point_query(
    es,
    df,
    lookup,
    q,
):

    print()
    print("=" * 100)

    print(
        q["qid"],
        "|",
        q["query_text"],
    )

    print("=" * 100)

    gt = brute_force_ground_truth(
        df,
        q,
    )

    print(
        "GT total:",
        gt["total_matches"],
    )

    methods = {

        "ES_NATIVE_DISTANCE":
        lambda:
        es_native_distance(
            es,
            q,
        ),

        "ES_ADAPTIVE_RADIUS":
        lambda:
        es_adaptive_radius(
            es,
            q,
        ),

        "H3_INCREMENTAL":
        lambda:
        h3_incremental_search(
            df,
            lookup,
            q,
        ),

        "H3_ES_HYBRID":
        lambda:
        h3_es_hybrid(
            es,
            df,
            lookup,
            q,
        ),
    }

    rows = []

    for name, fn in methods.items():

        b = benchmark_call(
            fn
        )

        r = b["result"]

        recall1 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            1,
        )

        recall5 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            5,
        )

        recall20 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            20,
        )

        top1 = top1_accuracy(
            r["ids"],
            gt["topk_ids"],
        )

        hit5 = hit_at_k(
            r["ids"],
            gt["topk_ids"],
            5,
        )

        cand1 = np.nan
        cand5 = np.nan
        cand20 = np.nan

        if (
            r["candidate_ids"]
            is not None
        ):

            cand1 = (
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt[
                        "topk_ids"
                    ],
                    1,
                )
            )

            cand5 = (
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt[
                        "topk_ids"
                    ],
                    5,
                )
            )

            cand20 = (
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt[
                        "topk_ids"
                    ],
                    20,
                )
            )

        radius_recall = (
            np.nan
        )

        if (
            q.get(
                "radius_m"
            )
            is not None
            and
            r[
                "candidate_ids"
            ]
            is not None
        ):

            radius_recall = (
                set_recall(
                    r[
                        "candidate_ids"
                    ],
                    gt[
                        "all_ids"
                    ],
                )
            )

        rows.append(
            {

                "qid":
                q["qid"],

                "query_text":
                q[
                    "query_text"
                ],

                "query_type":
                q[
                    "query_type"
                ],

                "method":
                name,

                "p50_ms":
                b["p50_ms"],

                "p95_ms":
                b["p95_ms"],

                "p99_ms":
                b["p99_ms"],

                "mean_ms":
                b["mean_ms"],

                "top1_acc":
                top1,

                "recall@1":
                recall1,

                "recall@5":
                recall5,

                "recall@20":
                recall20,

                "hit@5":
                hit5,

                "candidate_recall@1":
                cand1,

                "candidate_recall@5":
                cand5,

                "candidate_recall@20":
                cand20,

                "radius_recall":
                radius_recall,

                "gt_total":
                gt[
                    "total_matches"
                ],

                "candidate_count":
                r[
                    "candidate_count"
                ],

                "rings":
                r["rings"],

                "es_queries":
                r[
                    "es_queries"
                ],
            }
        )

        print(
            f"{name:24s}",
            f"top1={top1}",
            f"R1={recall1}",
            f"R5={recall5}",
            f"R20={recall20}",
        )

    return rows


# ======================================================================================
# POLYGON QUERIES
# ======================================================================================

def run_polygon_queries(
    es,
    df,
):

    rows = []

    for q in (
        POLYGON_QUERIES
    ):

        print()
        print("=" * 100)

        print(
            q["qid"],
            "|",
            q["query_text"],
        )

        print("=" * 100)

        try:

            geos = fetch_polygons(
                q["place"]
            )

        except Exception as e:

            print(
                "[skip]",
                e,
            )

            continue

        gt = (
            brute_force_polygon_ground_truth(
                df,
                q,
                geos,
            )
        )

        print(
            "GT polygon matches:",
            gt[
                "total_matches"
            ],
        )

        b = benchmark_call(
            lambda:
            es_polygon_search(
                es,
                q,
                geos,
                gt[
                    "center_lat"
                ],
                gt[
                    "center_lon"
                ],
            )
        )

        r = b["result"]

        top1 = top1_accuracy(
            r["ids"],
            gt["topk_ids"],
        )

        recall1 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            1,
        )

        recall5 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            5,
        )

        recall20 = recall_at_k(
            r["ids"],
            gt["topk_ids"],
            20,
        )

        rows.append(
            {

                "qid":
                q["qid"],

                "query_text":
                q[
                    "query_text"
                ],

                "query_type":
                q[
                    "query_type"
                ],

                "method":
                "ES_NATIVE_POLYGON",

                "p50_ms":
                b["p50_ms"],

                "p95_ms":
                b["p95_ms"],

                "p99_ms":
                b["p99_ms"],

                "mean_ms":
                b["mean_ms"],

                "top1_acc":
                top1,

                "recall@1":
                recall1,

                "recall@5":
                recall5,

                "recall@20":
                recall20,

                "hit@5":
                hit_at_k(
                    r["ids"],
                    gt[
                        "topk_ids"
                    ],
                    5,
                ),

                "candidate_recall@1":
                np.nan,

                "candidate_recall@5":
                np.nan,

                "candidate_recall@20":
                np.nan,

                "radius_recall":
                np.nan,

                "gt_total":
                gt[
                    "total_matches"
                ],

                "candidate_count":
                np.nan,

                "rings":
                np.nan,

                "es_queries":
                1,
            }
        )

    return rows


# ======================================================================================
# H3 SANITY
# ======================================================================================

def run_h3_ring0_sanity(
    df,
    lookup,
):

    print()
    print("=" * 100)
    print("H3 RING=0 SANITY CHECK")
    print("=" * 100)

    rows = []

    for q in QUERIES:

        if (
            q.get(
                "radius_m"
            )
            is not None
        ):
            continue

        gt = (
            brute_force_ground_truth(
                df,
                q,
            )
        )

        r = (
            h3_incremental_search(
                df,
                lookup,
                q,
                force_ring=0,
            )
        )

        rows.append(
            {

                "qid":
                q["qid"],

                "query":
                q[
                    "query_text"
                ],

                "candidates":
                r[
                    "candidate_count"
                ],

                "top1_acc":
                top1_accuracy(
                    r["ids"],
                    gt[
                        "topk_ids"
                    ],
                ),

                "recall@5":
                recall_at_k(
                    r["ids"],
                    gt[
                        "topk_ids"
                    ],
                    5,
                ),

                "recall@20":
                recall_at_k(
                    r["ids"],
                    gt[
                        "topk_ids"
                    ],
                    20,
                ),
            }
        )

    sanity = pd.DataFrame(
        rows
    )

    print(
        sanity
        .round(4)
        .to_string(
            index=False
        )
    )

    sanity.to_csv(
        SANITY_PATH,
        index=False,
    )

    return sanity


# ======================================================================================
# SUMMARY
# ======================================================================================

def build_summary(
    result,
):

    # gt_total == 0 query 제외
    valid = result[
        result["gt_total"] > 0
    ].copy()

    summary = (

        valid
        .groupby(
            "method",
            dropna=False,
        )
        .agg(

            queries=(
                "qid",
                "count",
            ),

            p50_ms=(
                "p50_ms",
                "median",
            ),

            p95_ms=(
                "p95_ms",
                "median",
            ),

            top1_acc=(
                "top1_acc",
                "mean",
            ),

            recall_1=(
                "recall@1",
                "mean",
            ),

            recall_5=(
                "recall@5",
                "mean",
            ),

            recall_20=(
                "recall@20",
                "mean",
            ),

            candidate_recall_5=(
                "candidate_recall@5",
                "mean",
            ),

            candidate_recall_20=(
                "candidate_recall@20",
                "mean",
            ),

        )

        .reset_index()
    )

    return summary


# ======================================================================================
# MAIN
# ======================================================================================

def main():

    np.random.seed(42)

    df = load_data()

    print()
    print("=" * 100)
    print("DATA")
    print("=" * 100)

    print(
        "POIs:",
        len(df),
    )

    print()

    print(
        df[
            "category"
        ]
        .value_counts()
    )

    df, lookup = (
        build_h3_lookup(
            df
        )
    )

    es = connect_es()

    create_es_index(
        es,
        df,
    )

    # ------------------------------------------------------------------
    # SANITY
    # ------------------------------------------------------------------

    run_h3_ring0_sanity(
        df,
        lookup,
    )

    # ------------------------------------------------------------------
    # POINT
    # ------------------------------------------------------------------

    all_rows = []

    for q in QUERIES:

        all_rows.extend(
            run_point_query(
                es,
                df,
                lookup,
                q,
            )
        )

    # ------------------------------------------------------------------
    # POLYGON
    # ------------------------------------------------------------------

    all_rows.extend(
        run_polygon_queries(
            es,
            df,
        )
    )

    result = pd.DataFrame(
        all_rows
    )

    result.to_csv(
        RESULT_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    cols = [

        "qid",
        "query_text",
        "query_type",
        "method",

        "p50_ms",
        "p95_ms",
        "p99_ms",

        "top1_acc",
        "recall@1",
        "recall@5",
        "recall@20",

        "candidate_recall@1",
        "candidate_recall@5",
        "candidate_recall@20",

        "radius_recall",

        "gt_total",
        "candidate_count",
        "rings",
        "es_queries",
    ]

    print()
    print("=" * 170)
    print("FINAL RESULTS")
    print("=" * 170)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        400,
    ):

        print(
            result[
                cols
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    summary = (
        build_summary(
            result
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 120)
    print("SUMMARY - gt_total > 0 ONLY")
    print("=" * 120)

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # QUERY TYPE SUMMARY
    # ------------------------------------------------------------------

    valid = result[
        result["gt_total"] > 0
    ].copy()

    by_type = (

        valid
        .groupby(
            [
                "query_type",
                "method",
            ]
        )
        .agg(

            n=(
                "qid",
                "count",
            ),

            p50_ms=(
                "p50_ms",
                "median",
            ),

            top1=(
                "top1_acc",
                "mean",
            ),

            recall5=(
                "recall@5",
                "mean",
            ),

            recall20=(
                "recall@20",
                "mean",
            ),

        )

        .reset_index()
    )

    print()
    print("=" * 120)
    print("BY QUERY TYPE")
    print("=" * 120)

    print(
        by_type
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("saved:")
    print(
        " ",
        RESULT_PATH,
    )
    print(
        " ",
        SUMMARY_PATH,
    )
    print(
        " ",
        SANITY_PATH,
    )


if __name__ == "__main__":
    main()
