#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOREA LARGE-SCALE GEO SEARCH BENCHMARK

Input:
- korea_osm_pois.csv
- korea_geo_queries.csv

Compare:
1. ES_NATIVE_DISTANCE
2. ES_ADAPTIVE_RADIUS
3. H3_INCREMENTAL
4. H3_ES_HYBRID

Evaluate:
- Top1 Accuracy
- Recall@1 / @5 / @20 / @100 / @500
- Candidate Recall
- Radius Recall
- Latency p50 / p95 / p99
- Candidate count
- H3 rings
- ES query count

Breakdowns:
- by K
- by density
- by category frequency
- by radius
- by query type
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from elasticsearch import Elasticsearch, helpers

try:
    import h3
except ImportError:
    raise RuntimeError("pip install h3")


# ======================================================================================
# CONFIG
# ======================================================================================

POI_CSV = "korea_osm_pois.csv"
QUERY_CSV = "korea_geo_queries.csv"

ES_URL = os.environ.get(
    "ES_URL",
    "http://localhost:9200",
)

INDEX = "korea_geo_benchmark"

RESULT_CSV = "korea_geo_benchmark_results.csv"
SUMMARY_CSV = "korea_geo_benchmark_summary.csv"

BY_K_CSV = "korea_geo_by_k.csv"
BY_DENSITY_CSV = "korea_geo_by_density.csv"
BY_CATEGORY_CSV = "korea_geo_by_category_freq.csv"
BY_RADIUS_CSV = "korea_geo_by_radius.csv"
BY_TYPE_CSV = "korea_geo_by_query_type.csv"

H3_RES = 9
MAX_H3_RING = 100

WARMUP = 2
REPEAT = 10

INITIAL_RADIUS_M = 50
MAX_RADIUS_M = 100_000
RADIUS_GROWTH = 2.0

EARTH_RADIUS_M = 6_371_008.8


# ======================================================================================
# H3 COMPAT
# ======================================================================================

def h3_cell(lat, lon, res):

    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(
            lat,
            lon,
            res,
        )

    return h3.geo_to_h3(
        lat,
        lon,
        res,
    )


def h3_exact_ring(cell, ring):

    if ring == 0:
        return [cell]

    if hasattr(h3, "grid_ring"):

        try:
            return list(
                h3.grid_ring(
                    cell,
                    ring,
                )
            )

        except Exception:
            pass

    if hasattr(h3, "grid_disk"):

        outer = set(
            h3.grid_disk(
                cell,
                ring,
            )
        )

        inner = set(
            h3.grid_disk(
                cell,
                ring - 1,
            )
        )

    else:

        outer = set(
            h3.k_ring(
                cell,
                ring,
            )
        )

        inner = set(
            h3.k_ring(
                cell,
                ring - 1,
            )
        )

    return list(
        outer - inner
    )


# ======================================================================================
# DISTANCE
# ======================================================================================

def haversine_vec(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = np.radians(
        float(lat1)
    )

    lon1 = np.radians(
        float(lon1)
    )

    lat2 = np.radians(
        np.asarray(
            lat2,
            dtype=np.float64,
        )
    )

    lon2 = np.radians(
        np.asarray(
            lon2,
            dtype=np.float64,
        )
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            dlon / 2.0
        ) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_M
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ======================================================================================
# LOAD
# ======================================================================================

def load_data():

    if not os.path.exists(
        POI_CSV
    ):
        raise RuntimeError(
            f"Missing {POI_CSV}"
        )

    if not os.path.exists(
        QUERY_CSV
    ):
        raise RuntimeError(
            f"Missing {QUERY_CSV}"
        )

    df = pd.read_csv(
        POI_CSV
    )

    queries = pd.read_csv(
        QUERY_CSV
    )

    df["poi_id"] = (
        df["poi_id"]
        .astype(str)
    )

    print()
    print("=" * 120)
    print("DATA")
    print("=" * 120)

    print(
        "POIs:",
        len(df),
    )

    print(
        "Queries:",
        len(queries),
    )

    print()
    print(
        df[
            "category"
        ]
        .value_counts()
    )

    return (
        df,
        queries,
    )


# ======================================================================================
# H3 LOOKUP
# ======================================================================================

def build_h3_lookup(df):

    print()
    print("=" * 120)
    print("BUILD H3 LOOKUP")
    print("=" * 120)

    lookup = defaultdict(
        list
    )

    cells = []

    for idx, row in df.iterrows():

        cell = h3_cell(
            float(
                row["lat"]
            ),
            float(
                row["lon"]
            ),
            H3_RES,
        )

        cells.append(
            cell
        )

        lookup[
            cell
        ].append(
            idx
        )

    df = df.copy()

    df[
        "h3"
    ] = cells

    print(
        "resolution:",
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
# ES
# ======================================================================================

def connect_es():

    es = Elasticsearch(
        ES_URL
    )

    info = es.info()

    print(
        "[ES]",
        info[
            "version"
        ][
            "number"
        ],
    )

    return es


def create_es_index(
    es,
    df,
):

    print()
    print("=" * 120)
    print("BUILD ES INDEX")
    print("=" * 120)

    if es.indices.exists(
        index=INDEX
    ):

        es.indices.delete(
            index=INDEX
        )

    es.indices.create(

        index=INDEX,

        settings={
            "number_of_shards":
                1,

            "number_of_replicas":
                0,

            "refresh_interval":
                "-1",
        },

        mappings={
            "properties": {

                "poi_id": {
                    "type":
                        "keyword"
                },

                "category": {
                    "type":
                        "keyword"
                },

                "name": {
                    "type":
                        "text"
                },

                "h3": {
                    "type":
                        "keyword"
                },

                "location": {
                    "type":
                        "geo_point"
                },
            }
        },
    )

    def actions():

        for _, row in (
            df.iterrows()
        ):

            yield {

                "_index":
                    INDEX,

                "_id":
                    str(
                        row[
                            "poi_id"
                        ]
                    ),

                "_source": {

                    "poi_id":
                        str(
                            row[
                                "poi_id"
                            ]
                        ),

                    "category":
                        str(
                            row[
                                "category"
                            ]
                        ),

                    "name":
                        str(
                            row.get(
                                "name",
                                "",
                            )
                        ),

                    "h3":
                        str(
                            row[
                                "h3"
                            ]
                        ),

                    "location": {

                        "lat":
                            float(
                                row[
                                    "lat"
                                ]
                            ),

                        "lon":
                            float(
                                row[
                                    "lon"
                                ]
                            ),
                    },
                },
            }

    helpers.bulk(
        es,
        actions(),
        chunk_size=5000,
        request_timeout=180,
    )

    es.indices.refresh(
        index=INDEX
    )

    print(
        "indexed:",
        es.count(
            index=INDEX
        )[
            "count"
        ],
    )


# ======================================================================================
# QUERY HELPERS
# ======================================================================================

def query_category(q):

    return str(
        q[
            "category"
        ]
    )


def query_k(q):

    return int(
        q[
            "k"
        ]
    )


def query_radius(q):

    x = q.get(
        "radius_m"
    )

    if pd.isna(x):
        return None

    return float(x)


# ======================================================================================
# BRUTE-FORCE GT
# ======================================================================================

def brute_force_gt(
    df,
    q,
):

    category = (
        query_category(q)
    )

    k = query_k(q)

    radius = (
        query_radius(q)
    )

    x = df[
        df[
            "category"
        ]
        == category
    ].copy()

    if len(x) == 0:

        return {
            "topk_ids":
                [],

            "all_ids":
                [],

            "total_matches":
                0,
        }

    x[
        "distance_m"
    ] = haversine_vec(

        q[
            "lat"
        ],

        q[
            "lon"
        ],

        x[
            "lat"
        ].values,

        x[
            "lon"
        ].values,
    )

    if radius is not None:

        x = x[
            x[
                "distance_m"
            ]
            <= radius
        ].copy()

    x = x.sort_values(
        [
            "distance_m",
            "poi_id",
        ]
    )

    return {

        "topk_ids":
            x
            .head(k)[
                "poi_id"
            ]
            .astype(str)
            .tolist(),

        "all_ids":
            x[
                "poi_id"
            ]
            .astype(str)
            .tolist(),

        "total_matches":
            len(x),
    }


# ======================================================================================
# METRICS
# ======================================================================================

def top1_accuracy(
    pred,
    gt,
):

    if len(gt) == 0:
        return np.nan

    if len(pred) == 0:
        return 0.0

    return float(
        pred[0]
        == gt[0]
    )


def recall_at_k(
    pred,
    gt,
    k,
):

    gt_k = list(
        gt[:k]
    )

    if len(gt_k) == 0:
        return np.nan

    pred_k = list(
        pred[:k]
    )

    return (
        len(
            set(
                pred_k
            )
            &
            set(
                gt_k
            )
        )
        /
        len(
            gt_k
        )
    )


def candidate_recall_at_k(
    candidate_ids,
    gt,
    k,
):

    if candidate_ids is None:
        return np.nan

    gt_k = set(
        gt[:k]
    )

    if len(gt_k) == 0:
        return np.nan

    return (
        len(
            set(
                candidate_ids
            )
            &
            gt_k
        )
        /
        len(
            gt_k
        )
    )


def radius_recall(
    candidate_ids,
    all_gt,
):

    if candidate_ids is None:
        return np.nan

    gt = set(
        all_gt
    )

    if len(gt) == 0:
        return np.nan

    return (
        len(
            set(
                candidate_ids
            )
            &
            gt
        )
        /
        len(
            gt
        )
    )


# ======================================================================================
# ES NATIVE
# ======================================================================================

def es_native_distance(
    es,
    q,
):

    k = query_k(q)

    radius = (
        query_radius(q)
    )

    filters = [

        {
            "term": {
                "category":
                    query_category(
                        q
                    )
            }
        }
    ]

    if radius is not None:

        filters.append(
            {
                "geo_distance": {

                    "distance":
                        f"{radius}m",

                    "location": {

                        "lat":
                            float(
                                q[
                                    "lat"
                                ]
                            ),

                        "lon":
                            float(
                                q[
                                    "lon"
                                ]
                            ),
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

                        "lat":
                            float(
                                q[
                                    "lat"
                                ]
                            ),

                        "lon":
                            float(
                                q[
                                    "lon"
                                ]
                            ),
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
        x[
            "_source"
        ][
            "poi_id"
        ]
        for x
        in r[
            "hits"
        ][
            "hits"
        ]
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
# ES ADAPTIVE RADIUS
# ======================================================================================

def es_count_radius(
    es,
    q,
    radius,
):

    return es.count(

        index=INDEX,

        query={
            "bool": {

                "filter": [

                    {
                        "term": {
                            "category":
                                query_category(
                                    q
                                )
                        }
                    },

                    {
                        "geo_distance": {

                            "distance":
                                f"{radius}m",

                            "location": {

                                "lat":
                                    float(
                                        q[
                                            "lat"
                                        ]
                                    ),

                                "lon":
                                    float(
                                        q[
                                            "lon"
                                        ]
                                    ),
                            },
                        }
                    },
                ]
            }
        },
    )[
        "count"
    ]


def es_adaptive_radius(
    es,
    q,
):

    k = query_k(q)

    explicit = (
        query_radius(q)
    )

    if explicit is not None:

        count = (
            es_count_radius(
                es,
                q,
                explicit,
            )
        )

        result = (
            es_native_distance(
                es,
                q,
            )
        )

        result[
            "candidate_count"
        ] = count

        result[
            "es_queries"
        ] = 2

        return result

    radius = (
        INITIAL_RADIUS_M
    )

    queries = 0
    count = 0

    while True:

        count = (
            es_count_radius(
                es,
                q,
                radius,
            )
        )

        queries += 1

        if count >= k:
            break

        if radius >= (
            MAX_RADIUS_M
        ):
            break

        radius = min(
            radius
            * RADIUS_GROWTH,
            MAX_RADIUS_M,
        )

    q2 = q.copy()

    q2[
        "radius_m"
    ] = radius

    result = (
        es_native_distance(
            es,
            q2,
        )
    )

    result[
        "candidate_count"
    ] = count

    result[
        "es_queries"
    ] = (
        queries
        + 1
    )

    return result


# ======================================================================================
# H3 CANDIDATES
# ======================================================================================

def get_h3_candidates(
    df,
    lookup,
    q,
):

    k = query_k(q)

    radius = (
        query_radius(q)
    )

    category = (
        query_category(q)
    )

    center = h3_cell(

        float(
            q[
                "lat"
            ]
        ),

        float(
            q[
                "lon"
            ]
        ),

        H3_RES,
    )

    cells = set()

    candidates = (
        df.iloc[
            0:0
        ].copy()
    )

    final_ring = 0

    for ring in range(
        MAX_H3_RING + 1
    ):

        cells.update(
            h3_exact_ring(
                center,
                ring,
            )
        )

        idxs = []

        for cell in cells:

            idxs.extend(
                lookup.get(
                    cell,
                    [],
                )
            )

        if idxs:

            candidates = (
                df.loc[
                    list(
                        set(
                            idxs
                        )
                    )
                ]
            )

            candidates = (
                candidates[
                    candidates[
                        "category"
                    ]
                    == category
                ]
                .copy()
            )

        else:

            candidates = (
                df.iloc[
                    0:0
                ].copy()
            )

        final_ring = ring

        # --------------------------------------------------------------
        # nearest query
        # --------------------------------------------------------------

        if radius is None:

            if len(
                candidates
            ) >= k:

                break

        # --------------------------------------------------------------
        # fixed-radius query
        #
        # We grow beyond requested radius conservatively.
        # Then exact distance filter is applied later.
        # --------------------------------------------------------------

        else:

            if len(
                candidates
            ) == 0:

                continue

            distances = (
                haversine_vec(

                    q[
                        "lat"
                    ],

                    q[
                        "lon"
                    ],

                    candidates[
                        "lat"
                    ].values,

                    candidates[
                        "lon"
                    ].values,
                )
            )

            # intentionally conservative
            #
            # Stop only once searched region contains
            # candidates substantially farther than radius.
            if (
                ring >= 1
                and
                distances.max()
                >= radius * 1.5
            ):

                break

    return (
        candidates,
        final_ring,
    )


# ======================================================================================
# H3 LOCAL RERANK
# ======================================================================================

def h3_incremental(
    df,
    lookup,
    q,
):

    k = query_k(q)

    radius = (
        query_radius(q)
    )

    candidates, ring = (
        get_h3_candidates(
            df,
            lookup,
            q,
        )
    )

    if len(
        candidates
    ) == 0:

        return {

            "ids":
                [],

            "candidate_ids":
                [],

            "candidate_count":
                0,

            "rings":
                ring,

            "es_queries":
                0,
        }

    candidates = (
        candidates.copy()
    )

    candidates[
        "distance_m"
    ] = haversine_vec(

        q[
            "lat"
        ],

        q[
            "lon"
        ],

        candidates[
            "lat"
        ].values,

        candidates[
            "lon"
        ].values,
    )

    if radius is not None:

        candidates = (
            candidates[
                candidates[
                    "distance_m"
                ]
                <= radius
            ]
            .copy()
        )

    candidate_ids = (
        candidates[
            "poi_id"
        ]
        .astype(str)
        .tolist()
    )

    ranked = (
        candidates
        .sort_values(
            [
                "distance_m",
                "poi_id",
            ]
        )
    )

    ids = (
        ranked
        .head(k)[
            "poi_id"
        ]
        .astype(str)
        .tolist()
    )

    return {

        "ids":
            ids,

        "candidate_ids":
            candidate_ids,

        "candidate_count":
            len(
                candidate_ids
            ),

        "rings":
            ring,

        "es_queries":
            0,
    }


# ======================================================================================
# H3 + ES
# ======================================================================================

def h3_es_hybrid(
    es,
    df,
    lookup,
    q,
):

    k = query_k(q)

    radius = (
        query_radius(q)
    )

    candidates, ring = (
        get_h3_candidates(
            df,
            lookup,
            q,
        )
    )

    if len(
        candidates
    ) == 0:

        return {

            "ids":
                [],

            "candidate_ids":
                [],

            "candidate_count":
                0,

            "rings":
                ring,

            "es_queries":
                0,
        }

    candidate_ids = (
        candidates[
            "poi_id"
        ]
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

    if radius is not None:

        filters.append(
            {
                "geo_distance": {

                    "distance":
                        f"{radius}m",

                    "location": {

                        "lat":
                            float(
                                q[
                                    "lat"
                                ]
                            ),

                        "lon":
                            float(
                                q[
                                    "lon"
                                ]
                            ),
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

                        "lat":
                            float(
                                q[
                                    "lat"
                                ]
                            ),

                        "lon":
                            float(
                                q[
                                    "lon"
                                ]
                            ),
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
        x[
            "_source"
        ][
            "poi_id"
        ]
        for x
        in r[
            "hits"
        ][
            "hits"
        ]
    ]

    return {

        "ids":
            ids,

        "candidate_ids":
            candidate_ids,

        "candidate_count":
            len(
                candidate_ids
            ),

        "rings":
            ring,

        "es_queries":
            1,
    }


# ======================================================================================
# BENCHMARK
# ======================================================================================

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

        elapsed = (
            time.perf_counter()
            - t0
        ) * 1000.0

        times.append(
            elapsed
        )

    return {

        "result":
            result,

        "p50_ms":
            float(
                np.percentile(
                    times,
                    50,
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    times,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    times,
                    99,
                )
            ),

        "mean_ms":
            float(
                np.mean(
                    times
                )
            ),
    }


# ======================================================================================
# SINGLE QUERY
# ======================================================================================

def evaluate_query(
    es,
    df,
    lookup,
    q,
):

    gt = brute_force_gt(
        df,
        q,
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
                h3_incremental(
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

    for method, fn in (
        methods.items()
    ):

        b = benchmark_call(
            fn
        )

        r = b[
            "result"
        ]

        gt_ids = (
            gt[
                "topk_ids"
            ]
        )

        row = {

            "qid":
                q[
                    "qid"
                ],

            "query_type":
                q[
                    "query_type"
                ],

            "method":
                method,

            "density":
                q.get(
                    "density",
                    None,
                ),

            "category":
                query_category(
                    q
                ),

            "category_freq":
                q.get(
                    "category_freq",
                    None,
                ),

            "k":
                query_k(
                    q
                ),

            "radius_m":
                query_radius(
                    q
                ),

            "gt_total":
                gt[
                    "total_matches"
                ],

            "p50_ms":
                b[
                    "p50_ms"
                ],

            "p95_ms":
                b[
                    "p95_ms"
                ],

            "p99_ms":
                b[
                    "p99_ms"
                ],

            "mean_ms":
                b[
                    "mean_ms"
                ],

            "top1_acc":
                top1_accuracy(
                    r[
                        "ids"
                    ],
                    gt_ids,
                ),

            "recall@1":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    1,
                ),

            "recall@5":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    5,
                ),

            "recall@20":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    20,
                ),

            "recall@100":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    100,
                ),

            "recall@500":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    500,
                ),

            "recall@query_k":
                recall_at_k(
                    r[
                        "ids"
                    ],
                    gt_ids,
                    query_k(
                        q
                    ),
                ),

            "candidate_recall@5":
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt_ids,
                    5,
                ),

            "candidate_recall@20":
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt_ids,
                    20,
                ),

            "candidate_recall@100":
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt_ids,
                    100,
                ),

            "candidate_recall@500":
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt_ids,
                    500,
                ),

            "candidate_recall@query_k":
                candidate_recall_at_k(
                    r[
                        "candidate_ids"
                    ],
                    gt_ids,
                    query_k(
                        q
                    ),
                ),

            "radius_recall":
                radius_recall(
                    r[
                        "candidate_ids"
                    ],
                    gt[
                        "all_ids"
                    ],
                )
                if query_radius(q)
                is not None
                else np.nan,

            "candidate_count":
                r[
                    "candidate_count"
                ],

            "rings":
                r[
                    "rings"
                ],

            "es_queries":
                r[
                    "es_queries"
                ],
        }

        rows.append(
            row
        )

    return rows


# ======================================================================================
# SUMMARY HELPERS
# ======================================================================================

def aggregate(
    df,
    group_cols,
):

    return (

        df
        .groupby(
            group_cols,
            dropna=False,
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

            p95_ms=(
                "p95_ms",
                "median",
            ),

            p99_ms=(
                "p99_ms",
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

            recall100=(
                "recall@100",
                "mean",
            ),

            recall500=(
                "recall@500",
                "mean",
            ),

            recall_query_k=(
                "recall@query_k",
                "mean",
            ),

            candidate_recall=(
                "candidate_recall@query_k",
                "mean",
            ),

            median_candidates=(
                "candidate_count",
                "median",
            ),

            median_rings=(
                "rings",
                "median",
            ),

            mean_es_queries=(
                "es_queries",
                "mean",
            ),

        )

        .reset_index()
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main():

    np.random.seed(
        42
    )

    df, queries = (
        load_data()
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

    rows = []

    total = len(
        queries
    )

    print()
    print("=" * 120)
    print("RUN BENCHMARK")
    print("=" * 120)

    for i, (_, q) in enumerate(
        queries.iterrows(),
        start=1,
    ):

        print(
            f"[{i:05d}/{total:05d}]",
            q[
                "qid"
            ],
            f"type={q['query_type']}",
            f"density={q.get('density')}",
            f"category={q['category']}",
            f"k={int(q['k'])}",
            f"radius={q.get('radius_m')}",
        )

        query_rows = (
            evaluate_query(
                es,
                df,
                lookup,
                q,
            )
        )

        rows.extend(
            query_rows
        )

        # checkpoint
        if i % 50 == 0:

            pd.DataFrame(
                rows
            ).to_csv(
                RESULT_CSV,
                index=False,
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        RESULT_CSV,
        index=False,
    )

    # Remove zero-GT queries from quality summary.
    valid = result[
        result[
            "gt_total"
        ]
        > 0
    ].copy()

    # ------------------------------------------------------------------
    # OVERALL
    # ------------------------------------------------------------------

    summary = aggregate(
        valid,
        [
            "method",
        ],
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # BY K
    # ------------------------------------------------------------------

    by_k = aggregate(
        valid,
        [
            "k",
            "method",
        ],
    )

    by_k.to_csv(
        BY_K_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # BY DENSITY
    # ------------------------------------------------------------------

    by_density = aggregate(
        valid,
        [
            "density",
            "method",
        ],
    )

    by_density.to_csv(
        BY_DENSITY_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # BY CATEGORY FREQUENCY
    # ------------------------------------------------------------------

    by_category = aggregate(
        valid,
        [
            "category_freq",
            "method",
        ],
    )

    by_category.to_csv(
        BY_CATEGORY_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # BY QUERY TYPE
    # ------------------------------------------------------------------

    by_type = aggregate(
        valid,
        [
            "query_type",
            "method",
        ],
    )

    by_type.to_csv(
        BY_TYPE_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # BY RADIUS
    # ------------------------------------------------------------------

    radius_df = valid[
        valid[
            "query_type"
        ]
        == "radius"
    ]

    if len(
        radius_df
    ):

        by_radius = aggregate(
            radius_df,
            [
                "radius_m",
                "method",
            ],
        )

        by_radius.to_csv(
            BY_RADIUS_CSV,
            index=False,
        )

    # ------------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------------

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        400,
    )

    print()
    print("=" * 160)
    print("OVERALL")
    print("=" * 160)

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 160)
    print("BY K")
    print("=" * 160)

    print(
        by_k
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 160)
    print("BY DENSITY")
    print("=" * 160)

    print(
        by_density
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 160)
    print("BY CATEGORY FREQUENCY")
    print("=" * 160)

    print(
        by_category
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 160)
    print("BY QUERY TYPE")
    print("=" * 160)

    print(
        by_type
        .round(4)
        .to_string(
            index=False
        )
    )

    if len(
        radius_df
    ):

        print()
        print("=" * 160)
        print("BY RADIUS")
        print("=" * 160)

        print(
            by_radius
            .round(4)
            .to_string(
                index=False
            )
        )

    print()
    print("saved:")
    print(" ", RESULT_CSV)
    print(" ", SUMMARY_CSV)
    print(" ", BY_K_CSV)
    print(" ", BY_DENSITY_CSV)
    print(" ", BY_CATEGORY_CSV)
    print(" ", BY_TYPE_CSV)

    if len(
        radius_df
    ):
        print(
            " ",
            BY_RADIUS_CSV,
        )


if __name__ == "__main__":
    main()
