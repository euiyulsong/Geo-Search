#!/usr/bin/env python3

import os
import math
import random
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

try:
    import osmium
except ImportError:
    raise RuntimeError("pip install osmium")

try:
    import h3
except ImportError:
    raise RuntimeError("pip install h3")


# =============================================================================
# CONFIG
# =============================================================================

PBF_URL = (
    "https://download.geofabrik.de/asia/"
    "south-korea-latest.osm.pbf"
)

PBF_PATH = "south-korea-latest.osm.pbf"
POI_CSV = "korea_osm_pois.csv"
QUERY_CSV = "korea_geo_queries.csv"

H3_RES = 9
DENSITY_RES = 8

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# CATEGORY MAPPING
# =============================================================================

def normalize_category(tags):

    amenity = tags.get("amenity")
    shop = tags.get("shop")
    tourism = tags.get("tourism")
    leisure = tags.get("leisure")

    # FOOD
    if amenity in {
        "restaurant",
        "fast_food",
        "food_court",
    }:
        return "restaurant"

    if amenity == "cafe":
        return "cafe"

    if amenity in {
        "bar",
        "pub",
        "biergarten",
    }:
        return "bar"

    # HEALTH
    if amenity in {
        "hospital",
        "clinic",
        "doctors",
    }:
        return "healthcare"

    if amenity == "pharmacy":
        return "pharmacy"

    # EDUCATION
    if amenity in {
        "school",
        "university",
        "college",
        "kindergarten",
    }:
        return "education"

    # TRANSPORT
    if amenity in {
        "bus_station",
        "taxi",
        "ferry_terminal",
    }:
        return "transport"

    if amenity in {
        "fuel",
        "charging_station",
    }:
        return "fuel"

    if amenity == "parking":
        return "parking"

    # RETAIL
    if shop in {
        "convenience",
        "supermarket",
    }:
        return "grocery"

    if shop in {
        "clothes",
        "shoes",
        "fashion",
    }:
        return "fashion"

    if shop in {
        "electronics",
        "computer",
        "mobile_phone",
    }:
        return "electronics"

    if shop in {
        "bakery",
        "butcher",
        "greengrocer",
    }:
        return "food_shop"

    if shop is not None:
        return "other_shop"

    # TOURISM
    if tourism in {
        "hotel",
        "motel",
        "hostel",
        "guest_house",
    }:
        return "accommodation"

    if tourism in {
        "attraction",
        "museum",
        "gallery",
        "viewpoint",
    }:
        return "tourism"

    # LEISURE
    if leisure in {
        "park",
        "sports_centre",
        "fitness_centre",
        "stadium",
    }:
        return "leisure"

    return None


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_file():

    if os.path.exists(PBF_PATH):
        print("[exists]", PBF_PATH)
        return

    print("[download]", PBF_URL)

    with requests.get(
        PBF_URL,
        stream=True,
        timeout=600,
    ) as r:

        r.raise_for_status()

        total = int(
            r.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0

        with open(PBF_PATH, "wb") as f:

            for chunk in r.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                print(
                    f"\r{downloaded / 1e6:.1f} MB"
                    f" / {total / 1e6:.1f} MB",
                    end="",
                )

    print()


# =============================================================================
# OSM EXTRACTION
# =============================================================================

class POIHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()

        self.rows = []

    def add(
        self,
        osm_type,
        osm_id,
        tags,
        lat,
        lon,
    ):

        cat = normalize_category(tags)

        if cat is None:
            return

        if not (
            33.0 <= lat <= 39.5
            and
            124.0 <= lon <= 132.0
        ):
            return

        self.rows.append(
            {
                "poi_id":
                    f"{osm_type}_{osm_id}",

                "name":
                    tags.get(
                        "name",
                        "",
                    ),

                "category":
                    cat,

                "lat":
                    lat,

                "lon":
                    lon,
            }
        )

    def node(self, n):

        if not n.location.valid():
            return

        tags = dict(n.tags)

        self.add(
            "node",
            n.id,
            tags,
            n.location.lat,
            n.location.lon,
        )

    def way(self, w):

        tags = dict(w.tags)

        if normalize_category(tags) is None:
            return

        coords = []

        for node in w.nodes:

            try:

                if node.location.valid():

                    coords.append(
                        (
                            node.location.lat,
                            node.location.lon,
                        )
                    )

            except Exception:
                pass

        if not coords:
            return

        lat = float(
            np.mean(
                [x[0] for x in coords]
            )
        )

        lon = float(
            np.mean(
                [x[1] for x in coords]
            )
        )

        self.add(
            "way",
            w.id,
            tags,
            lat,
            lon,
        )


def extract_pois():

    if os.path.exists(POI_CSV):

        print("[load]", POI_CSV)

        return pd.read_csv(
            POI_CSV
        )

    handler = POIHandler()

    print("[parse PBF]")

    handler.apply_file(
        PBF_PATH,
        locations=True,
    )

    df = pd.DataFrame(
        handler.rows
    )

    df = (
        df
        .drop_duplicates(
            "poi_id"
        )
        .dropna(
            subset=[
                "lat",
                "lon",
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "POIs:",
        len(df),
    )

    print()
    print(
        df.category
        .value_counts()
        .to_string()
    )

    df.to_csv(
        POI_CSV,
        index=False,
    )

    return df


# =============================================================================
# H3
# =============================================================================

def h3_cell(lat, lon, res):

    if hasattr(
        h3,
        "latlng_to_cell",
    ):

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


# =============================================================================
# DENSITY MAP
# =============================================================================

def add_density(df):

    print()
    print("=" * 100)
    print("BUILD DENSITY MAP")
    print("=" * 100)

    df = df.copy()

    df["density_cell"] = [
        h3_cell(
            lat,
            lon,
            DENSITY_RES,
        )
        for lat, lon
        in zip(
            df.lat,
            df.lon,
        )
    ]

    counts = (
        df
        .groupby(
            "density_cell"
        )
        .size()
    )

    df["local_density"] = (
        df[
            "density_cell"
        ]
        .map(counts)
    )

    print(
        df.local_density
        .describe(
            percentiles=[
                .1,
                .25,
                .5,
                .75,
                .9,
                .95,
                .99,
            ]
        )
    )

    q33 = df.local_density.quantile(
        0.33
    )

    q67 = df.local_density.quantile(
        0.67
    )

    def label(x):

        if x <= q33:
            return "sparse"

        if x <= q67:
            return "medium"

        return "dense"

    df["density_bucket"] = (
        df.local_density
        .map(label)
    )

    return df


# =============================================================================
# CATEGORY DIFFICULTY
# =============================================================================

def category_frequency_buckets(df):

    counts = (
        df.category
        .value_counts()
    )

    cats = counts.index.tolist()

    n = len(cats)

    common = set(
        cats[
            :max(
                1,
                n // 3,
            )
        ]
    )

    rare = set(
        cats[
            max(
                1,
                2 * n // 3,
            ):
        ]
    )

    result = {}

    for c in cats:

        if c in common:

            result[c] = "common"

        elif c in rare:

            result[c] = "rare"

        else:

            result[c] = "medium"

    return result


# =============================================================================
# QUERY GENERATION
# =============================================================================

def generate_queries(df):

    print()
    print("=" * 100)
    print("GENERATE HARD QUERIES")
    print("=" * 100)

    category_bucket = (
        category_frequency_buckets(
            df
        )
    )

    rows = []

    qid = 0

    # -----------------------------------------------------------------
    # NEAREST TOP-K workloads
    # -----------------------------------------------------------------

    K_VALUES = [
        1,
        5,
        20,
        100,
        500,
    ]

    for density in [
        "dense",
        "medium",
        "sparse",
    ]:

        subset = df[
            df.density_bucket
            == density
        ]

        if len(subset) == 0:
            continue

        # 여러 지역을 자동 sampling
        anchors = subset.sample(
            n=min(
                20,
                len(subset),
            ),
            random_state=SEED,
        )

        for _, anchor in anchors.iterrows():

            category = anchor.category

            # 원래 POI 위치에서 살짝 jitter
            #
            # query가 POI와 정확히 일치하는 쉬운
            # benchmark를 방지한다.
            for _ in range(2):

                lat_jitter = np.random.normal(
                    0,
                    0.0015,
                )

                lon_jitter = np.random.normal(
                    0,
                    0.0015,
                )

                qlat = (
                    anchor.lat
                    + lat_jitter
                )

                qlon = (
                    anchor.lon
                    + lon_jitter
                )

                for k in K_VALUES:

                    qid += 1

                    rows.append(
                        {
                            "qid":
                                f"nearest_{qid}",

                            "query_type":
                                "nearest",

                            "lat":
                                qlat,

                            "lon":
                                qlon,

                            "category":
                                category,

                            "category_freq":
                                category_bucket[
                                    category
                                ],

                            "density":
                                density,

                            "k":
                                k,

                            "radius_m":
                                np.nan,
                        }
                    )

    # -----------------------------------------------------------------
    # Fixed radius workloads
    # -----------------------------------------------------------------

    RADIUS_VALUES = [
        100,
        300,
        1000,
        3000,
        10000,
    ]

    sample = df.sample(
        n=min(
            100,
            len(df),
        ),
        random_state=123,
    )

    for _, anchor in sample.iterrows():

        for radius in RADIUS_VALUES:

            qid += 1

            rows.append(
                {
                    "qid":
                        f"radius_{qid}",

                    "query_type":
                        "radius",

                    "lat":
                        anchor.lat
                        + np.random.normal(
                            0,
                            0.002,
                        ),

                    "lon":
                        anchor.lon
                        + np.random.normal(
                            0,
                            0.002,
                        ),

                    "category":
                        anchor.category,

                    "category_freq":
                        category_bucket[
                            anchor.category
                        ],

                    "density":
                        anchor.density_bucket,

                    "k":
                        500,

                    "radius_m":
                        radius,
                }
            )

    queries = pd.DataFrame(
        rows
    )

    queries.to_csv(
        QUERY_CSV,
        index=False,
    )

    print(
        "queries:",
        len(queries),
    )

    print()

    print(
        queries.groupby(
            [
                "query_type",
                "density",
            ]
        )
        .size()
    )

    print()

    print(
        queries.groupby(
            [
                "category_freq",
                "k",
            ]
        )
        .size()
    )

    return queries


# =============================================================================
# MAIN
# =============================================================================

def main():

    download_file()

    df = extract_pois()

    df = add_density(
        df
    )

    queries = generate_queries(
        df
    )

    print()
    print("=" * 100)
    print("DATASET READY")
    print("=" * 100)

    print(
        "POIs:",
        len(df),
    )

    print(
        "Queries:",
        len(queries),
    )

    print(
        "\nPOI category distribution:"
    )

    print(
        df.category
        .value_counts()
    )

    print(
        "\nDensity distribution:"
    )

    print(
        df.density_bucket
        .value_counts()
    )

    print()
    print(
        "saved:",
        POI_CSV,
        QUERY_CSV,
    )


if __name__ == "__main__":
    main()
