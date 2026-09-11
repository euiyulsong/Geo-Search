# fair_weighted_vs_lambda.py

#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import ndcg_score

# ============================================================
# REUSE EXACT WORKING POINTREC PIPELINE
# ============================================================

from local_search_weighted import (
    load_information_needs,
    load_qrels,
    load_pois,
    check_join_coverage,
    build_dataset,
    split_queries,
    evaluate,
    bootstrap_ndcg,
    apply_weighted_score,
    generate_weights,
    WEIGHT_FEATURES,
)


# ============================================================
# RANDOM NEGATIVES FROM THE FULL POI CORPUS
# ============================================================

def add_random_corpus_negatives(qrels, pois, negatives_per_query=20, seed=42):
    """
    Add unjudged POIs sampled uniformly from the *entire POI corpus* as
    relevance-0 training candidates for every query.

    Important:
      - sampling pool = all poi_id values in `pois`
      - already judged (qid, poi_id) pairs are excluded
      - sampling is without replacement within each query
      - sampled rows receive relevance 0
      - deterministic for a fixed seed

    The returned frame has the same columns as qrels, so the existing
    build_dataset(qrels, queries, pois) pipeline can compute exactly the
    same features for positives/judged docs and random negatives.
    """
    if negatives_per_query <= 0:
        return qrels.copy()

    required = {"qid", "poi_id"}
    missing_qrels = required - set(qrels.columns)
    if missing_qrels:
        raise ValueError(f"qrels missing required columns: {sorted(missing_qrels)}")
    if "poi_id" not in pois.columns:
        raise ValueError("pois must contain poi_id")

    # Find the relevance column used by the source qrels.
    rel_candidates = [
        "human_rel", "relevance", "rel", "label",
        "rating", "relevance_score"
    ]
    rel_col = next((c for c in rel_candidates if c in qrels.columns), None)
    if rel_col is None:
        # Fallback: qrels usually has exactly one payload column besides IDs.
        payload = [c for c in qrels.columns if c not in {"qid", "poi_id"}]
        if len(payload) == 1:
            rel_col = payload[0]
        else:
            raise ValueError(
                "Could not identify qrels relevance column. "
                f"Columns={list(qrels.columns)}"
            )

    rng = np.random.default_rng(seed)
    corpus_ids = pd.Index(pois["poi_id"].dropna().unique())
    rows = []

    for qid, judged in qrels.groupby("qid", sort=True):
        judged_ids = set(judged["poi_id"].dropna().tolist())
        candidates = corpus_ids[~corpus_ids.isin(judged_ids)].to_numpy()

        n = min(int(negatives_per_query), len(candidates))
        if n == 0:
            continue

        sampled = rng.choice(candidates, size=n, replace=False)
        for poi_id in sampled:
            row = {c: np.nan for c in qrels.columns}
            row["qid"] = qid
            row["poi_id"] = poi_id
            row[rel_col] = 0
            rows.append(row)

    if not rows:
        return qrels.copy()

    negatives = pd.DataFrame(rows, columns=qrels.columns)
    out = pd.concat([qrels, negatives], ignore_index=True)

    # Hard safety checks: no duplicate pair and no sampled judged pair.
    if out.duplicated(["qid", "poi_id"]).any():
        dup = out[out.duplicated(["qid", "poi_id"], keep=False)]
        raise AssertionError(
            "Duplicate (qid, poi_id) after random-negative sampling:\n"
            + dup[["qid", "poi_id"]].head(20).to_string(index=False)
        )

    print("\nRandom corpus negatives:")
    print(f"  source corpus POIs : {len(corpus_ids):,}")
    print(f"  per query requested: {negatives_per_query}")
    print(f"  negatives added    : {len(negatives):,}")
    print(f"  qrels before/after : {len(qrels):,} -> {len(out):,}")
    print(f"  relevance column   : {rel_col}")

    return out


# ============================================================
# LABELS
# ============================================================

def make_labels(df, mode, seed=42):
    """
    EXACT SAME labels will be consumed by BOTH rankers.

    full_graded:
        original human_rel 0/1/2/3

    top_grade:
        every query-level maximum-relevance document = 1
        all others = 0

    strict_top1:
        randomly select EXACTLY ONE document among
        query-level maximum-relevance documents.
        No rating/review_count/feature leakage.
    """

    out = df.copy()

    if mode == "full_graded":

        out["train_label"] = (
            out["human_rel"]
            .astype(int)
        )

    elif mode == "top_grade":

        max_rel = (
            out.groupby("qid")["human_rel"]
            .transform("max")
        )

        out["train_label"] = (
            out["human_rel"] == max_rel
        ).astype(int)

    elif mode == "strict_top1":

        out["train_label"] = 0

        rng = np.random.default_rng(seed)

        for qid in sorted(out["qid"].unique()):

            g = out[
                out["qid"] == qid
            ]

            max_rel = g["human_rel"].max()

            candidates = (
                g[
                    g["human_rel"] == max_rel
                ]
                .index
                .to_numpy()
            )

            chosen = rng.choice(candidates)

            out.loc[
                chosen,
                "train_label"
            ] = 1

    else:
        raise ValueError(mode)

    return out


# ============================================================
# NDCG AGAINST ARBITRARY LABEL
# ============================================================

def mean_ndcg_label(
    df,
    score_col,
    label_col,
    k=5
):
    """
    Same NDCG definition, but allows training/validation
    against train_label rather than human_rel.
    """

    values = []

    for _, group in df.groupby("qid"):

        y_true = np.maximum(
            group[label_col]
            .to_numpy(dtype=float),
            0
        )

        y_score = (
            group[score_col]
            .to_numpy(dtype=float)
        )

        if np.max(y_true) <= 0:
            continue

        value = ndcg_score(
            y_true.reshape(1, -1),
            y_score.reshape(1, -1),
            k=min(k, len(group))
        )

        values.append(float(value))

    if not values:
        return np.nan

    return float(np.mean(values))


# ============================================================
# PAIR AGREEMENT AGAINST ARBITRARY LABEL
# ============================================================

def pair_agree_label(
    df,
    score_col,
    label_col
):

    correct = 0.0
    total = 0

    for _, group in df.groupby("qid"):

        labels = (
            group[label_col]
            .to_numpy()
        )

        pred = (
            group[score_col]
            .to_numpy()
        )

        n = len(group)

        for i in range(n):

            for j in range(i + 1, n):

                if labels[i] == labels[j]:
                    continue

                total += 1

                label_order = (
                    labels[i] > labels[j]
                )

                if pred[i] == pred[j]:

                    correct += 0.5
                    continue

                model_order = (
                    pred[i] > pred[j]
                )

                if model_order == label_order:
                    correct += 1.0

    if total == 0:
        return np.nan

    return correct / total


# ============================================================
# WEIGHTED SUM — TRAIN
# ============================================================

def search_weighted_train(
    train,
    step=0.05,
    top_k=100
):

    rows = []

    count = 0

    for weights in generate_weights(step):

        count += 1

        tmp = apply_weighted_score(
            train,
            weights,
            "_score"
        )

        ndcg5 = mean_ndcg_label(
            tmp,
            "_score",
            "train_label",
            5
        )

        row = {
            "train_ndcg5": ndcg5
        }

        for feature, weight in zip(
            WEIGHT_FEATURES,
            weights
        ):
            row[
                f"w_{feature}"
            ] = float(weight)

        rows.append(row)

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "train_ndcg5",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        f"Weighted combinations: {count:,}"
    )

    print(
        f"Best train label-NDCG@5: "
        f"{result.iloc[0]['train_ndcg5']:.4f}"
    )

    return (
        result.head(top_k).copy(),
        result
    )


# ============================================================
# WEIGHTED SUM — VALIDATION
# ============================================================

def select_weighted_valid(
    candidates,
    valid
):

    rows = []

    for _, candidate in candidates.iterrows():

        weights = np.array([
            candidate[f"w_{feature}"]
            for feature in WEIGHT_FEATURES
        ])

        tmp = apply_weighted_score(
            valid,
            weights,
            "_score"
        )

        ndcg5 = mean_ndcg_label(
            tmp,
            "_score",
            "train_label",
            5
        )

        ndcg10 = mean_ndcg_label(
            tmp,
            "_score",
            "train_label",
            10
        )

        pair = pair_agree_label(
            tmp,
            "_score",
            "train_label"
        )

        row = {
            "train_ndcg5":
                candidate["train_ndcg5"],

            "valid_label_ndcg5":
                ndcg5,

            "valid_label_ndcg10":
                ndcg10,

            "valid_label_pairagree":
                pair,
        }

        for feature, weight in zip(
            WEIGHT_FEATURES,
            weights
        ):
            row[
                f"w_{feature}"
            ] = float(weight)

        rows.append(row)

    result = pd.DataFrame(rows)

    # EXACT SAME SELECTION POLICY:
    #
    # NDCG@5
    # -> PairAgree
    # -> NDCG@10
    #
    # But calculated against SAME supervision label.

    result = (
        result
        .sort_values(
            [
                "valid_label_ndcg5",
                "valid_label_pairagree",
                "valid_label_ndcg10",
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    best = result.iloc[0]

    weights = np.array([
        best[f"w_{feature}"]
        for feature in WEIGHT_FEATURES
    ])

    return weights, result


# ============================================================
# TRAIN WEIGHTED
# ============================================================

def train_weighted(
    train,
    valid,
    step,
    top_k
):

    candidates, train_search = (
        search_weighted_train(
            train,
            step=step,
            top_k=top_k
        )
    )

    weights, valid_search = (
        select_weighted_valid(
            candidates,
            valid
        )
    )

    print("\nSelected Weighted Sum weights:")

    for f, w in zip(
        WEIGHT_FEATURES,
        weights
    ):
        print(
            f"  {f:20s}: {w:.2f}"
        )

    return (
        weights,
        train_search,
        valid_search
    )


# ============================================================
# LAMBDARANK DATA
# ============================================================

def prepare_rank_data(df):

    x = (
        df
        .sort_values(
            ["qid", "poi_id"]
        )
        .reset_index(drop=True)
        .copy()
    )

    X = (
        x[WEIGHT_FEATURES]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    y = (
        x["train_label"]
        .to_numpy(dtype=int)
    )

    groups = (
        x.groupby(
            "qid",
            sort=False
        )
        .size()
        .tolist()
    )

    assert sum(groups) == len(x)

    return x, X, y, groups


# ============================================================
# TRAIN LAMBDARANK
# ============================================================

def train_lambda(
    train,
    valid,
    seed
):

    (
        train_sorted,
        X_train,
        y_train,
        train_groups
    ) = prepare_rank_data(train)

    (
        valid_sorted,
        X_valid,
        y_valid,
        valid_groups
    ) = prepare_rank_data(valid)

    model = lgb.LGBMRanker(

        objective="lambdarank",
        metric="ndcg",

        learning_rate=0.03,
        n_estimators=1000,

        num_leaves=15,
        max_depth=5,

        min_child_samples=10,

        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,

        group=train_groups,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        eval_group=[
            valid_groups
        ],

        eval_at=[5, 10],

        callbacks=[
            lgb.early_stopping(
                100,
                verbose=False
            )
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_weighted(
    test,
    weights,
    bootstrap
):

    pred = apply_weighted_score(
        test,
        weights,
        "score"
    )

    # ORIGINAL human_rel evaluation
    metrics = evaluate(
        pred,
        "score"
    )

    low, high = bootstrap_ndcg(
        pred,
        "score",
        bootstrap
    )

    metrics["NDCG5_CI_low"] = low
    metrics["NDCG5_CI_high"] = high

    return metrics, pred


def evaluate_lambda(
    test,
    model,
    bootstrap
):

    pred = test.copy()

    X = (
        pred[WEIGHT_FEATURES]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    pred["score"] = model.predict(
        X,
        num_iteration=model.best_iteration_
    )

    # ORIGINAL human_rel evaluation
    metrics = evaluate(
        pred,
        "score"
    )

    low, high = bootstrap_ndcg(
        pred,
        "score",
        bootstrap
    )

    metrics["NDCG5_CI_low"] = low
    metrics["NDCG5_CI_high"] = high

    return metrics, pred


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def lambda_importance(model):

    result = pd.DataFrame({

        "feature":
            WEIGHT_FEATURES,

        "gain":
            model.booster_
            .feature_importance(
                importance_type="gain"
            ),

        "split":
            model.booster_
            .feature_importance(
                importance_type="split"
            ),
    })

    gain_sum = result["gain"].sum()

    if gain_sum > 0:

        result["gain_pct"] = (
            result["gain"]
            / gain_sum
            * 100
        )

    else:

        result["gain_pct"] = 0.0

    return (
        result
        .sort_values(
            "gain",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# SHARED LABEL CHECK
# ============================================================

def verify_shared_labels(
    train,
    valid,
    test,
    mode
):

    print("\nLabel sanity:")

    for name, x in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:

        counts = (
            x["train_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"{name:6s}: "
            f"{x.qid.nunique():3d} queries / "
            f"{len(x):4d} pairs / "
            f"{counts}"
        )

        if mode == "strict_top1":

            positives = (
                x.groupby("qid")
                ["train_label"]
                .sum()
            )

            assert (
                positives == 1
            ).all(), (
                "strict_top1 must have "
                "exactly one positive/query"
            )


# ============================================================
# FEATURE SANITY
# ============================================================

def feature_sanity(df):

    print(
        "\n"
        + "=" * 100
    )

    print("FEATURE SANITY")

    print("=" * 100)

    print(
        df[WEIGHT_FEATURES]
        .describe()
        .T
        .to_string()
    )

    print("\nNONZERO")

    for f in WEIGHT_FEATURES:

        n = (
            df[f]
            .fillna(0)
            .ne(0)
            .sum()
        )

        print(
            f"{f:20s}: "
            f"{n:,}/{len(df):,}"
        )

    assert (
        df["category_score"]
        .fillna(0)
        .abs()
        .sum()
        > 0
    ), "category_score is dead"

    assert (
        df["bm25"]
        .fillna(0)
        .abs()
        .sum()
        > 0
    ), "bm25 is dead"


# ============================================================
# BASELINES
# ============================================================

def baselines(test):

    rows = []

    for name, feature in [
        (
            "category_only",
            "category_score"
        ),
        (
            "bm25_only",
            "bm25"
        ),
    ]:

        x = test.copy()

        x["score"] = x[feature]

        m = evaluate(
            x,
            "score"
        )

        rows.append({
            "ranker": name,
            "supervision": "none",
            **m
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="pointrec"
    )

    parser.add_argument(
        "--train_queries",
        type=int,
        default=60
    )

    parser.add_argument(
        "--valid_queries",
        type=int,
        default=20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--top_candidates",
        type=int,
        default=100
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--random_negatives_per_query",
        type=int,
        default=20,
        help=(
            "Uniform random negatives sampled per query from the entire "
            "POI corpus. Use 0 to reproduce the old judged-pairs-only behavior."
        )
    )

    args = parser.parse_args()

    print(
        "=" * 120
    )

    print(
        "POINTREC — TRUE APPLES-TO-APPLES"
    )

    print(
        "Weighted Sum vs LambdaRank"
    )

    print(
        "=" * 120
    )

    # ========================================================
    # EXACT SAME DATA
    # ========================================================

    queries = load_information_needs(
        args.root
    )

    qrels = load_qrels(
        args.root
    )

    pois = load_pois(
        args.root
    )

    # IMPORTANT: the original script built the dataset directly from qrels,
    # so it used judged pairs only. Here we explicitly augment qrels with
    # uniformly sampled negatives from the FULL POI corpus first.
    qrels_original = qrels.copy()
    qrels = add_random_corpus_negatives(
        qrels,
        pois,
        negatives_per_query=args.random_negatives_per_query,
        seed=args.seed
    )

    check_join_coverage(
        qrels,
        pois
    )

    df = build_dataset(
        qrels,
        queries,
        pois
    )

    print(
        "\nDataset:"
    )

    print(
        f"{df.qid.nunique()} queries / "
        f"{len(df)} judged pairs"
    )

    print(
        "\nHuman relevance:"
    )

    print(
        df["human_rel"]
        .value_counts()
        .sort_index()
    )

    feature_sanity(df)

    # ========================================================
    # ONE SHARED SPLIT
    # ========================================================

    train_base, valid_base, test_base = (
        split_queries(
            df,
            n_train=args.train_queries,
            n_valid=args.valid_queries
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print("SHARED SPLIT")

    print("=" * 100)

    print(
        "train:",
        train_base.qid.nunique(),
        "queries /",
        len(train_base),
        "pairs"
    )

    print(
        "valid:",
        valid_base.qid.nunique(),
        "queries /",
        len(valid_base),
        "pairs"
    )

    print(
        "test :",
        test_base.qid.nunique(),
        "queries /",
        len(test_base),
        "pairs"
    )

    # Old judged-pairs-only split sizes were (2751, 914, 1443).
    # They are intentionally no longer expected once random negatives are added.
    expected = (2751, 914, 1443) if args.random_negatives_per_query == 0 else None

    actual = (
        len(train_base),
        len(valid_base),
        len(test_base)
    )

    if expected is None:
        print(
            "\n[OK] Random negatives enabled; split pair counts are expected "
            "to be larger than the old judged-only split."
        )
    elif actual == expected:
        print(
            "\n[OK] Previous working split reproduced."
        )
    else:
        print(
            "\n[WARNING] Previous split was "
            f"{expected}; current={actual}"
        )

    # ========================================================
    # SAVE SHARED DATA / SPLIT
    # ========================================================

    df.to_csv(
        "fair_shared_features.csv",
        index=False
    )

    split_rows = []

    for split_name, part in [
        ("train", train_base),
        ("valid", valid_base),
        ("test", test_base),
    ]:

        for qid in sorted(
            part["qid"].unique()
        ):

            split_rows.append({
                "qid": qid,
                "split": split_name
            })

    pd.DataFrame(
        split_rows
    ).to_csv(
        "fair_shared_split.csv",
        index=False
    )

    # ========================================================
    # BASELINES
    # ========================================================

    baseline = baselines(
        test_base
    )

    print(
        "\n"
        + "=" * 100
    )

    print("BASELINES")

    print("=" * 100)

    print(
        baseline.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # 2 x 3 EXPERIMENT
    # ========================================================

    modes = [
        "full_graded",
        "top_grade",
        "strict_top1",
    ]

    result_rows = []

    for mode in modes:

        print(
            "\n\n"
            + "=" * 120
        )

        print(
            f"SUPERVISION = {mode}"
        )

        print(
            "=" * 120
        )

        # ====================================================
        # CREATE LABEL EXACTLY ONCE
        # ====================================================

        labeled = make_labels(
            df,
            mode,
            seed=args.seed
        )

        train = (
            labeled[
                labeled["qid"].isin(
                    train_base["qid"].unique()
                )
            ]
            .copy()
        )

        valid = (
            labeled[
                labeled["qid"].isin(
                    valid_base["qid"].unique()
                )
            ]
            .copy()
        )

        test = (
            labeled[
                labeled["qid"].isin(
                    test_base["qid"].unique()
                )
            ]
            .copy()
        )

        verify_shared_labels(
            train,
            valid,
            test,
            mode
        )

        # Save labels to prove both rankers
        # consumed exactly the same supervision.

        labeled[
            [
                "qid",
                "poi_id",
                "human_rel",
                "train_label",
            ]
        ].to_csv(
            f"fair_labels_{mode}.csv",
            index=False
        )

        # ====================================================
        # A. WEIGHTED SUM
        # ====================================================

        print(
            "\n"
            + "-" * 100
        )

        print(
            "A. WEIGHTED SUM"
        )

        print(
            "-" * 100
        )

        (
            weights,
            train_search,
            valid_search
        ) = train_weighted(
            train,
            valid,
            args.step,
            args.top_candidates
        )

        weighted_metrics, weighted_pred = (
            evaluate_weighted(
                test,
                weights,
                args.bootstrap
            )
        )

        print(
            "\nWeighted TEST "
            "(original human_rel 0-3)"
        )

        for k, v in (
            weighted_metrics.items()
        ):

            print(
                f"{k:15s}: {v:.4f}"
            )

        train_search.to_csv(
            f"fair_weighted_{mode}_train_search.csv",
            index=False
        )

        valid_search.to_csv(
            f"fair_weighted_{mode}_valid_search.csv",
            index=False
        )

        weighted_pred.to_csv(
            f"fair_weighted_{mode}_predictions.csv",
            index=False
        )

        row = {
            "ranker":
                "WeightedSum",

            "supervision":
                mode,

            **weighted_metrics,

            "best_iteration":
                np.nan,
        }

        for f, w in zip(
            WEIGHT_FEATURES,
            weights
        ):

            row[
                f"w_{f}"
            ] = w

        result_rows.append(row)

        # ====================================================
        # B. LAMBDARANK
        # ====================================================

        print(
            "\n"
            + "-" * 100
        )

        print(
            "B. LAMBDARANK"
        )

        print(
            "-" * 100
        )

        model = train_lambda(
            train,
            valid,
            args.seed
        )

        lambda_metrics, lambda_pred = (
            evaluate_lambda(
                test,
                model,
                args.bootstrap
            )
        )

        print(
            "\nLambdaRank TEST "
            "(original human_rel 0-3)"
        )

        for k, v in (
            lambda_metrics.items()
        ):

            print(
                f"{k:15s}: {v:.4f}"
            )

        print(
            f"{'best_iteration':15s}: "
            f"{model.best_iteration_}"
        )

        importance = (
            lambda_importance(
                model
            )
        )

        print(
            "\nLambdaRank importance:"
        )

        print(
            importance.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

        lambda_pred.to_csv(
            f"fair_lambda_{mode}_predictions.csv",
            index=False
        )

        importance.to_csv(
            f"fair_lambda_{mode}_importance.csv",
            index=False
        )

        row = {
            "ranker":
                "LambdaRank",

            "supervision":
                mode,

            **lambda_metrics,

            "best_iteration":
                model.best_iteration_,
        }

        for f in WEIGHT_FEATURES:

            row[
                f"w_{f}"
            ] = np.nan

        result_rows.append(row)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    results = pd.DataFrame(
        result_rows
    )

    columns = [
        "ranker",
        "supervision",
        "NDCG@5",
        "NDCG@10",
        "MRR",
        "Spearman",
        "Kendall",
        "PairAgree",
        "NDCG5_CI_low",
        "NDCG5_CI_high",
        "best_iteration",
    ]

    print(
        "\n\n"
        + "=" * 140
    )

    print(
        "FINAL TRUE APPLES-TO-APPLES COMPARISON"
    )

    print(
        "=" * 140
    )

    print(
        results[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # PIVOT — EASY COMPARISON
    # ========================================================

    pivot = results.pivot(
        index="supervision",
        columns="ranker",
        values="NDCG@5"
    )

    if (
        "WeightedSum" in pivot.columns
        and
        "LambdaRank" in pivot.columns
    ):

        pivot[
            "Lambda_minus_Weighted"
        ] = (
            pivot["LambdaRank"]
            -
            pivot["WeightedSum"]
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NDCG@5 RANKER EFFECT"
    )

    print(
        "=" * 100
    )

    print(
        pivot.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # LABEL RETENTION WITHIN EACH RANKER
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUPERVISION RETENTION"
    )

    print(
        "=" * 100
    )

    for ranker in [
        "WeightedSum",
        "LambdaRank",
    ]:

        subset = results[
            results["ranker"]
            == ranker
        ]

        full = float(
            subset[
                subset["supervision"]
                == "full_graded"
            ]["NDCG@5"]
            .iloc[0]
        )

        print(
            f"\n{ranker}"
        )

        for mode in [
            "full_graded",
            "top_grade",
            "strict_top1",
        ]:

            score = float(
                subset[
                    subset["supervision"]
                    == mode
                ]["NDCG@5"]
                .iloc[0]
            )

            retention = (
                score / full
                if full > 0
                else np.nan
            )

            print(
                f"{mode:15s}: "
                f"{score:.4f} "
                f"({retention * 100:.2f}% "
                f"of full)"
            )

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        "fair_weighted_vs_lambda_results.csv",
        index=False
    )

    pivot.to_csv(
        "fair_weighted_vs_lambda_ndcg5_pivot.csv"
    )

    baseline.to_csv(
        "fair_baselines.csv",
        index=False
    )

    print(
        "\n"
        + "=" * 100
    )

    print("SAVED")

    print("=" * 100)

    print(
        "fair_shared_features.csv"
    )

    print(
        "fair_shared_split.csv"
    )

    print(
        "fair_baselines.csv"
    )

    print(
        "fair_weighted_vs_lambda_results.csv"
    )

    print(
        "fair_weighted_vs_lambda_ndcg5_pivot.csv"
    )

    for mode in modes:

        print(
            f"fair_labels_{mode}.csv"
        )

        print(
            f"fair_weighted_{mode}_predictions.csv"
        )

        print(
            f"fair_lambda_{mode}_predictions.csv"
        )


if __name__ == "__main__":
    main()
