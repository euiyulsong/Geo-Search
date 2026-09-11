#!/usr/bin/env python3
"""
POINTREC evaluation:
Compare original graded relevance (0/1/2/3) against binary relevance
where every relevant test item is treated equally:

    0 -> 0
    1 -> 1
    2 -> 1
    3 -> 1

This does NOT retrain either model. It only changes TEST evaluation labels.

Example:
python3 compare_graded_vs_binary_eval.py \
  --ridge fair_ridge_full_graded_predictions.csv \
  --lambda fair_lambda_full_graded_predictions.csv
"""

import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ridge", required=True, help="Ridge prediction CSV")
    p.add_argument("--lambda", dest="lambda_path", required=True,
                   help="LambdaRank prediction CSV")
    p.add_argument("--out", default="graded_vs_binary_eval.csv")
    return p.parse_args()


def normalize_columns(df):
    df = df.copy()

    aliases = {
        "qid": ["qid", "query_id", "queryid"],
        "poi_id": ["poi_id", "doc_id", "docid", "pid"],
        "human_rel": ["human_rel", "relevance", "rel", "label"],
        "score": ["score", "prediction", "pred", "rank_score"],
    }

    lower = {c.lower(): c for c in df.columns}
    rename = {}

    for canonical, names in aliases.items():
        if canonical in df.columns:
            continue
        for name in names:
            if name.lower() in lower:
                rename[lower[name.lower()]] = canonical
                break

    df = df.rename(columns=rename)

    required = ["qid", "poi_id", "human_rel", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nAvailable: {list(df.columns)}"
        )

    df["human_rel"] = pd.to_numeric(df["human_rel"], errors="raise")
    df["score"] = pd.to_numeric(df["score"], errors="raise")

    return df[required].copy()


def load_predictions(path, model):
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df.rename(columns={"score": f"score_{model}"})


def merge_predictions(ridge, lamb):
    df = ridge.merge(
        lamb,
        on=["qid", "poi_id"],
        how="inner",
        suffixes=("_ridge", "_lambda"),
        validate="one_to_one",
    )

    if len(df) != len(ridge) or len(df) != len(lamb):
        print(
            f"[WARN] Row counts differ: Ridge={len(ridge):,}, "
            f"Lambda={len(lamb):,}, intersection={len(df):,}"
        )

    mismatch = (
        df["human_rel_ridge"].astype(float)
        != df["human_rel_lambda"].astype(float)
    ).sum()

    if mismatch:
        raise ValueError(f"human_rel mismatch: {mismatch} rows")

    df["human_rel"] = df["human_rel_ridge"].astype(float)
    df["binary_rel"] = (df["human_rel"] > 0).astype(int)

    return df.drop(columns=["human_rel_ridge", "human_rel_lambda"])


def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    if len(rels) == 0:
        return 0.0

    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, len(rels) + 2))

    return float(np.sum(gains / discounts))


def ndcg_query(group, score_col, rel_col, k):
    ranked = group.sort_values(score_col, ascending=False).head(k)
    actual = dcg(ranked[rel_col].to_numpy())

    ideal = group.sort_values(rel_col, ascending=False).head(k)
    ideal_dcg = dcg(ideal[rel_col].to_numpy())

    if ideal_dcg <= 0:
        return np.nan

    return actual / ideal_dcg


def mrr_query(group, score_col, rel_col):
    ranked = group.sort_values(score_col, ascending=False)
    rels = ranked[rel_col].to_numpy()

    relevant = np.flatnonzero(rels > 0)
    if len(relevant) == 0:
        return np.nan

    return 1.0 / (relevant[0] + 1)


def evaluate(df, score_col, rel_col):
    rows = []

    for qid, g in df.groupby("qid", sort=False):
        rows.append({
            "qid": qid,
            "NDCG@5": ndcg_query(g, score_col, rel_col, 5),
            "NDCG@10": ndcg_query(g, score_col, rel_col, 10),
            "MRR": mrr_query(g, score_col, rel_col),
        })

    qdf = pd.DataFrame(rows)

    return {
        "NDCG@5": qdf["NDCG@5"].mean(),
        "NDCG@10": qdf["NDCG@10"].mean(),
        "MRR": qdf["MRR"].mean(),
    }, qdf


def winner(delta, eps=1e-12):
    if delta > eps:
        return "LambdaRank"
    if delta < -eps:
        return "Ridge"
    return "Tie"


def main():
    args = parse_args()

    ridge = load_predictions(args.ridge, "ridge")
    lamb = load_predictions(args.lambda_path, "lambda")
    df = merge_predictions(ridge, lamb)

    print("=" * 110)
    print("POINTREC — GRADED vs BINARY TEST RELEVANCE")
    print("=" * 110)
    print(f"rows    : {len(df):,}")
    print(f"queries : {df['qid'].nunique():,}")

    print("\nOriginal relevance distribution")
    print(df["human_rel"].value_counts().sort_index())

    print("\nBinary relevance distribution")
    print(df["binary_rel"].value_counts().sort_index())

    all_results = []
    query_results = {}

    for eval_name, rel_col in [
        ("graded_0_1_2_3", "human_rel"),
        ("binary_0_1", "binary_rel"),
    ]:
        for model, score_col in [
            ("Ridge", "score_ridge"),
            ("LambdaRank", "score_lambda"),
        ]:
            metrics, qdf = evaluate(df, score_col, rel_col)
            query_results[(eval_name, model)] = qdf

            all_results.append({
                "evaluation": eval_name,
                "model": model,
                **metrics,
            })

    results = pd.DataFrame(all_results)

    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    comparison = []

    for metric in ["NDCG@5", "NDCG@10", "MRR"]:
        g_r = results[
            (results["evaluation"] == "graded_0_1_2_3")
            & (results["model"] == "Ridge")
        ][metric].iloc[0]

        g_l = results[
            (results["evaluation"] == "graded_0_1_2_3")
            & (results["model"] == "LambdaRank")
        ][metric].iloc[0]

        b_r = results[
            (results["evaluation"] == "binary_0_1")
            & (results["model"] == "Ridge")
        ][metric].iloc[0]

        b_l = results[
            (results["evaluation"] == "binary_0_1")
            & (results["model"] == "LambdaRank")
        ][metric].iloc[0]

        graded_delta = g_l - g_r
        binary_delta = b_l - b_r

        comparison.append({
            "metric": metric,
            "graded_ridge": g_r,
            "graded_lambda": g_l,
            "graded_delta": graded_delta,
            "graded_winner": winner(graded_delta),
            "binary_ridge": b_r,
            "binary_lambda": b_l,
            "binary_delta": binary_delta,
            "binary_winner": winner(binary_delta),
            "winner_same": winner(graded_delta) == winner(binary_delta),
            "delta_change": binary_delta - graded_delta,
        })

    comparison = pd.DataFrame(comparison)

    print("\n" + "=" * 110)
    print("MODEL-SELECTION COMPARISON")
    print("=" * 110)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Query-level agreement: does Lambda beat Ridge on each query under
    # graded vs binary labels?
    query_rows = []

    for metric in ["NDCG@5", "NDCG@10", "MRR"]:
        gr = query_results[("graded_0_1_2_3", "Ridge")][["qid", metric]].rename(
            columns={metric: "graded_ridge"}
        )
        gl = query_results[("graded_0_1_2_3", "LambdaRank")][["qid", metric]].rename(
            columns={metric: "graded_lambda"}
        )
        br = query_results[("binary_0_1", "Ridge")][["qid", metric]].rename(
            columns={metric: "binary_ridge"}
        )
        bl = query_results[("binary_0_1", "LambdaRank")][["qid", metric]].rename(
            columns={metric: "binary_lambda"}
        )

        q = gr.merge(gl, on="qid").merge(br, on="qid").merge(bl, on="qid")
        q["graded_delta"] = q["graded_lambda"] - q["graded_ridge"]
        q["binary_delta"] = q["binary_lambda"] - q["binary_ridge"]

        q["graded_sign"] = np.sign(q["graded_delta"])
        q["binary_sign"] = np.sign(q["binary_delta"])
        q["winner_same"] = q["graded_sign"] == q["binary_sign"]

        valid = q[
            q["graded_delta"].notna()
            & q["binary_delta"].notna()
        ]

        query_rows.append({
            "metric": metric,
            "queries": len(valid),
            "query_winner_agreement": valid["winner_same"].mean()
            if len(valid) else np.nan,
            "graded_delta_mean": valid["graded_delta"].mean()
            if len(valid) else np.nan,
            "binary_delta_mean": valid["binary_delta"].mean()
            if len(valid) else np.nan,
        })

    query_summary = pd.DataFrame(query_rows)

    print("\n" + "=" * 110)
    print("QUERY-LEVEL WINNER AGREEMENT")
    print("=" * 110)
    print(query_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results.to_csv(args.out, index=False)

    comparison_path = args.out.replace(".csv", "_comparison.csv")
    query_path = args.out.replace(".csv", "_query_agreement.csv")

    comparison.to_csv(comparison_path, index=False)
    query_summary.to_csv(query_path, index=False)

    print("\n" + "=" * 110)
    print("INTERPRETATION")
    print("=" * 110)
    print("graded_0_1_2_3:")
    print("  human_rel 0/1/2/3 is used exactly as annotated.")
    print("")
    print("binary_0_1:")
    print("  human_rel=0 -> 0")
    print("  human_rel=1/2/3 -> 1")
    print("")
    print("If winner_same=True, collapsing relevance grades does not change")
    print("the overall model winner for that metric.")
    print("")
    print("query_winner_agreement measures how often the per-query Ridge-vs-Lambda")
    print("preference remains identical after 1/2/3 are collapsed to 1.")
    print("")
    print("Saved:")
    print(args.out)
    print(comparison_path)
    print(query_path)


if __name__ == "__main__":
    main()
