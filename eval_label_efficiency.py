#!/usr/bin/env python3
"""
Evaluation-label-efficiency experiment for POINTREC.

Question:
    If we label only K evaluation results per query, can we still estimate
    model quality / choose the better model similarly to full-qrels evaluation?

This script DOES NOT train a model.
It consumes prediction CSVs produced by the previous ranking experiment.

It compares:
  1) Full-qrels NDCG@5 / NDCG@10 / MRR (ground truth)
  2) Partial-label estimates with K labels/query, repeated over many Monte Carlo trials

Policies:
  - random: uniformly sample K candidates/query
  - top_union: prioritize documents appearing near the top of either model
               (recommended for model comparison)
  - model_top: label top-K for each model separately (less fair for direct A/B)

Partial-label metric:
  Unjudged documents are treated as 0 ONLY for the estimator.
  Because this estimator is biased when K is tiny, the main outputs also include:
    - absolute error vs full metric
    - model-selection agreement
    - delta error: estimated (B-A) vs true (B-A)
    - sign agreement of B-A
    - Spearman correlation across trials where meaningful

Example:
python3 eval_label_efficiency.py \
  --ridge fair_ridge_full_graded_predictions.csv \
  --lambda fair_lambda_full_graded_predictions.csv \
  --ks 1,2,3,5,10 \
  --trials 1000 \
  --seed 42
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ridge", required=True, help="Prediction CSV for model A (Ridge)")
    p.add_argument("--lambda", dest="lambda_path", required=True,
                   help="Prediction CSV for model B (LambdaRank)")
    p.add_argument("--ks", default="1,2,3,5,10",
                   help="Labels/query to simulate, comma-separated")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy", choices=["random", "top_union"], default="random")
    p.add_argument("--top_pool", type=int, default=10,
                   help="For top_union: build sampling priority from each model's top-N")
    p.add_argument("--out", default="eval_label_efficiency")
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
        for n in names:
            if n.lower() in lower:
                rename[lower[n.lower()]] = canonical
                break

    df = df.rename(columns=rename)

    required = ["qid", "poi_id", "human_rel", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {list(df.columns)}"
        )

    df["human_rel"] = pd.to_numeric(df["human_rel"], errors="raise")
    df["score"] = pd.to_numeric(df["score"], errors="raise")
    return df


def load_predictions(path, model_name):
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = df[["qid", "poi_id", "human_rel", "score"]].copy()
    df = df.rename(columns={"score": f"score_{model_name}"})
    return df


def merge_predictions(a, b):
    m = a.merge(
        b,
        on=["qid", "poi_id"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    if len(m) != len(a) or len(m) != len(b):
        print(
            f"[WARN] prediction rows differ: A={len(a):,}, B={len(b):,}, "
            f"intersection={len(m):,}"
        )

    mismatch = (
        m["human_rel_a"].astype(float) != m["human_rel_b"].astype(float)
    ).sum()
    if mismatch:
        raise ValueError(f"human_rel mismatch between model files: {mismatch} rows")

    m["human_rel"] = m["human_rel_a"]
    return m.drop(columns=["human_rel_a", "human_rel_b"])


def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    if len(rels) == 0:
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(gains / discounts))


def ndcg_for_query(group, score_col, rel_col, k):
    ranked = group.sort_values(score_col, ascending=False).head(k)
    dcg_val = dcg(ranked[rel_col].to_numpy())

    ideal = group.sort_values(rel_col, ascending=False).head(k)
    idcg = dcg(ideal[rel_col].to_numpy())

    if idcg <= 0:
        return np.nan
    return dcg_val / idcg


def mrr_for_query(group, score_col, rel_col):
    ranked = group.sort_values(score_col, ascending=False)
    rels = ranked[rel_col].to_numpy()
    positive = np.flatnonzero(rels > 0)
    if len(positive) == 0:
        return np.nan
    return 1.0 / float(positive[0] + 1)


def evaluate(df, score_col, rel_col):
    ndcg5 = []
    ndcg10 = []
    mrr = []

    for _, g in df.groupby("qid", sort=False):
        x = ndcg_for_query(g, score_col, rel_col, 5)
        y = ndcg_for_query(g, score_col, rel_col, 10)
        z = mrr_for_query(g, score_col, rel_col)

        if not np.isnan(x):
            ndcg5.append(x)
        if not np.isnan(y):
            ndcg10.append(y)
        if not np.isnan(z):
            mrr.append(z)

    return {
        "NDCG@5": float(np.mean(ndcg5)) if ndcg5 else np.nan,
        "NDCG@10": float(np.mean(ndcg10)) if ndcg10 else np.nan,
        "MRR": float(np.mean(mrr)) if mrr else np.nan,
    }


def choose_random_labels(df, k, rng):
    chosen = []
    for _, g in df.groupby("qid", sort=False):
        idx = g.index.to_numpy()
        n = min(k, len(idx))
        chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    return np.asarray(chosen, dtype=int)


def choose_top_union_labels(df, k, rng, top_pool):
    """
    Shared judging pool for fair A/B comparison.

    For each query, create the union of top-N from A and B.
    If k < union size, sample k uniformly from that union.
    If union has fewer than k, fill randomly from remaining candidates.
    """
    chosen = []

    for _, g in df.groupby("qid", sort=False):
        top_a = g.nlargest(min(top_pool, len(g)), "score_ridge").index.to_numpy()
        top_b = g.nlargest(min(top_pool, len(g)), "score_lambda").index.to_numpy()
        pool = np.unique(np.concatenate([top_a, top_b]))

        if len(pool) >= k:
            take = rng.choice(pool, size=k, replace=False)
        else:
            take = list(pool)
            remaining = np.setdiff1d(g.index.to_numpy(), pool, assume_unique=False)
            need = min(k - len(take), len(remaining))
            if need:
                take.extend(rng.choice(remaining, size=need, replace=False).tolist())
            take = np.asarray(take, dtype=int)

        chosen.extend(np.asarray(take).tolist())

    return np.asarray(chosen, dtype=int)


def make_partial_relevance(df, judged_idx):
    """
    Naive partial-qrels estimator:
      judged rows -> true human_rel
      unjudged rows -> 0

    This intentionally measures how bad/good a cheap partial-label evaluation
    would be relative to the full-qrels ground truth.
    """
    out = df.copy()
    out["partial_rel"] = 0.0
    out.loc[judged_idx, "partial_rel"] = out.loc[judged_idx, "human_rel"]
    return out


def sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def main():
    args = parse_args()
    ks = sorted(set(int(x.strip()) for x in args.ks.split(",") if x.strip()))
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers")

    a = load_predictions(args.ridge, "ridge")
    b = load_predictions(args.lambda_path, "lambda")
    df = merge_predictions(a, b)

    print("=" * 110)
    print("POINTREC — EVALUATION LABEL EFFICIENCY")
    print("Can K human relevance labels/query approximate full-qrels evaluation?")
    print("=" * 110)
    print(f"rows    : {len(df):,}")
    print(f"queries : {df['qid'].nunique():,}")
    print(f"policy  : {args.policy}")
    print(f"trials  : {args.trials:,}")
    print(f"ks      : {ks}")

    full_r = evaluate(df, "score_ridge", "human_rel")
    full_l = evaluate(df, "score_lambda", "human_rel")

    print("\nFULL-QRELS GROUND TRUTH")
    print("-" * 110)
    print(
        f"{'metric':10s} {'Ridge':>12s} {'LambdaRank':>12s} "
        f"{'Lambda-Ridge':>15s} {'winner':>12s}"
    )
    for metric in ["NDCG@5", "NDCG@10", "MRR"]:
        delta = full_l[metric] - full_r[metric]
        winner = "Lambda" if delta > 0 else ("Ridge" if delta < 0 else "Tie")
        print(
            f"{metric:10s} {full_r[metric]:12.4f} {full_l[metric]:12.4f} "
            f"{delta:15.4f} {winner:>12s}"
        )

    rng_master = np.random.default_rng(args.seed)
    rows = []

    for k in ks:
        print(f"\nK={k} labels/query ...")

        for trial in range(args.trials):
            rng = np.random.default_rng(
                int(rng_master.integers(0, np.iinfo(np.int32).max))
            )

            if args.policy == "random":
                judged = choose_random_labels(df, k, rng)
            else:
                judged = choose_top_union_labels(df, k, rng, args.top_pool)

            partial = make_partial_relevance(df, judged)

            er = evaluate(partial, "score_ridge", "partial_rel")
            el = evaluate(partial, "score_lambda", "partial_rel")

            for metric in ["NDCG@5", "NDCG@10", "MRR"]:
                true_delta = full_l[metric] - full_r[metric]
                est_delta = el[metric] - er[metric]

                rows.append({
                    "k": k,
                    "trial": trial,
                    "metric": metric,
                    "ridge_est": er[metric],
                    "lambda_est": el[metric],
                    "ridge_full": full_r[metric],
                    "lambda_full": full_l[metric],
                    "ridge_abs_error": abs(er[metric] - full_r[metric]),
                    "lambda_abs_error": abs(el[metric] - full_l[metric]),
                    "true_delta": true_delta,
                    "estimated_delta": est_delta,
                    "delta_abs_error": abs(est_delta - true_delta),
                    "winner_agree": int(sign(est_delta) == sign(true_delta)),
                    "labels_total": len(judged),
                })

    trials = pd.DataFrame(rows)

    summary_rows = []
    for (k, metric), g in trials.groupby(["k", "metric"], sort=True):
        summary_rows.append({
            "k": k,
            "metric": metric,
            "labels/query": k,
            "total_labels": int(g["labels_total"].iloc[0]),
            "ridge_full": g["ridge_full"].iloc[0],
            "ridge_est_mean": g["ridge_est"].mean(),
            "ridge_MAE": g["ridge_abs_error"].mean(),
            "lambda_full": g["lambda_full"].iloc[0],
            "lambda_est_mean": g["lambda_est"].mean(),
            "lambda_MAE": g["lambda_abs_error"].mean(),
            "true_delta": g["true_delta"].iloc[0],
            "estimated_delta_mean": g["estimated_delta"].mean(),
            "delta_MAE": g["delta_abs_error"].mean(),
            "model_selection_agreement": g["winner_agree"].mean(),
            "estimated_delta_p05": g["estimated_delta"].quantile(0.05),
            "estimated_delta_p95": g["estimated_delta"].quantile(0.95),
        })

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 110)
    print("SUMMARY — MAIN QUESTION")
    print("=" * 110)

    show = summary[summary["metric"] == "NDCG@5"].copy()
    cols = [
        "k", "total_labels",
        "ridge_full", "ridge_est_mean", "ridge_MAE",
        "lambda_full", "lambda_est_mean", "lambda_MAE",
        "true_delta", "estimated_delta_mean", "delta_MAE",
        "model_selection_agreement",
        "estimated_delta_p05", "estimated_delta_p95",
    ]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    prefix = Path(args.out)
    trials_path = prefix.with_name(prefix.name + "_trials.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 110)
    print("INTERPRETATION GUIDE")
    print("=" * 110)
    print("model_selection_agreement:")
    print("  1.00 = partial labels always choose the same winner as full qrels")
    print("  0.50 = roughly coin-flip model selection")
    print("")
    print("delta_MAE:")
    print("  Error in estimating the model gap (LambdaRank - Ridge). Lower is better.")
    print("")
    print("IMPORTANT:")
    print("  With K=1, partial NDCG itself is a biased estimator because almost all")
    print("  unjudged documents are treated as relevance 0. Therefore the strongest")
    print("  practical question is usually whether model-selection agreement remains")
    print("  high, not whether partial NDCG equals full NDCG exactly.")
    print("")
    print("Saved:")
    print(trials_path)
    print(summary_path)


if __name__ == "__main__":
    main()
