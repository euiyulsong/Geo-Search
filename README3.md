# POINTREC — Weighted Sum vs. LambdaRank Fair Comparison

## 1. 실험 목적

동일한 POINTREC 데이터와 feature를 사용하여 **label supervision의 정보량**과 **ranking model의 차이**가 성능에 미치는 영향을 비교한다.

비교하는 supervision은 다음 3가지다.

| Supervision      | 설명                                              |
| ---------------- | ----------------------------------------------- |
| **Full Graded**  | 원본 human relevance `0/1/2/3` 전체 사용              |
| **Top-Grade**    | query별 최고 relevance POI 전체를 `1`, 나머지를 `0`       |
| **Strict Top-1** | query별 최고 relevance POI 중 무작위로 **1개만 positive** |

두 ranker 모두 **동일한 supervision label**을 사용하고, 최종 평가는 동일한 원본 `human_rel 0–3`으로 수행했다.

---

## 2. 실험 데이터

* Queries: **112**
* Judged query-POI pairs: **5,108**
* POI corpus: **695,035**
* Qrel POI join coverage: **99.26%**
* Train: **60 queries / 2,751 pairs**
* Validation: **20 queries / 914 pairs**
* Test: **32 queries / 1,443 pairs**

이전 정상 실험과 동일한 split이 정확히 재현되었다. 

### Features

사용 feature는 모든 실험에서 동일하다.

```text
BM25
Category Score
Token Overlap
Rating Score
Popularity Score
```

모든 feature가 정상적으로 활성화되어 있으며, 특히 이전 코드에서 문제가 있었던 `category_score`도 **4,580 / 5,108 pairs**에서 non-zero로 확인되었다. 

---

## 3. Baseline

| Method            |     NDCG@5 |    NDCG@10 |    MRR |  PairAgree |
| ----------------- | ---------: | ---------: | -----: | ---------: |
| **Category Only** | **0.7290** | **0.7526** | 0.5588 | **0.7321** |
| BM25 Only         |     0.5145 |     0.5627 | 0.3981 |     0.5314 |

Category-only baseline이 매우 강하다. 이전 정상 실험의 `Category ≈ .729`, `BM25 ≈ .515`도 그대로 재현되었다. 

---

# 4. 최종 결과

## NDCG@5 기준

| Supervision      | Weighted Sum | LambdaRank | Lambda − Weighted |
| ---------------- | -----------: | ---------: | ----------------: |
| **Full Graded**  |   **0.7597** |     0.7097 |           -0.0500 |
| **Top-Grade**    |   **0.7335** |     0.7049 |           -0.0287 |
| **Strict Top-1** |       0.7074 | **0.7135** |           +0.0061 |

전체 metric 결과는 다음과 같다. 

| Ranker           | Supervision  |     NDCG@5 |    NDCG@10 |        MRR |  PairAgree |
| ---------------- | ------------ | ---------: | ---------: | ---------: | ---------: |
| **Weighted Sum** | Full         | **0.7597** | **0.7728** | **0.6469** |     0.7194 |
| LambdaRank       | Full         |     0.7097 |     0.7395 |     0.5464 | **0.7285** |
| **Weighted Sum** | Top-grade    | **0.7335** | **0.7484** | **0.6753** |     0.6657 |
| LambdaRank       | Top-grade    |     0.7049 |     0.7269 |     0.5314 | **0.6964** |
| Weighted Sum     | Strict Top-1 |     0.7074 | **0.7249** |     0.5080 |     0.6662 |
| **LambdaRank**   | Strict Top-1 | **0.7135** |     0.7210 | **0.6410** | **0.6992** |

---

# 5. Supervision 감소 효과

## Weighted Sum

```text
Full Graded
0.7597  (100.0%)
   │
   ▼ -0.0262
Top-Grade
0.7335  (96.55%)
   │
   ▼ -0.0261
Strict Top-1
0.7074  (93.11%)
```

Weighted Sum에서는 supervision을 줄일수록 **일관되게 성능이 감소**한다. 

### 해석

**Full → Top-grade**

* 0.7597 → 0.7335
* 절대 감소: **-0.0262**
* Full 성능의 **96.55% 유지**

따라서 세밀한 `0/1/2/3` annotation이 도움이 되지만, **최고 relevance POI들을 binary positive로 확보하는 것만으로도 대부분의 ranking quality를 유지**했다.

**Full → Strict Top-1**

* 0.7597 → 0.7074
* 절대 감소: **-0.0523**
* Full 성능의 **93.11% 유지**

Query당 positive를 단 하나만 제공해도 상당한 성능이 유지되지만, supervision coverage 감소에 따른 성능 하락은 명확하다.

---

# 6. LambdaRank 결과

LambdaRank에서는 다른 패턴이 나타난다.

```text
Full Graded       0.7097
Top-Grade         0.7049
Strict Top-1      0.7135
```

Full 대비 retention은:

* Top-grade: **99.32%**
* Strict Top-1: **100.54%**

즉 이번 실험에서 LambdaRank는 **label granularity에 거의 민감하지 않았다.** 

다만 이를

> Strict Top-1이 Full Graded보다 좋다.

라고 해석하면 안 된다.

NDCG@5 차이는 `+0.0038`에 불과하고 confidence interval도 크게 겹친다.

| Supervision  | NDCG@5 |           95% CI |
| ------------ | -----: | ---------------: |
| Full         | 0.7097 | [0.6492, 0.7751] |
| Top-grade    | 0.7049 | [0.6410, 0.7631] |
| Strict Top-1 | 0.7135 | [0.6523, 0.7697] |



따라서 현재 결론은:

> **LambdaRank performance was relatively insensitive to supervision granularity.**

---

# 7. Weighted Sum vs. LambdaRank

### Full Graded

```text
Weighted Sum   0.7597
LambdaRank     0.7097
────────────────────
Difference    +0.0500
```

60-query의 작은 training set에서는 복잡한 tree-based ranker보다 **단순한 Weighted Sum이 더 잘 generalize**했다.

Weighted Sum이 선택한 weight는:

```text
BM25             0.00
Category          0.90
Token overlap     0.05
Rating            0.05
Popularity        0.00
```

즉 사실상 **강력한 category signal을 중심으로 작은 correction만 추가하는 구조**다. 

LambdaRank 역시 category가 가장 중요한 feature라는 것은 학습했다.

| Feature       | Gain importance |
| ------------- | --------------: |
| Category      |       **54.7%** |
| BM25          |           18.3% |
| Token overlap |           13.1% |
| Popularity    |           10.1% |
| Rating        |            3.9% |



따라서 feature 자체의 문제라기보다 **작은 데이터에서 LambdaRank의 추가 model capacity가 generalization 이득으로 연결되지 않은 것**으로 해석할 수 있다.

---

# 8. 주의할 점

### Strict Top-1 seed variance

현재 Strict Top-1은 `seed=42` 한 번의 결과다.

Query마다 여러 최고 relevance POI 중 하나를 무작위 선택하기 때문에:

```text
Weighted:    0.7074
LambdaRank:  0.7135
Difference: +0.0061
```

정도의 작은 차이는 seed에 따라 충분히 뒤집힐 수 있다.

따라서 **10–20개 이상의 random seed**에 대해:

```text
NDCG@5 = mean ± std
```

를 비교하는 것이 필요하다.

### LambdaRank early stopping

Top-grade LambdaRank는 `best_iteration=1`, Strict Top-1은 `5`에 불과했다.  

따라서 binary supervision에서 현재 hyperparameter가 LambdaRank의 최적 성능을 충분히 끌어냈는지는 추가 확인이 필요하다.

### Statistical significance

동일한 32개 test query에서 두 ranker를 평가했으므로 단순히 각각의 CI가 겹치는지만 볼 것이 아니라 **query-level paired bootstrap 또는 permutation test**가 적절하다.

---

# 9. 결론

이번 fair comparison에서 가장 명확한 결과는 다음과 같다.

> **Fine-grained relevance labels improve ranking quality, but most of the performance can be retained with substantially weaker supervision.**

특히 Weighted Sum에서는:

```text
Full 0–3 labels       0.7597
        ↓
Top-grade binary      0.7335   → 96.6% retained
        ↓
One positive/query    0.7074   → 93.1% retained
```

반면 LambdaRank에서는:

```text
Full                  0.7097
Top-grade             0.7049
Strict Top-1          0.7135
```

로 supervision granularity에 따른 차이가 거의 나타나지 않았다.

따라서 현재 실험이 보여주는 핵심은 **“더 세밀한 label은 분명 가치가 있지만 annotation cost 대비 marginal gain은 생각보다 작을 수 있으며, 그 가치는 ranker와 데이터 규모에 따라 달라진다”**는 것이다.

또한 이 POINTREC의 **60-query low-data regime에서는 단순하고 category-dominant한 Weighted Sum이 LambdaRank보다 더 좋은 NDCG@5를 보였다.** 다만 최종적으로 강한 결론을 내리려면 **Strict Top-1 multi-seed + paired significance test**를 추가하는 것이 적절하다.
