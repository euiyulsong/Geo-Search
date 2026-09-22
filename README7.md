## Top-1 Label Evaluation vs. Full Graded Evaluation

### Experiment Setup

Geo-Search의 기존 POINTREC test split을 사용하여, query당 하나의 relevance label만 사용하는 sparse evaluation이 전체 graded relevance evaluation을 얼마나 잘 근사하는지 측정하였다.

* Test queries: **32**
* Query–POI pairs: **2,083**
* Ranking systems: **1,001**
* Features:

  * `bm25`
  * `category_score`
  * `token_overlap`
  * `rating_score`
  * `popularity_score`
* Weight step: **0.1**
* Full relevance labels: **0 / 1 / 2 / 3**
* Full evaluation metric: **NDCG@5**
* Sparse evaluation:

  * 각 query에서 가장 높은 relevance grade를 가진 POI 중 **1개만 positive label로 유지**
  * 최고 relevance POI가 여러 개인 경우 random sampling
* Sparse-label trials: **1,000**

Test set의 relevance distribution은 다음과 같다.

| Relevance | # Pairs |
| --------: | ------: |
|         0 |     876 |
|         1 |     370 |
|         2 |     474 |
|         3 |     363 |

32개 query 모두 top-1 sparse evaluation이 가능하였다. 특히 **32개 중 31개 query에서 최고 relevance를 가진 POI가 2개 이상 존재**하였다.

---

## Results

| Metric                       |      Mean |   Std |    P05 | Median |   P95 |
| ---------------------------- | --------: | ----: | -----: | -----: | ----: |
| Kendall τ                    | **0.344** | 0.231 | -0.095 |  0.386 | 0.646 |
| Spearman ρ                   | **0.464** | 0.307 | -0.169 |  0.544 | 0.829 |
| Top-10 system overlap        | **0.229** | 0.165 |  0.000 |  0.200 | 0.500 |
| Exact best-system agreement  | **0.077** | 0.267 |  0.000 |  0.000 | 1.000 |
| Full winner rank under Top-1 | **183.5** | 257.5 |      1 |     40 |   789 |

Full graded evaluation에서 가장 높은 성능을 기록한 system은 `ws_00283`이었다.

```text
bm25                 = 0.00
category_score       = 0.90
token_overlap        = 0.00
rating_score         = 0.10
popularity_score     = 0.00

Full NDCG@5          = 0.707650
```

---

## Analysis

### 1. Top-1 evaluation은 full graded system ranking을 충분히 보존하지 못했다

Full graded evaluation과 Top-1 sparse evaluation 사이의 평균 Kendall correlation은

**τ = 0.344**

였으며, Spearman correlation은

**ρ = 0.464**

였다.

두 값 모두 어느 정도 양의 상관관계는 존재하지만, **full evaluation의 system ordering을 안정적으로 대체할 정도로 높은 correlation은 아니다.**

특히 Kendall τ의 median도 0.386에 불과하여, 일반적인 trial에서도 상당수 system pair의 상대적인 순서가 바뀌었다.

더 중요한 것은 trial 간 variation이다.

```text
Kendall τ
P05    = -0.095
Median =  0.386
P95    =  0.646
```

일부 top-1 sampling에서는 correlation이 **음수까지 내려갔다.**

따라서 POINTREC에서는 어떤 POI 하나를 top-1 gold로 선택하는지에 따라 evaluation 결과가 크게 달라질 수 있다.

---

### 2. 가장 좋은 system을 고르는 용도로는 특히 불안정했다

Full evaluation에서 가장 좋은 system과 Top-1 evaluation에서 가장 좋은 system이 정확히 일치한 비율은

**7.7%**

뿐이었다.

즉 1,000개의 sparse-label trials 중 약 77개 정도에서만 동일한 best system을 선택하였다.

또한 full evaluation의 실제 winner는 Top-1 evaluation에서 평균적으로

**183위 / 1,001 systems**

까지 내려갔다.

Median rank도 **40위**였다.

```text
Full winner rank under Top-1

mean   = 183
median = 40
P95    = 789
```

따라서 Top-1 label만 사용하여 hyperparameter 또는 ranking model을 선택할 경우 full graded evaluation에서 가장 좋은 configuration을 놓칠 가능성이 상당히 높다.

---

### 3. Top-k 후보군 수준에서도 차이가 컸다

Full evaluation의 Top-10 systems와 Top-1 sparse evaluation의 Top-10 systems 간 overlap은 평균

**22.9%**

였다.

즉 평균적으로 full Top-10 중 약 **2.3개 system만** Top-1 evaluation의 Top-10에도 포함되었다.

Median overlap은 20%였으며, 일부 trial에서는 overlap이 **0%**였다.

이는 단순히 1등 system만 불안정한 것이 아니라, **상위권 model/configuration의 ordering 자체가 상당히 변화한다는 것**을 보여준다.

---

## Why Is Top-1 Particularly Unstable on POINTREC?

가장 중요한 원인은 **multiple relevant POIs**이다.

32개 test queries 중 **31개에서 동일한 최고 relevance grade를 가진 POI가 여러 개 존재**하였다.

예를 들어 어떤 query에 relevance 3인 POI가 다음과 같이 존재한다고 가정할 수 있다.

```text
POI A → 3
POI B → 3
POI C → 3
POI D → 2
...
```

Full graded evaluation에서는 A, B, C를 모두 높은-quality result로 인정한다.

하지만 현재 Top-1 experiment에서는 매 trial마다 다음 중 하나만 gold가 된다.

```text
Trial 1 → A만 relevant
Trial 2 → C만 relevant
Trial 3 → B만 relevant
```

따라서 A 대신 B를 높은 rank에 배치한 system은 실제 full relevance 기준에서는 좋은 system임에도 불구하고 sparse evaluation에서는 실패로 처리될 수 있다.

POINTREC에서는 거의 모든 query가 이러한 구조를 가지므로 random Top-1 선택에 대한 sensitivity가 매우 높게 나타난다.

---

## Important Interpretation

이번 결과가 의미하는 것은 다음과 같다.

> **POINTREC에서 기존 graded labels로부터 최고 relevance POI 하나를 임의로 선택하여 evaluation하는 방식은 full graded ranking evaluation을 안정적으로 대체하지 못한다.**

다만 이번 실험은 다음 주장까지 검증한 것은 아니다.

> **Human annotator에게 처음부터 query당 가장 좋은 POI 하나만 annotation하도록 했을 때도 성능이 낮다.**

현재 실험에서는 이미 여러 POI가 relevance 3으로 labeling되어 있는데 그중 하나를 **random하게 선택**하였다.

따라서 실제 annotation 단계에서 사람이

```text
이 query에 대해 정말 가장 좋은 POI 하나
```

를 명시적으로 선택하는 설정과는 차이가 있다.

즉 현재 실험은 보다 정확히는 **random single-positive sparsification**의 robustness를 측정한다.

---

## Conclusion

POINTREC에서 query당 하나의 relevance judgment만 사용하는 sparse evaluation은 full graded evaluation과 어느 정도 correlation을 보였지만, system selection을 안정적으로 보존하기에는 부족하였다.

주요 결과는 다음과 같다.

* Kendall τ: **0.344**
* Spearman ρ: **0.464**
* Top-10 overlap: **22.9%**
* Best-system agreement: **7.7%**
* Full-evaluation winner의 sparse-evaluation median rank: **40 / 1,001**

특히 **31/32 queries가 multiple maximum-relevance POIs를 가지고 있다는 점**이 single-positive evaluation의 높은 variance에 크게 기여하는 것으로 보인다.

따라서 현재 POINTREC 설정에서는 **query당 random Top-1 label 하나만 사용하는 evaluation보다 multiple relevance judgments 또는 graded relevance judgments를 유지하는 것이 system comparison에 더 안정적이다.**

### Recommended Next Experiment

다음 단계에서는 labeling budget에 따른 trade-off를 직접 측정하는 것이 적절하다.

```text
Full graded labels
        ↓
K=1
K=2
K=3
K=5
All max-grade labels
        ↓
Full NDCG@5 system ranking과 비교
        ↓
Kendall τ / Spearman ρ / Top-10 overlap / Winner agreement
```

이를 통해 단순히 "Top-1이 부족하다"에서 끝나지 않고,

> **POINTREC에서는 query당 몇 개의 relevance judgments가 있어야 full ranked-list evaluation을 안정적으로 근사할 수 있는가?**

를 정량적으로 제시할 수 있다.
