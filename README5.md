# Evaluation Label Efficiency 실험 결과

## 1. 실험 목적

이번 실험의 질문은 다음과 같다.

> **평가셋에서 query당 POI 하나만 human labeling해도 full labeling과 비슷하게 모델 성능을 판단할 수 있는가?**

학습 데이터의 label을 줄이는 실험이 아니라, **evaluation annotation cost를 줄일 수 있는지** 확인하는 실험이다.

* Test queries: **32**
* Test pairs: **2,083**
* Sampling: query별 random
* Monte Carlo: **1,000회**
* Labels/query: **1, 2, 3, 5, 10**

---

## 2. Full-label Ground Truth

전체 relevance label을 사용했을 때:

| Metric  |  Ridge | LambdaRank | Lambda − Ridge | Winner     |
| ------- | -----: | ---------: | -------------: | ---------- |
| NDCG@5  | 0.4776 | **0.5958** |        +0.1181 | LambdaRank |
| NDCG@10 | 0.5130 | **0.6228** |        +0.1098 | LambdaRank |
| MRR     | 0.8646 | **0.9073** |        +0.0427 | LambdaRank |

따라서 실제 ground truth에서는 **LambdaRank가 명확하게 우세**하다.

NDCG@5 차이가 `+0.1181`이므로 모델 간 차이가 아주 작은 상황도 아니다.

---

## 3. Query당 K개만 Labeling한 결과

### NDCG@5 기준

| Labels/query | 총 Label | Ridge 추정 | Lambda 추정 |    Δ 추정 |  Δ MAE | 모델 선택 정확도 |
| -----------: | ------: | -------: | --------: | ------: | -----: | --------: |
|        **1** |  **32** |   0.0626 |    0.0707 | +0.0081 | 0.1110 | **52.4%** |
|            2 |      64 |   0.0692 |    0.0783 | +0.0092 | 0.1093 |     56.2% |
|            3 |      96 |   0.0747 |    0.0869 | +0.0122 | 0.1061 |     60.6% |
|            5 |     160 |   0.0829 |    0.1005 | +0.0176 | 0.1006 |     67.5% |
|           10 |     320 |   0.1092 |    0.1339 | +0.0247 | 0.0934 | **76.5%** |

## 4. 핵심 결론: Random 1-label/query는 안 됨

가장 중요한 숫자는:

```text
K=1
총 annotation = 32개
model selection agreement = 52.4%
```

즉 full evaluation에서는 LambdaRank가 이기는 게 정답인데, query당 랜덤 POI 하나만 labeling하면 **1000번 중 약 절반 정도만 LambdaRank가 더 좋다고 판단한다.**

52.4%는 사실상 **동전 던지기 수준**이다.

따라서 현재 결과에서는:

> **평가 query당 random result 하나만 labeling하는 방식으로 모델 성능을 비교하는 것은 신뢰하기 어렵다.**

라고 결론 내리는 게 맞다.

---

## 5. Label을 늘리면 점진적으로 좋아짐

모델 선택 정확도가:

```text
1 label  → 52.4%
2 labels → 56.2%
3 labels → 60.6%
5 labels → 67.5%
10 labels → 76.5%
```

로 꾸준히 증가한다.

즉 **annotation을 늘릴수록 full-qrels 판단에 가까워지는 정상적인 label-efficiency curve**가 나타난다.

다만 query당 10개를 labeling해도 76.5%이므로 random sampling은 상당히 비효율적이다.

---

## 6. 왜 NDCG 값 자체가 이렇게 작나?

Full:

```text
Ridge       0.4776
LambdaRank  0.5958
```

그런데 K=1에서는:

```text
Ridge       0.0626
LambdaRank  0.0707
```

이다.

이걸 보고 모델 성능이 떨어졌다고 해석하면 안 된다.

K=1에서는 query당 하나만 human relevance를 알고 나머지를 전부 unjudged → 0으로 처리했기 때문이다.

예를 들어 실제 ranking이:

```text
Rank 1 → relevant 3
Rank 2 → relevant 3
Rank 3 → relevant 2
Rank 4 → relevant 2
Rank 5 → relevant 1
...
```

이어도 랜덤으로 Rank 27 하나만 labeling했다면 top-5의 relevance는 전부 모르는 상태가 된다.

그래서 **partial NDCG 자체가 강하게 downward biased**되어 있다.

따라서 이 실험에서 중요한 것은:

> `partial NDCG ≈ full NDCG`인가?

보다는

> **partial labeling으로 모델 A와 B 중 어느 것이 좋은지 맞힐 수 있는가?**

이다.

---

## 7. 모델 간 성능 차이 추정도 실패

실제 NDCG@5 gap:

$$
0.5958-0.4776=\mathbf{0.1181}
$$

하지만 K=1 추정치는:

$$
0.0707-0.0626=\mathbf{0.0081}
$$

이다.

실제 차이의 약 **6.9% 정도밖에 관측하지 못한다.**

더 중요한 것은 1-label의 Monte Carlo 90% 범위:

```text
estimated delta p05 = -0.0886
estimated delta p95 = +0.1022
```

즉 같은 실험을 다시 하면:

```text
Ridge가 더 좋다고 나올 수도 있고
        ↕
LambdaRank가 더 좋다고 나올 수도 있음
```

정도가 아니라 **방향 자체가 상당히 불안정**하다.

K=10에서도:

```text
true delta     = +0.1181
estimated mean = +0.0247
p05            = -0.0364
p95            = +0.0855
```

여전히 일부 sampling에서는 Ridge가 이기는 것으로 판단한다.

---

# 8. 최종 결론

### ❌ Random result 1개/query labeling

현재 POINTREC 실험에서는 **평가용으로 쓰기 어렵다.**

32 queries × 1 label = **32개 annotation**만 사용하면 모델 선택 정확도가:

> **52.4%**

로 거의 random이다.

### ⚠️ Random 5~10개/query

어느 정도 좋아지지만:

* 5개: 67.5%
* 10개: 76.5%

여전히 안정적인 offline model selection 기준으로는 부족해 보인다.

### 핵심적으로 얻은 결과

> **문제는 “label 하나” 자체라기보다 “random한 하나”다.**

NDCG@5는 특히 **ranking 상단의 relevance**가 중요한 metric인데 corpus candidate에서 랜덤으로 하나를 뽑는 것은 대부분 평가에 중요한 top result를 보지 않는다.

그래서 다음 실험이 훨씬 중요해.

```text
random 1/query
vs
model top-1/query
vs
top-5 pool에서 1개/query
vs
Ridge/Lambda top 결과 union에서 1개/query
```

특히 **모델의 top-1만 query당 하나 labeling**하는 방식은 random 1-label과 결과가 크게 달라질 가능성이 있다. 네 원래 질문인 **“평가셋 하나만 labeling해도 되나?”**를 테스트하려면 다음은 `random 1`이 아니라 **top-1 judging 전략**을 비교하는 게 맞다.
