# POINTREC — Graded vs Binary Test Relevance 분석

## 1. 실험 목적

이번 실험은 **테스트셋 relevance를 굳이 1/2/3으로 세분화해서 labeling해야 하는가**를 확인하는 실험이다.

비교한 두 평가 방식은 다음과 같다.

| 원래 relevance | Graded 평가 | Binary 평가 |
| -----------: | --------: | --------: |
|            0 |         0 |         0 |
|            1 |         1 |         1 |
|            2 |         2 |         1 |
|            3 |         3 |         1 |

즉 Binary에서는 **relevant인지 아닌지만 labeling**한다고 가정한다.

테스트셋은 32 queries / 2,083 pairs이며, 원래 relevant POI는 총 **1,207개**이다.

---

## 2. 전체 결과

| Evaluation     | Model          |     NDCG@5 |    NDCG@10 |        MRR |
| -------------- | -------------- | ---------: | ---------: | ---------: |
| Graded 0/1/2/3 | Ridge          |     0.4776 |     0.5130 |     0.8646 |
| Graded 0/1/2/3 | **LambdaRank** | **0.5958** | **0.6228** | **0.9073** |
| Binary 0/1     | Ridge          |     0.8012 |     0.8159 |     0.8646 |
| Binary 0/1     | **LambdaRank** | **0.8911** | **0.8884** | **0.9073** |

가장 중요한 결과는 **평가 label을 0/1로 단순화해도 전체 모델 선택 결과가 바뀌지 않았다는 것**이다.

```text
Graded evaluation → LambdaRank 승
Binary evaluation → LambdaRank 승
```

NDCG@5, NDCG@10, MRR 모두 동일하다.

---

## 3. NDCG@5

### Graded

```text
Ridge       0.4776
LambdaRank  0.5958
Gap        +0.1181
```

### Binary

```text
Ridge       0.8012
LambdaRank  0.8911
Gap        +0.0900
```

Binary로 바꿔도 **LambdaRank가 명확하게 우세**하다.

다만 모델 간 차이는

$$
0.1181 \rightarrow 0.0900
$$

으로 약간 감소했다.

즉 1/2/3의 relevance 강도에는 LambdaRank의 장점이 일부 포함되어 있다. 그 정보를 제거하면 LambdaRank의 advantage가 조금 줄지만 **모델 선택 자체는 그대로 유지된다.**

---

## 4. NDCG 값이 왜 0.48 → 0.80처럼 크게 올라갔나?

이건 binary가 더 좋은 평가 방법이라는 뜻이 아니다.

Graded NDCG에서는 예를 들어:

```text
정답:
3 > 3 > 2 > 2 > 1

모델:
1 > 2 > 3 > 3 > 2
```

라면 모두 relevant POI를 찾았더라도 **3짜리를 위에 놓지 못했기 때문에 penalty**를 받는다.

반면 Binary에서는:

```text
3 → 1
2 → 1
1 → 1
```

이므로 위 결과가 전부:

```text
1 > 1 > 1 > 1 > 1
```

이 된다.

따라서 binary NDCG는 사실상:

> **상위권에 relevant POI가 들어왔는가?**

를 강하게 보는 반면 graded NDCG는:

> **상위권에 relevant POI가 들어왔으며, 그중에서도 더 relevant한 POI를 더 위에 올렸는가?**

까지 평가한다.

그래서 binary NDCG가 훨씬 높아진 것이다.

---

## 5. MRR이 완전히 동일한 이유

결과가:

```text
                Graded    Binary
Ridge           0.8646    0.8646
LambdaRank      0.9073    0.9073
```

로 정확히 같다.

정상적인 결과다.

MRR은 첫 relevant result의 **relevance가 1인지 2인지 3인지 관심이 없다.**

예를 들어:

```text
Rank 1: rel=0
Rank 2: rel=3
```

이면 reciprocal rank는 `1/2`.

```text
Rank 1: rel=0
Rank 2: rel=1
```

이어도 역시 `1/2`.

따라서 `1/2/3 → 1`로 바꿔도 MRR은 변하지 않는다.

---

# 6. 다만 Query-level에서는 차이가 꽤 큼

여기는 중요한 부분이다.

| Metric  | Query별 모델 우열 유지율 |
| ------- | ---------------: |
| NDCG@5  |       **56.25%** |
| NDCG@10 |       **62.50%** |
| MRR     |         **100%** |

전체 평균에서는:

```text
Graded → LambdaRank 승
Binary → LambdaRank 승
```

으로 동일하다.

하지만 개별 query를 보면 NDCG@5에서 **56.25%만 Ridge/LambdaRank의 우열이 동일**하다.

즉 binary와 graded가 완전히 같은 평가 방법은 아니다.

예를 들어 어떤 query에서:

```text
Ridge:
3, 3, 2, 0, 0

Lambda:
1, 1, 1, 1, 1
```

이라면 graded와 binary가 상당히 다른 판단을 내릴 수 있다.

Binary에서는 Lambda가 relevant POI를 많이 찾은 것이 매우 좋게 평가되지만, graded에서는 Ridge가 높은 relevance POI를 먼저 가져온 것도 중요하기 때문이다.

---

# 7. 그래서 "1/2/3 labeling 안 해도 되나?"

## 모델 선택만 목적이라면 → 가능성이 꽤 있음

이번 데이터에서는:

```text
                Graded winner    Binary winner
NDCG@5          LambdaRank       LambdaRank
NDCG@10         LambdaRank       LambdaRank
MRR             LambdaRank       LambdaRank
```

**모든 aggregate metric에서 모델 선택이 동일했다.**

따라서 목적이 단순히

> "새 모델 A/B 중 전체적으로 어떤 모델이 더 좋은가?"

라면 이번 결과는 **0/1 labeling만으로도 충분할 가능성**을 지지한다.

특히 annotation 관점에서는 사람이

```text
0 = irrelevant
1 = relevant
```

만 판단하면 되기 때문에 1/2/3의 경계를 정의하고 일관되게 평가하는 부담을 줄일 수 있다.

---

## Fine-grained ranking quality가 중요하면 → 1/2/3 유지 필요

반대로 제품 요구사항이:

> "괜찮은 장소와 정말 좋은 장소를 구분해서 정말 좋은 결과를 위에 올려야 한다."

라면 binary로 바꾸면 안 된다.

Binary에서는:

```text
약간 relevant = 1
매우 relevant = 1
완벽한 결과   = 1
```

이 되어 **relevance intensity 정보 자체를 버리기 때문**이다.

Query-level agreement가 NDCG@5에서 56.25%밖에 안 된다는 것도 이 차이가 실제로 상당함을 보여준다.

---

# 8. 이번 실험의 결론

> **POINTREC 테스트셋에서 relevance 1/2/3을 모두 1로 collapse해도 Ridge와 LambdaRank의 전체 모델 우열은 유지되었다.**

특히 NDCG@5에서:

$$
\text{Graded gap}=+0.1181
$$

$$
\text{Binary gap}=+0.0900
$$

으로 둘 다 LambdaRank 우세였다.

따라서 **aggregate model selection이 목적이라면 binary relevance labeling으로 평가 비용을 낮출 가능성이 있다.**

다만 query-level winner agreement는 NDCG@5 **56.25%**, NDCG@10 **62.5%**에 불과하므로,

> **binary relevance가 graded relevance를 완전히 대체한다고 볼 수는 없다.**

정리하면 **“모델 전체 A/B 평가에는 0/1도 가능해 보임, relevance 품질의 미세한 차이까지 평가하려면 0/1/2/3 필요”**가 이번 결과에서 가장 적절한 결론이야.

그리고 이 결론을 더 강하게 만들려면 **Ridge vs Lambda 한 쌍만 볼 게 아니라 여러 ranker/checkpoint를 넣고, graded 평가의 모델 순위와 binary 평가의 모델 순위 간 Spearman/Kendall correlation을 측정하는 실험**이 다음 단계로 가장 좋아.
