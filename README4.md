# POINTREC Ridge Regression vs LambdaRank 분석

## 1. 실험 설정

POINTREC 전체 **112 queries / 695,035 POIs**를 사용했고, 기존 qrels 5,143개에 각 query당 **전체 corpus에서 uniform random negative 20개**를 추가했다. 총 2,240개의 random negative가 추가되어 7,348개의 pair가 최종 feature dataset에 포함되었다. Train/Valid/Test는 각각 **60 / 20 / 32 queries**이며 동일 split을 Ridge와 LambdaRank가 공유한다.  

평가는 학습 label과 무관하게 원래 `human_rel ∈ {0,1,2,3}`를 사용한다. 비교한 supervision은 다음 세 가지다.

| Supervision   | 학습 label                             |
| ------------- | ------------------------------------ |
| `full_graded` | 원래 relevance 0/1/2/3                 |
| `top_grade`   | 최고 relevance 문서들은 1, 나머지는 0          |
| `strict_top1` | query당 최고 relevance 문서 하나만 1, 나머지는 0 |

---

## 2. 최종 결과

| Model          | Supervision  |     NDCG@5 |    NDCG@10 |        MRR |  PairAgree |
| -------------- | ------------ | ---------: | ---------: | ---------: | ---------: |
| Ridge          | full graded  |     0.5819 |     0.6096 |     0.4651 |     0.7629 |
| **LambdaRank** | full graded  | **0.6947** | **0.7170** | **0.6089** | **0.8034** |
| Ridge          | top grade    |     0.6637 |     0.6708 | **0.5787** | **0.7875** |
| LambdaRank     | top grade    | **0.6778** | **0.7040** |     0.5276 |     0.7675 |
| Ridge          | strict top-1 |     0.6021 |     0.6266 |     0.4800 |     0.7509 |
| **LambdaRank** | strict top-1 | **0.7319** | **0.7201** | **0.7109** |     0.7291 |

전체 최고 NDCG@5는 **LambdaRank + strict_top1 = 0.7319**다. 

---

## 3. 가장 중요한 결과: LambdaRank가 Ridge보다 ranking supervision을 훨씬 잘 활용

NDCG@5 차이는 다음과 같다.

```text
full_graded : LambdaRank - Ridge = +0.1128
top_grade   : LambdaRank - Ridge = +0.0141
strict_top1 : LambdaRank - Ridge = +0.1298
```

특히 `strict_top1`에서 차이가 가장 크다. 

이는 두 모델의 objective 차이와 잘 맞는다.

```text
Ridge
feature → absolute relevance target 예측

LambdaRank
query 내부 document ordering 직접 최적화
```

검색에서는 relevance 값 자체를 정확하게 회귀하는 것보다 **같은 query 안에서 positive를 negative보다 위에 올리는 것**이 핵심이므로 LambdaRank의 objective가 평가 지표와 더 직접적으로 정렬되어 있다.

---

## 4. 의외로 full graded label이 항상 좋은 것은 아님

상당히 흥미로운 결과다.

### Ridge

```text
full graded    0.5819
top grade      0.6637
strict top-1   0.6021
```

오히려 `top_grade`가 full graded보다 **+0.0818 NDCG@5** 높다.

### LambdaRank

```text
full graded    0.6947
top grade      0.6778
strict top-1   0.7319
```

여기서는 `strict_top1`이 full graded보다 **+0.0372** 높다. 실제 retention 계산에서도 strict_top1이 full의 105.35%를 기록했다. 

즉 이 실험에서는

> **더 세밀한 0/1/2/3 relevance supervision이 반드시 더 높은 top-ranking quality로 이어지지 않았다.**

오히려 최고 문서를 명확하게 구분하는 binary supervision이 NDCG@5에서 경쟁력이 있다.

다만 **한 seed의 차이만으로 strict_top1이 full보다 우월하다고 결론 내리면 안 된다.** CI도 상당히 겹친다.

---

## 5. Random negative 추가 효과를 볼 때 주의할 점

random negative 추가 후 relevance=0 데이터가 총 **3,164개**까지 증가했다. 원래 graded positive는 그대로 1: 1,348 / 2: 1,649 / 3: 1,187이다. 

특히 strict_top1 train은:

```text
positive = 60
negative = 3,891
```

로 극단적인 **1 : 64.9** imbalance다. 

그런데도 LambdaRank NDCG@5가 0.7319다.

이건 꽤 의미가 있다. LambdaRank는 단순히 전체 0 label의 비율을 맞추는 게 아니라 **query group 내부에서 positive가 negative보다 위로 가도록 학습**하기 때문에 이런 binary + highly imbalanced supervision에서도 상당히 잘 작동한 것으로 볼 수 있다.

반대로 Ridge에서는 3,891개의 0 target이 loss를 지배하기 쉬워서 strict-top1의 coefficient가 전체적으로 매우 작아졌다. 

---

## 6. Feature 관점에서도 category가 압도적으로 중요

Full-graded LambdaRank gain importance:

```text
category_score      60.15%
token_overlap       26.30%
bm25                 9.13%
rating_score         2.70%
popularity_score     1.72%
```

즉 **category + token overlap만 약 86%**의 gain을 차지한다. 

Top-grade에서도 category가 57.9%로 가장 중요하다. 반면 strict-top1에서는:

```text
category     33.9%
BM25         26.3%
token        19.1%
popularity   16.0%
rating        4.7%
```

로 feature 사용이 훨씬 분산된다. 

이것도 흥미롭다. **graded relevance를 맞출 때는 category signal에 크게 의존하지만, 하나의 최고 POI를 찾아야 할 때는 lexical relevance와 popularity 같은 추가 신호가 더 중요해진다.**

---

## 7. Baseline과 비교하면 더 명확함

Baseline은:

```text
Category-only NDCG@5 = 0.6730
BM25-only     NDCG@5 = 0.4533
```

이다. 

따라서:

```text
Category only              0.6730
LambdaRank top-grade       0.6778
LambdaRank full-graded     0.6947
LambdaRank strict-top1     0.7319
```

LambdaRank strict-top1은 강력한 category-only baseline보다 **+0.0589 absolute**, 약 **+8.8% relative** 향상이다.

반대로 Ridge full-graded는 **0.5819로 category-only보다 오히려 낮다.**

따라서 이 데이터에서는 단순 pointwise regression보다는 **ranking objective가 feature combination을 학습하는 데 훨씬 적합하다**는 증거가 강하다.

---

## 8. 결론

이번 결과에서 가장 중요한 메시지는 다음과 같다.

> **POINTREC의 소규모 query 환경에서도 corpus random negatives를 추가한 LambdaRank는 pointwise Ridge Regression보다 일관되게 높은 NDCG@5를 보였다. 특히 query당 하나의 positive만 사용하는 strict-top1 supervision에서도 NDCG@5 0.7319로 전체 실험 중 최고 성능을 기록했다. 이는 local search ranking에서 세밀한 graded relevance 값을 회귀하는 것보다 query 내부의 positive-negative ordering을 직접 학습하는 ranking objective가 더 효과적일 수 있음을 보여준다.**

다만 다음 실험에서는 **`strict_top1` seed를 10~20개 반복**하는 게 가장 중요하다. 현재 strict-top1의 positive 선택 자체가 random이므로, `0.7319 > 0.6947`을 “1개 positive만으로 full graded보다 좋다”고 주장하려면 multi-seed 평균 ± 표준편차와 paired significance가 필요하다.

그리고 한 가지 더 중요한 실험은 **`1 positive + random corpus negatives만` 남기고 나머지 judged POI를 학습에서 제거하는 것**이다. 그 결과까지 좋으면 정말로 “query당 positive annotation 하나 + unlabeled corpus만으로도 강한 ranker를 학습할 수 있다”는 훨씬 깔끔한 결론을 낼 수 있다.
