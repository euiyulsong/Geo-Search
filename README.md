# Elasticsearch Native vs H3 지역검색 실험 결과

## 1. 실험 비교 대상

지역 기반 후보 검색 방식 3가지를 비교했다.

* **ES_NATIVE**

  * Elasticsearch의 native geo index 사용
  * `geo_distance` / 거리 기반 검색
  * 기준 정답 역할
* **H3_INCREMENTAL**

  * H3 cell을 중심에서부터 ring 단위로 확장
  * 필요한 후보 수를 확보하면 탐색 종료
* **H3_ES_HYBRID**

  * H3로 coarse candidate를 먼저 제한
  * 이후 Elasticsearch geo 검색을 추가 적용

평가 지표:

* `p50 / p95 / p99 latency`
* `Recall@1 / @5 / @20 / @100 / @500`
* `candidate_recall`
* 탐색 candidate 수
* H3 ring 수

---

# 2. Overall

| Method         |         p50 |         p95 | Recall@20 | Recall@500 | Candidate Recall |
| -------------- | ----------: | ----------: | --------: | ---------: | ---------------: |
| **ES_NATIVE**  | **1.88 ms** | **2.12 ms** | **1.000** |  **1.000** |                - |
| H3_INCREMENTAL |     2.46 ms |     2.62 ms |     0.966 |      0.938 |            0.938 |
| H3_ES_HYBRID   |     4.10 ms |     4.50 ms |     0.966 |      0.938 |            0.938 |

전체적으로 **ES Native가 latency와 recall 모두 가장 좋았다.**
H3 Incremental은 Hybrid보다 빠르지만 ES Native보다 느리고, 전체 candidate recall도 약 **93.8%**에 그쳤다. Hybrid는 H3 후보생성 뒤 ES 작업까지 추가되면서 가장 느렸다.

### 핵심

```text
ES Native
→ 가장 빠름
→ Recall 100%

H3 Incremental
→ ES보다 약간 느림
→ Recall 약 94%

Hybrid
→ 가장 느림
→ H3와 동일한 Recall
```

따라서 **Elasticsearch를 이미 사용하는 환경에서는 H3를 추가하는 실익이 크지 않았다.**

---

# 3. Top-K 크기에 따른 결과

## K가 아주 작을 때는 H3가 빠름

### K = 1

| Method         |         p50 |  Recall@1 |
| -------------- | ----------: | --------: |
| ES_NATIVE      |     1.62 ms | **1.000** |
| H3_INCREMENTAL | **0.77 ms** |     0.889 |
| Hybrid         |     1.43 ms |     0.889 |

H3 Incremental은 약 **2배 빠르지만 Recall@1이 88.9%**로 떨어졌다.

### K = 5

```text
ES Native        1.69 ms / Recall 1.000
H3 Incremental   0.98 ms / Recall 0.887
Hybrid           1.71 ms / Recall 0.887
```

역시 H3가 latency 면에서는 빠르지만 약 11%의 candidate를 놓친다.

---

## K = 20

```text
ES Native        p50 1.88 ms / Recall@20 1.000
H3 Incremental   p50 1.36 ms / Recall@20 0.906
Hybrid           p50 2.27 ms / Recall@20 0.906
```

H3 Incremental은 ES보다 약 0.5ms 빠르지만 **Top-20 Recall이 약 90.6%**로 감소한다.

즉:

> 작은 K에서는 H3 Incremental이 latency를 줄일 수 있지만, Recall 손실을 감수해야 한다.

---

# 4. K가 커지면 H3가 급격히 불리해짐

### K = 100

```text
ES Native        2.65 ms
H3 Incremental   4.05 ms
Hybrid           6.11 ms
```

H3는 평균 11.5 ring까지 확장하고 약 108개 후보를 탐색한다. Recall@100은 약 92.3%다.

### K = 500

```text
ES Native         6.50 ms
H3 Incremental   24.93 ms
Hybrid           31.70 ms
```

H3는 약 **28.5 ring**, 약 **515 candidates**까지 확장해야 하며 Recall@500은 **78.97%**까지 떨어진다.

따라서:

```text
K 작음
→ H3가 latency 이점 가능

K 큼
→ ring 확장 비용 증가
→ ES Native가 훨씬 우세
```

---

# 5. 지역 Density별 결과

## Dense 지역

| Method         |         p50 |    Recall |
| -------------- | ----------: | --------: |
| ES_NATIVE      |     2.21 ms | **1.000** |
| H3_INCREMENTAL | **1.52 ms** |     0.953 |
| Hybrid         |     3.10 ms |     0.953 |

Dense 지역에서는 H3 Incremental이 평균 **2 ring** 정도면 후보를 확보할 수 있어서 ES보다 빠르다.

즉 강남역·홍대 같은 곳에서는:

```text
H3 ring 0
→ ring 1
→ ring 2
→ 후보 충분
```

이 가능하기 때문에 H3의 장점이 가장 잘 나타난다.

---

## Medium 지역

```text
ES Native       1.88 ms
H3 Incremental  2.31 ms
Hybrid          3.62 ms
```

H3는 평균 5 ring이 필요해지면서 이미 ES Native보다 느려진다.

---

## Sparse 지역

```text
ES Native       1.50 ms
H3 Incremental  5.34 ms
Hybrid          6.55 ms
```

H3는 평균 **15.5 ring**까지 확장해야 해서 오히려 매우 비효율적이다.

### 결론

```text
Dense
→ H3 Incremental 가능

Medium
→ ES Native 우세

Sparse
→ ES Native 압도적
```

**H3의 incremental expansion은 density가 낮아질수록 탐색해야 하는 ring 수가 빠르게 증가한다.**

---

# 6. Category 빈도별 결과

## Common category

```text
ES Native        2.05 ms
H3 Incremental   2.07 ms
Hybrid           3.45 ms
```

거의 비슷한 latency지만 ES Native는 Recall 1.0, H3는 약 0.936이다.

즉 restaurant/cafe 같은 흔한 카테고리에서는 H3 candidate가 빠르게 확보돼 상대적으로 괜찮다.

---

## Medium frequency

```text
ES Native        1.46 ms
H3 Incremental   5.04 ms
Hybrid           6.30 ms
```

H3는 약 13.5 ring을 탐색한다.

---

## Rare category

```text
ES Native        1.41 ms
H3 Incremental   6.30 ms
Hybrid           7.50 ms
```

평균 16 ring까지 확장해야 한다.

따라서:

```text
음식점 / 카페
→ H3가 어느 정도 경쟁 가능

희귀한 POI
→ H3 expansion 비효율
→ ES Native가 유리
```

---

# 7. 명시적 Radius 검색

이 결과가 가장 명확하다.

## 100m

```text
ES Native        1.33 ms
H3 Incremental   1.23 ms
Hybrid           2.15 ms
```

거의 차이가 없고 모두 Recall 1.0이다.

## 300m

```text
ES Native        1.43 ms
H3 Incremental   1.40 ms
Hybrid           2.50 ms
```

역시 거의 동일하다.

## 1km

```text
ES Native        1.49 ms
H3 Incremental   2.79 ms
Hybrid           4.37 ms
```

1km부터는 ES Native가 명확히 더 빠르다.

## 3km

```text
ES Native         2.04 ms
H3 Incremental    8.64 ms
Hybrid           11.97 ms
```

H3는 평균 16 ring이 필요하다.

## 10km

```text
ES Native         3.87 ms
H3 Incremental   52.86 ms
Hybrid           60.49 ms
```

H3는 평균 **51 ring**까지 확장된다.

따라서 명시적 radius query는:

> **그냥 Elasticsearch native `geo_distance`를 쓰는 것이 압도적으로 좋다.**

---

# 8. 최종 결론

## ES Native가 기본 선택

전체적으로:

```text
ES Native
- 가장 안정적인 latency
- Recall 100%
- density 영향 작음
- category rarity 영향 작음
- radius가 커져도 비교적 안정적
```

따라서 Elasticsearch를 이미 사용한다면:

```text
geo_point
+ geo_distance
+ geo_shape
```

를 기본 geo retrieval로 사용하는 것이 가장 합리적이다.

---

## H3 Incremental이 유리할 수 있는 경우

H3가 의미가 있는 조건은 상당히 제한적이었다.

```text
Dense area
AND
Common category
AND
Small Top-K
AND
Very local search
```

예:

```text
강남역 주변 음식점 Top 5
내 주변 카페 Top 10
```

이런 경우:

```text
H3 center
→ ring 1~2
→ 바로 후보 확보
```

가 가능해서 latency를 줄일 수 있다.

하지만 그 경우에도 **Recall 손실**이 존재했다.

---

## Hybrid는 현재 실험에서는 불필요

H3 + Elasticsearch Hybrid는:

```text
H3 candidate 생성 비용
+
ES geo query 비용
```

이 모두 발생하면서 대부분 가장 느렸다.

그리고 Recall도 H3 candidate generation에 의해 제한되므로:

```text
Hybrid Recall
≈ H3 Recall
```

이었다.

따라서 현재 형태의 Hybrid는:

> **ES Native에 비해 latency도 나쁘고 Recall도 낮으므로 사용할 이유가 거의 없다.**

---

# 9. Query Type별 추천

| Query type          | 예시             | 추천                  |
| ------------------- | -------------- | ------------------- |
| Explicit radius     | `1km 내 맛집`     | **ES geo_distance** |
| 주변 검색               | `내 주변 음식점`     | **ES Native 기본**    |
| Dense + Top-K 매우 작음 | `강남역 카페 Top 5` | H3 실험 가능            |
| Sparse 지역           | `시골역 주변 맛집`    | **ES Native**       |
| Rare category       | `내 주변 특정 시설`   | **ES Native**       |
| 큰 radius            | `10km 내 병원`    | **ES Native**       |
| 행정구역                | `송파구 맛집`       | **ES geo_shape**    |

---

# 10. 핵심 Takeaway

이번 실험에서 가장 중요한 결과는:

> **Elasticsearch의 native BKD 기반 geo search가 생각보다 매우 강하다.**

H3를 별도로 붙이면 항상 빨라질 것 같지만 실제로는:

```text
candidate가 가까이에 많이 존재
→ H3가 빠를 수 있음

candidate가 희소하거나
K/radius가 커짐
→ ring expansion 비용 폭증
→ ES Native가 더 빠름
```

이라는 결과가 나왔다.

따라서 production 설계는:

```text
Default
→ Elasticsearch native geo search

Optional optimization
→ dense + common category + small-K에 대해서만
   H3 routing 여부 추가 실험
```

정도가 가장 합리적이다.
