# 실제 OSM 기반 Geo Search Benchmark 결과

## 1. 실험 개요

실제 OpenStreetMap(OSM) POI 데이터를 이용해 지역 검색 방식별 성능을 비교했다.

### 데이터

* 전체 OSM feature: **24,423개**
* 사용 가능한 POI: **24,412개**

| Category   |  Count |
| ---------- | -----: |
| restaurant | 21,852 |
| cafe       |  2,067 |
| fast_food  |    485 |
| food_court |      8 |

검색 기준점으로부터 POI 거리 분포는 다음과 같다.

| Metric | Distance |
| ------ | -------: |
| Min    |     14 m |
| Median |  6.91 km |
| P90    |  8.68 km |
| P95    |  9.10 km |
| P99    | 10.29 km |
| Max    | 11.07 km |

Elasticsearch에 총 **24,412개 문서**를 인덱싱했고, `Top-K=20` 기준으로 평가했다.

---

# 2. 1차 Benchmark

비교 방식:

* `ES_NATIVE`

  * Elasticsearch native geo distance 검색
* `H3_INCREMENTAL`

  * H3 cell을 중심에서 ring 단위로 확장
* `HYBRID_H3_ES`

  * H3로 후보군을 제한한 뒤 Elasticsearch에서 검색/정렬

## 결과

| Method         |         P50 |         P95 |         P99 | Recall@20 | Candidates | Rings |
| -------------- | ----------: | ----------: | ----------: | --------: | ---------: | ----: |
| **ES_NATIVE**  | **4.33 ms** | **7.67 ms** | **9.51 ms** | **1.000** |          - |     - |
| H3_INCREMENTAL |    10.54 ms |    14.27 ms |    23.29 ms |     1.000 |        382 |     1 |
| HYBRID_H3_ES   |     6.71 ms |     9.38 ms |    16.08 ms |     1.000 |        382 |     1 |

### 결과 해석

세 방식 모두 `Recall@20 = 1.0`으로 검색 정확도 차이는 없었다.

하지만 latency에서는:

```text
ES_NATIVE
    ↓ 약 55% 느림
HYBRID_H3_ES
    ↓ 추가로 느림
H3_INCREMENTAL
```

순서가 명확했다.

특히 median latency 기준:

```text
ES Native       4.33 ms
Hybrid          6.71 ms
H3              10.54 ms
```

즉 현재 규모에서는 H3를 사용해서 후보를 382개까지 줄이는 것보다 Elasticsearch가 직접 geo index를 검색하는 것이 더 빨랐다.

---

# 3. Query Type별 Benchmark

검색 질의를 실제 서비스에서 발생할 수 있는 형태로 나눴다.

* `near_me`: 내 주변 음식점
* `point_anchor`: 강남역/잠실역/홍대입구역 등 특정 지점
* `landmark`: 롯데월드/코엑스
* `explicit_radius`: 강남역 1km
* `admin_area`: 송파구/강남구
* `neighborhood`: 역삼동

---

# 4. Near-me 검색

### `내 주변 음식점`

| Method                 |         P50 |         P95 | Recall@20 | Candidates | Queries |
| ---------------------- | ----------: | ----------: | --------: | ---------: | ------: |
| **ES_NATIVE_DISTANCE** | **2.73 ms** | **3.45 ms** |       1.0 |          - |   **1** |
| H3_ES_HYBRID           |     4.92 ms |     5.74 ms |       1.0 |        356 |       3 |
| H3_INCREMENTAL         |     6.65 ms |     8.27 ms |       1.0 |        356 |       2 |
| ES_ADAPTIVE_RADIUS     |     7.67 ms |     9.06 ms |       1.0 |        200 |       2 |

### 결론

`near_me`에서도 Elasticsearch native distance search가 가장 좋았다.

H3를 이용해 후보를 356개로 줄였지만, H3 계산 및 추가 query overhead 때문에 오히려 latency가 증가했다.

---

# 5. 특정 지점(Point Anchor) 검색

## 강남역 맛집

| Method                 |         P50 |
| ---------------------- | ----------: |
| **ES_NATIVE_DISTANCE** | **1.99 ms** |
| H3_ES_HYBRID           |     4.10 ms |
| ES_ADAPTIVE_RADIUS     |     5.57 ms |
| H3_INCREMENTAL         |     6.68 ms |

H3는 **ring 1**만으로 356개의 후보를 확보했다.

---

## 잠실역 맛집

| Method                 |         P50 | Rings |
| ---------------------- | ----------: | ----: |
| **ES_NATIVE_DISTANCE** | **1.64 ms** |     - |
| ES_ADAPTIVE_RADIUS     |     6.42 ms |     - |
| H3_ES_HYBRID           |     7.31 ms |     4 |
| H3_INCREMENTAL         |    10.06 ms |     4 |

잠실에서는 주변 POI density가 낮아 H3가 **4 ring**까지 확장되어야 했다.

결과적으로:

```text
H3 ring 증가
→ H3 cell 증가
→ Elasticsearch query 횟수 증가
→ latency 증가
```

현상이 확인됐다.

---

## 홍대입구역 카페

| Method                 |         P50 | Rings |
| ---------------------- | ----------: | ----: |
| **ES_NATIVE_DISTANCE** | **1.60 ms** |     - |
| H3_ES_HYBRID           |     5.83 ms |     3 |
| ES_ADAPTIVE_RADIUS     |     6.75 ms |     - |
| H3_INCREMENTAL         |     8.15 ms |     3 |

역시 native geo query가 압도적으로 빠르다.

---

# 6. Landmark 검색

## 롯데월드 맛집

| Method                 |         P50 | Rings |
| ---------------------- | ----------: | ----: |
| **ES_NATIVE_DISTANCE** | **1.66 ms** |     - |
| H3_ES_HYBRID           |     6.58 ms |     4 |
| ES_ADAPTIVE_RADIUS     |     6.64 ms |     - |
| H3_INCREMENTAL         |     9.10 ms |     4 |

---

## 코엑스 카페

가장 H3에 불리한 케이스였다.

| Method                 |         P50 |         P95 | Rings | Queries |
| ---------------------- | ----------: | ----------: | ----: | ------: |
| **ES_NATIVE_DISTANCE** | **1.61 ms** | **1.91 ms** |     - |       1 |
| ES_ADAPTIVE_RADIUS     |     9.63 ms |    10.78 ms |     - |       5 |
| H3_ES_HYBRID           |    13.19 ms |    15.00 ms |    10 |      12 |
| H3_INCREMENTAL         |    16.72 ms |    18.18 ms |    10 |      11 |

카페 density가 낮기 때문에 H3가 **10 ring**까지 확장됐다.

이는 incremental H3 방식의 가장 큰 약점을 보여준다.

```text
Sparse region

ring 1
  ↓ 후보 부족
ring 2
  ↓
ring 3
  ↓
...
ring 10
  ↓
충분한 candidate 확보
```

반면 Elasticsearch native geo search는 POI density와 관계없이 **단일 query**로 약 1.6 ms에 결과를 반환했다.

---

# 7. Explicit Radius 검색

## `강남역 1km 맛집`

| Method                 |         P50 | Recall |
| ---------------------- | ----------: | -----: |
| **ES_NATIVE_DISTANCE** | **1.46 ms** |    1.0 |
| H3_ES_HYBRID           |     3.64 ms |    1.0 |
| ES_ADAPTIVE_RADIUS     |     4.88 ms |    1.0 |
| H3_INCREMENTAL         |     6.42 ms |    1.0 |

명시적인 반경이 존재하는 경우에도 native geo query가 가장 빠르다.

따라서:

```text
"1km 이내 음식점"
"500m 근처 카페"
"3km 이내 주차장"
```

같은 질의는 H3 ring expansion을 사용할 이유가 특히 적다.

Elasticsearch의 `geo_distance` filter가 질의 의미와도 직접적으로 일치한다.

---

# 8. 행정구역 검색

행정구역 검색은 거리 검색이 아니라 polygon 기반으로 처리했다.

| Query  | Method            |         P50 |     P95 | Recall |
| ------ | ----------------- | ----------: | ------: | -----: |
| 송파구 맛집 | ES_NATIVE_POLYGON |     3.58 ms | 5.26 ms |    1.0 |
| 강남구 카페 | ES_NATIVE_POLYGON |     2.74 ms | 3.09 ms |    1.0 |
| 역삼동 맛집 | ES_NATIVE_POLYGON | **2.00 ms** | 2.43 ms |    1.0 |

행정구역 역시 Elasticsearch native polygon query만으로 충분히 빠른 성능을 보였다.

따라서 이런 질의:

```text
송파구 맛집
강남구 카페
역삼동 음식점
```

은 H3가 아니라 **행정구역 polygon / geo_shape 검색**으로 처리하는 것이 자연스럽다.

---

# 9. 전체 Query Type별 추천 방식

| Query Type      | Example    | Recommended                         |
| --------------- | ---------- | ----------------------------------- |
| Near-me         | 내 주변 음식점   | **ES geo_distance**                 |
| Point Anchor    | 강남역 맛집     | **ES geo_distance**                 |
| Landmark        | 롯데월드 맛집    | **ES geo_distance**                 |
| Explicit Radius | 강남역 1km 맛집 | **ES geo_distance + radius filter** |
| Admin Area      | 송파구 맛집     | **ES geo_shape / polygon**          |
| Neighborhood    | 역삼동 맛집     | **ES geo_shape / polygon**          |

이번 실험에서는 사실상 모든 질의 유형에서 Elasticsearch native geo functionality가 최선이었다.

---

# 10. H3가 느려진 이유

H3 자체가 느려서라기보다 **검색 pipeline이 복잡해지기 때문**이다.

## ES Native

```text
query
  ↓
Elasticsearch spatial index
  ↓
Top-K
```

대부분 **1 query**면 끝난다.

## H3 Incremental

```text
query location
  ↓
H3 cell 계산
  ↓
ring 0 검색
  ↓
candidate 부족?
  ↓
ring 1
  ↓
candidate 부족?
  ↓
ring 2 ...
  ↓
distance 계산 / sorting
  ↓
Top-K
```

density가 낮을수록 이 overhead가 증가한다.

실제 결과에서도:

```text
강남역
rings = 1
p50 = 6.68 ms

잠실역
rings = 4
p50 = 10.06 ms

코엑스
rings = 10
p50 = 16.72 ms
```

로 ring 수와 latency가 같이 증가하는 패턴이 나타났다.

---

# 11. Hybrid도 Native를 이기지 못한 이유

Hybrid는 pure H3보다 일관되게 빨랐다.

예:

```text
강남역
H3 incremental : 6.68 ms
Hybrid         : 4.10 ms

코엑스
H3 incremental : 16.72 ms
Hybrid         : 13.19 ms
```

즉 H3 candidate generation 이후 실제 ranking/search를 Elasticsearch에 맡기는 것은 개선 효과가 있다.

하지만 여전히:

```text
H3 candidate 생성
+
추가 ES query
```

라는 비용이 존재하기 때문에 native ES보다 느리다.

```text
ES Native     ≈ 1~3 ms
Hybrid        ≈ 4~13 ms
```

이번 데이터에서는 candidate pruning으로 얻는 이득보다 candidate generation overhead가 더 컸다.

---

# 12. Adaptive Radius도 효과가 없었던 이유

Adaptive radius 방식은 후보 수를 약 **200개**로 안정적으로 제한한다.

하지만 native distance query보다 더 느렸다.

예:

```text
강남역

Native
1 query
1.99 ms

Adaptive Radius
2 queries
5.57 ms
```

코엑스처럼 sparse한 영역에서는:

```text
5 queries
9.63 ms
```

까지 증가한다.

즉 **candidate count 감소보다 반복 query 비용이 더 컸다.**

---

# 13. 가장 중요한 관찰

## ① Candidate 수가 작다고 반드시 빠르지 않다

H3:

```text
24,412 docs
   ↓
356 candidates
```

로 검색 공간을 크게 줄였음에도 native ES보다 느렸다.

Elasticsearch의 geo index가 이미 공간 검색을 효율적으로 수행하기 때문에, application layer에서 수동으로 candidate를 줄이는 것이 반드시 이득이 아니다.

---

## ② Query 횟수가 latency에 매우 중요하다

결과에서 가장 명확한 패턴이다.

```text
ES native
1 query
≈ 1~3 ms

H3
2~11 queries
≈ 6~17 ms
```

즉 현재 규모에서는 **candidate count보다 round trip / query execution 횟수가 더 중요한 병목**이었다.

---

## ③ H3 Incremental은 density sensitivity가 크다

Dense area:

```text
강남역
ring = 1
```

Sparse area:

```text
코엑스 카페
ring = 10
```

같은 알고리즘인데 데이터 density에 따라 latency가 크게 달라진다.

따라서 tail latency 측면에서도 불리하다.

---

## ④ Elasticsearch native search는 density에 비교적 안정적이다

Native 결과:

```text
강남역      1.99 ms
잠실역      1.64 ms
홍대입구역   1.60 ms
롯데월드     1.66 ms
코엑스      1.61 ms
```

POI density가 크게 달라져도 latency 변화가 매우 작다.

서비스 운영 관점에서는 이게 상당히 중요한 장점이다.

---

# 14. 최종 결론

이번 실제 OSM 24K POI 실험에서는:

> **Elasticsearch native geo query가 가장 단순하면서도 가장 빠르고 정확했다.**

모든 방식의 `Recall@20`은 1.0이었으므로 품질 차이는 없었으며, 성능은 대체로:

```text
ES Native
    >
H3 + ES Hybrid
    >
Adaptive Radius
    >
H3 Incremental
```

순이었다.

따라서 현재 실험 결과만 놓고 보면 검색 architecture는 굳이 H3 기반으로 만들 필요가 없다.

```text
Query Understanding
        │
        ├── "내 주변 / 강남역 / 롯데월드"
        │        ↓
        │   ES geo_distance
        │
        ├── "1km 이내"
        │        ↓
        │   ES geo_distance + radius
        │
        └── "송파구 / 역삼동"
                 ↓
            ES geo_shape
```

정도가 가장 단순하고 합리적이다.

---

# 15. 단, H3가 불필요하다는 뜻은 아님

이번 실험으로 말할 수 있는 것은 정확히:

> **24K 규모의 단일 Elasticsearch index에서 Top-K 지역 검색을 하기 위해 application-level H3 ring expansion을 추가하는 것은 이점이 없었다.**

이다.

H3가 유리할 가능성이 있는 상황은 별도로 존재한다.

예를 들어:

* 수천만~수억 POI
* Elasticsearch 외부에서 spatial sharding이 필요한 경우
* cell별 pre-aggregation
* 지역별 cache key
* distributed routing
* 실시간 heatmap
* 공간 단위 batch processing
* offline feature aggregation

등에서는 H3 자체의 장점이 있을 수 있다.

즉 H3의 주요 역할을:

```text
Top-K geo search accelerator
```

로 보는 것보다는

```text
Spatial partitioning / aggregation / routing
```

으로 보는 것이 이번 결과와 더 잘 맞는다.

---

# 한 줄 결론

> **실제 OSM 데이터 기준으로 `내 주변`, `강남역`, `랜드마크`, `1km`, `행정구역` 모두 Elasticsearch native geo search가 가장 빨랐으며, H3 ring expansion은 candidate 수는 줄였지만 반복 query와 ring expansion overhead 때문에 오히려 느려졌다.**
