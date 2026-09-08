POINTREC Human Relevance Labeling 실험

1. Dataset

사용 데이터셋은 POINTREC (SIGIR 2021) 이다.

POINTREC은 실제 커뮤니티의 POI 추천 질의를 기반으로 만든 Point-of-Interest recommendation/search benchmark이며, 각 Query–POI pair에 사람이 직접 0~3 relevance label을 부여했다.

항목	개수
Dataset	POINTREC
Information Needs / Queries	112
Qrels	5,143
전체 POI corpus	약 695K
Feature와 join된 Query–POI pairs	5,108
Train queries	60
Validation queries	20
Test queries	32

실험에서 사용한 relevance 분포는 다음과 같다.

Human relevance	의미	개수
0	Not relevant	924
1	Somewhat relevant	1,348
2	Relevant	1,649
3	Highly relevant	1,187
	Total	5,108

⸻

2. 실제 데이터 예시

POINTREC의 query는 단순 "restaurant" 같은 keyword가 아니라 자연어로 표현된 실제 POI search intent에 가깝다.

예를 들어 다음과 같은 query가 존재한다.

Query:
"Rotterdam Sunday dinner, ideally not Asian"
Candidate POIs:
POI                         Human Rel
-------------------------------------
좋은 조건 일치 Restaurant       3
부분적으로 조건 일치              2
Supersauer (Korean)             1
관련 없는 POI                    0

여기서 단순히 Restaurant category가 맞는지만 보는 게 아니다.

Sunday dinner라는 목적뿐 아니라 not Asian이라는 negative constraint까지 고려해서 사람이 relevance를 판단한다.

또 다른 실제 failure example:

Query:
"Milwaukee mini golf"
POI:
River Falls Family Fun Center
Human relevance = 3
Rating          = 2.5

즉 별점이 낮더라도 query intent에는 매우 relevant할 수 있다.

반대로:

Query:
Spain / Portugal travel POI recommendation
POI:
Tapabento
Rating          = 5.0
Reviews         = 330
Human relevance = 0

인기 있고 별점이 높아도 해당 query에 맞지 않으면 relevance=0이다.

이것이 rating/popularity 같은 proxy와 human relevance ground truth의 핵심 차이다.

⸻

3. 세 가지 Human Label 실험

동일한 weighted-sum ranking model을 사용하고 human supervision의 양만 변경했다.

사용한 ranking function은:

Score =
w_b BM25+
w_c Category+
w_t TokenOverlap+
w_r Rating+
w_p Popularity

Weight는 0.05 단위로 움직이며 합이 1이 되도록 했다.

총:

\boxed{10,626\text{ weight combinations}}

을 exhaustive grid search했다.

⸻

Experiment 1 — Full Graded Relevance

Label

POINTREC의 원래 human relevance를 그대로 전부 사용한다.

Query Q1
POI A → 3
POI B → 3
POI C → 2
POI D → 1
POI E → 0
...

즉 단순 relevant/irrelevant가 아니라 relevance의 정도까지 학습에 사용한다.

Train에는:

60 queries
2,751 Query–POI pairs

가 존재한다.

Best weights

BM25        0.00
Category    0.90
Token       0.05
Rating      0.05
Popularity  0.00

Test

Metric	Result
NDCG@5	0.7597
NDCG@10	0.7728
PairAgree	0.7194

세 실험 중 가장 높은 NDCG를 기록했다.

⸻

4. Experiment 2 — Top-Grade Only

이번에는 0/1/2/3의 세밀한 차이를 버린다.

대신 각 query에서 가장 높은 relevance를 받은 POI들을 전부 positive로 만든다.

예를 들어 원래 데이터가:

POI A → 3
POI B → 3
POI C → 3
POI D → 2
POI E → 1
POI F → 0

였다면:

POI A → 1
POI B → 1
POI C → 1
POI D → 0
POI E → 0
POI F → 0

가 된다.

즉 Top-1 하나가 아니라 최고등급 POI를 모두 positive로 사용한다.

Label 수

Train 60 queries에서:

Positive : 628
나머지   : Negative

즉 query당 평균:

628/60 \approx \boxed{10.5\ positive/query}

정도를 확보한 셈이다.

Best weights

BM25        0.00
Category    0.80
Token       0.05
Rating      0.15
Popularity  0.00

Test

Metric	Result
NDCG@5	0.7411
NDCG@10	0.7270
PairAgree	0.6420

Full Graded와 비교하면:

0.7597 \rightarrow 0.7411

으로 생각보다 성능 감소가 작았다.

Full Graded 성능의 약:

\frac{0.7411}{0.7597}= \boxed{97.6\%}

를 유지했다.

⸻

5. Experiment 3 — Strict Top-1

마지막으로 query 하나당 positive를 정말 딱 하나만 준다.

Query Q1
POI A → 1   ← only positive
POI B → 0
POI C → 0
POI D → 0
...

따라서 dataset이:

Split	Queries	Pairs	Positive
Train	60	2,751	60
Valid	20	914	20
Test	32	1,443	32

가 된다.

즉 train에서 2,751개의 pair가 있는데 positive supervision은 고작 60개다.

Best weights

결과가 상당히 달라졌다.

BM25        0.00
Category    0.05
Token       0.00
Rating      0.80
Popularity  0.15

Full Graded에서는:

Category = 0.90
Rating   = 0.05

였는데 Strict Top-1에서는:

Category = 0.05
Rating   = 0.80

으로 완전히 뒤집혔다.

Test

Metric	Result
NDCG@5	0.5544
NDCG@10	0.5659
PairAgree	0.5378
TopOnly MRR	0.3734
Top Hit@1	0.1250

⸻

6. 세 실험 최종 비교

Labeling 방법	Train positive 정보량	NDCG@5	NDCG@10	PairAgree
🥇 Full Graded	2,751 pairs에 0–3 grade	0.7597	0.7728	0.7194
🥈 Top-Grade Only	628 positives	0.7411	0.7270	0.6420
Category Only	Human supervision 없음	—	0.7290	0.7526
🥉 Strict Top-1	60 positives	0.5544	0.5659	0.5378
BM25	Human supervision 없음	—	0.5145	0.5627

NDCG@5만 보면:

Full Graded       0.7597  ████████████████████
Top-Grade Only    0.7411  ███████████████████
Category Only     0.7290  ███████████████████
Strict Top-1      0.5544  ██████████████
BM25              0.5145  █████████████

⸻

7. 실제 Failure Example

Strict Top-1에서 가장 명확하게 문제가 드러났다.

높은 Rating인데 실제로는 Irrelevant

Query:
Spain / Portugal travel
POI:
Tapabento
Rating          = 5.0
Reviews         = 330
Human relevance = 0
Strict Top-1 Rank = 1
Human Rank        = 48.5

Rating은 낮지만 실제로는 Highly Relevant

Query:
Milwaukee mini golf
POI:
River Falls Family Fun Center
Rating          = 2.5
Human relevance = 3
Human Rank        = 1
Strict Top-1 Rank = 46

즉 sparse한 Top-1 supervision에서는 모델이 query relevance 대신 Rating이라는 쉬운 proxy를 잡아버렸다.

⸻

8. 결론

세 실험의 핵심은 단순히 “label이 많으면 좋다”보다 조금 더 구체적이다.

\boxed{\text{Fine-grained label보다 Positive Coverage가 먼저 중요}}

Strict Top-1
1 positive/query
        ↓
60 train positives
        ↓
NDCG@5 = 0.5544 ❌
Top-Grade Only
약 10.5 positives/query
        ↓
628 train positives
        ↓
NDCG@5 = 0.7411 ✓
Full Graded
전체 candidate에 0/1/2/3
        ↓
2,751 graded pairs
        ↓
NDCG@5 = 0.7597 ✓✓

따라서 이번 POINTREC 실험에서는 query당 하나의 정답 POI만 labeling하는 방식은 부족했다.

반면 최고 relevance 후보들을 여러 개 확보하면 0/1/2/3을 세밀하게 구분하지 않아도 Full Graded의 97.6% NDCG@5를 유지했다.

실무적 결론: labeling 비용을 줄여야 한다면 “query당 POI 하나만 정답 처리”하는 것보다, grade를 단순화하더라도 query당 여러 relevant POI를 확보하는 것이 훨씬 효과적이었다.

주의: Strict Top-1 실험에서 rel=3 후보가 여러 개인 경우 하나를 고르는 tie-break가 rating/review count를 사용했다면, Strict Top-1 결과의 Rating=0.80에는 인위적인 bias가 들어갈 수 있다. 따라서 이 실험을 보고서에 넣을 때는 “true human Top-1”이 아니라 “one-positive-per-query simulation”이라고 표현하는 게 정확하다.
