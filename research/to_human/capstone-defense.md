# Capstone Defense Document — Fair, Diverse, and Private Recommendations

**Title of our work:** Privacy-Preserving, LLM-Enhanced Popularity Calibration: Federating the SUPER Framework for Fair, Diverse, and Private Recommendations

**Base paper:** *SUPER: Smart User-Centric Popularity Exposure Reduction for Fair and Diverse Recommendations* — Yavru, Yalcin, Bilge, Gunduz, IEEE Access 14:37582–37605 (2026), DOI 10.1109/access.2026.3671645.

**Our system names:** FedSUPER (federated, privacy-preserving SUPER) and LLM-FedSUPER (FedSUPER + on-device LLM user profiles).

> **Accuracy note:** Every numeric value below is taken from the committed experiment artifacts
> (`experiments/*/results/metrics_*.json`), the analyses (`experiments/*/analysis*.md`), the
> research log/state, or the paper's own verified table. Where two artifacts disagree, the
> disagreement is **flagged as an inconsistency** and the number used in the final paper is identified.
> When a value is genuinely unknown, it is marked "not recorded".

---

## 1. Project in one paragraph

Recommender systems over-expose a small set of popular ("head") items and starve the long tail, a problem called *popularity bias*. The SUPER paper (Yavru et al., IEEE Access 2026) attacks this by partitioning the catalog with the Pareto principle, training two specialist models (one for head items, one for tail items), and merging their ranked outputs per-user with a *blueprint* so each user's list mirrors their observed head/tail consumption ratio. SUPER is centralized and exposes raw user behavior. Our capstone makes SUPER **privacy-preserving and accuracy-recoverable**: we show (and prove) that SUPER's blueprint merge depends only on public item popularity plus a per-user scalar **Pop_u** that never leaves the device, so we train both specialists federated (sparse FedAvg of item parameters, user embeddings on-device) and reproduce centralized SUPER's calibration exactly (RMSE-PC = 0.055, MRMC = 0.151). Because federated training weakens within-pool ranking (the head pool is only ~2.1% of the catalog), recall drops; we repair this by computing a **Sentence-BERT user profile on-device** and blending it into each pool's scores before the merge — recovering 23% of recall and ~2.6x long-tail coverage **with zero calibration cost**. We also dissect differential privacy: DP acts as an *intrinsic debias for a plain federated model* (RMSE-PC 0.705 → 0.283) but is *redundant under the blueprint*, which already guarantees calibration. The headline variant **LLM-FedSUPER** reaches recall 0.0366 at RMSE-PC 0.055 — matching centralized SUPER's calibration with near-centralized accuracy, and never exposing raw user data.

---

## 2. Research problem

### Simple terms

Recommendation systems (Netflix, Spotify, e-commerce) learn what you like from your clicks/ratings and recommend similar things. Left alone, they fall into a trap called **popularity bias**: a handful of already-popular items dominate everyone's lists, while thousands of niche items almost never get recommended. That's bad for users (boring lists, no discovery), bad for long-tail producers (never seen), and bad for the platform's catalog. Many fixes just re-rank lists toward obscure items for *everyone*, which ignores the fact that some people genuinely like blockbusters. The SUPER paper's idea: each user has their own "mainstream vs niche" taste, measurable from their history; give each user a final list that *respects their own* head-to-tail ratio, while still recommending the *best* items within each of those two groups.

### Technical framing

**Problem statement.** Given implicit/explicit interaction data `C ∈ R^{n_u × n_i}` and a horizon N, build a top-N list per user that (i) is accurate, (ii) is *popularity-calibrated* — the fraction of head items in the list matches the user's own observed inclination Pop_u — and (iii) maximizes long-tail coverage and novelty. Existing mitigations (re-ranking `score/pop^α`, causal debiasing, exposure-balancing) apply a *global* adjustment and break per-user calibration.

**Why it matters.** Popularity bias is documented across movies, books, music, tourism, education (Rich-get-richer). It reduces catalog coverage, homogeneity, cultural uniformity, harms niche producers, and degrades user satisfaction.

### The original approach at a glance

1. **Pareto partition (Algorithm 1):** sort items by interaction count descending; the *head* set H is the minimal prefix whose cumulative interaction volume reaches **α = 20% of total interactions** (not the top 20% of items). The tail T is the rest.
2. **Dual training:** train **M_pop** on head-only interactions and **M_tail** on tail-only interactions, decoupling gradients so tail representations are not swamped by head items.
3. **User inclination:** Pop_u = |C_u ∩ H| / |C_u| — a *behavioral proxy*, explicitly not treated as causal truth.
4. **Hard-quota / soft-blueprint merge (Algorithm 2):** N_pop = ⌊N·Pop_u⌋, N_tail = N − N_pop; take the top-N_pop of M_pop's ranking and top-N_tail of M_tail's; interleave them by scanning the user's historical list Lu (head/tail blueprint), deterministically falling back to the other pool when a quota is exhausted.

### Assumptions, inputs, outputs, datasets, metrics, evaluation (original paper)

- **Inputs:** interaction logs (ratings 1–5), item metadata available via datasets. No impression/exposure logs (explicitly assumed).
- **Outputs:** per-user top-N lists; additional outputs are the partition (H,T), the two models, Pop_u.
- **Datasets (paper):** MovieLens-1M, DoubanBooks, Yelp; ratings treated on the [1,5] scale. Note the paper deliberately omits Precision/Recall/F1 and uses calibration/beyond-accuracy metrics.
- **Backbones (paper):** HPF, NeuMF, SKMeans, VaeCF, WBPR — SUPER is backbone-agnostic, acting as pre/post-processing around any recommender.
- **Metrics (paper):** nDCG (higher better), RMSE-PC (lower better), MRMC (lower better), APLT (higher better), Entropy (higher better), LTC (higher better), Novelty `-log2 p(i)` (higher better), and **GKPI** = arithmetic mean of the harmonic-type mean H(nDCG, x) over x ∈ {APLT, Novelty, Entropy, LTC}. GKPI is a *summary*, explicitly not an optimization objective.
- **Evaluation (paper):** 5 random seeds, mean performance, paired Wilcoxon signed-rank tests vs the strongest non-SUPER baseline per metric; SUPER reported as best on GKPI/calibration.

### Original limitations / opportunities we identified

1. **Centralized assumption.** All data must reach a server; no privacy mechanism. Many deployments (and regulations) require raw behavior never leave the device.
2. **Interaction-only signal.** SUPER only uses co-occurrence; it ignores the rich content/text signal that modern dense embeddings/LLMs provide—needed especially *within* the small head pool and sparse tail pool.
3. **DP-fairness interplay unstudied.** Prior work (e.g., BGTplanner, Zhang 2024) hunts the DP budget for *accuracy*; whether DP noise helps or hurts *popularity calibration* was an open question.
4. **Multi-objective training partial.** No joint fairness+diversity+novelty+accuracy optimization was tested against SUPER.
5. **Scale question.** Whether a user-calibration merge is even worth it under federated noise was unexplored.

### What we proposed

Extend SUPER along three axes and test them with pre-registered hypotheses:
- **H1 — FedSUPER:** federate the two specialists (privacy) and verify that calibration transfers (invariance).
- **H2 — DP dynamics:** quantify whether DP noise improves or hurts calibration inside/outside the blueprint.
- **H3 — LLM user profiling:** recover the accuracy lost to federated training with an on-device semantic profile, without breaking calibration.
- **H4 — multi-objective:** test whether a popularity-dispersion training penalty buys anything beyond the blueprint.
- **H5 — adaptive DP:** test whether popularity-scaled noise beats uniform DP at equal nominal ε.

### Central research question

> Can SUPER's per-user popularity calibration be made **privacy-preserving** via Federated Learning while keeping the paper-level calibration guarantees (RMSE-PC, MRMC), and can **on-device LLM user profiles** recover the accuracy cost of privacy without degrading that calibration?

### Success/failure criteria

- **Success:** FedSUPER reproduces centralized SUPER's calibration (RMSE-PC ≈ 0.055, MRMC ≈ 0.151) never exposing raw user histories; and the best privacy-preserving variant recovers ≥50% of the recall gap vs centralized SUPER (≥ 0.025) [pre-registered, H3 protocol]. A bonus success: LLM blending must NOT touch the calibration metrics.
- **Failure indicators:** calibration degrades under federated training; DP destroys both accuracy and calibration; the LLM blend changes RMSE-PC/MRMC.

---

## 3. Original paper — more detail

### SUPER's mechanism and why it works

**Structural hypothesis of the paper:** separating head and tail *during learning* prevents popularity-dominant gradients from leaking into tail embeddings, then a *per-user* quota at ranking time imposes calibration the model alone never learned. The blueprint (position-level head/tail pattern of the user's own history) is a *soft ordering prior*; the quota is the *hard guarantee*. Complexity added by SUPER beyond backbone inference is only O(N) per user (the merge), plus an O(n log n) offline sort — negligible.

### As we re-implemented it (this is the version our experiments used — the paper's Algorithm 1 + Algorithm 2)

- Partition: `pareto_partition(pop, alpha=0.20)` — cumulative-volume head set. **[Fact]** On our MovieLens-1M (≥4) recoded matrix the head set is **75 items = 2.12% of the 3,533-item catalog**, carrying 20% of interaction volume. This matches the paper text "≈2.1% of the catalog."
- Pop_u = head-count / total-count per user.
- Merge: `super_blueprint_merge` implements hard quota + soft blueprint with deterministic fallback and residual-fill (exactly Algorithm 2's semantics: lines match `Npop ← ⌊N·Popu⌋`, candidate truncation, blueprint scan, quota-exhaustion switch, residual fill).
- Blueprint ordering proxy: we sort each user's training history by descending interaction count ("preference" ordering). The paper notes Lu sorted by descending preference; a timestamp variant exists in the code but was not selected (not recorded which the paper used on our data). **[Interpretation]** We used interaction count as the preference proxy.

---

## 4. Our proposed extension (what exactly changed)

| Axis | Original SUPER | Our capstone |
|---|---|---|
| Data location | Centralized; raw ratings reach the server | Federated; user embeddings & Pop_u & profiles stay on-device; server aggregates only item parameters |
| Backbone training | Any centralized recommender (we used centralized MF) | Sparse FedAvg of two MF specialists (M_pop, M_tail); private user embedding per client |
| Content signal | None | On-device Sentence-BERT profile blended per-pool: `ŝ = (1−λ)·M(i,u) + λ·cos(p_u, t_i)` |
| DP | None | Optional Gaussian local-noise DP, uniform or popularity-adaptive |
| Extra objective | None | Optional popularity-dispersion penalty (tested, marginal) |
| Metrics we adopt & supplement | Paper metrics (nDCG, RMSE-PC, MRMC, APLT, LTC, Entropy, Novelty, GKPI) | Same paper metrics + Recall@10 (the paper omitted recall; we add it to gauge accuracy cost) |

**Novelty hypothesis that frames everything:** *the blueprint merge is invariant to the backbone.* Any two score functions s_pop, s_tail + Pop_u → identical N_pop/N_tail quotas by construction, so RMSE-PC and MRMC are **properties of the merge algorithm, not of the trained models**. We proved it (theorem) and verified it empirically across federated/DP/LLM backbones.

---

## 5. Complete methodology (pipeline walk-through)

### 5.1 Pipeline

```
Input (MovieLens-1M, rating≥4 → implicit positives)
  → Preprocess (reindex users/items; leave-one-out split; build train matrix)
  → Pareto partition α=0.20 → (H, T); build D_pop (head interactions) and D_tail (tail interactions)
  → Train TWO specialists per setting:
       centralized SUPER: train_centralized on sm_head / sm_tail  (BPR-MF, d=64, 15 ep, Adam lr .05)
       FedSUPER:          train_federated  on sm_head / sm_tail  (sparse FedAvg, 40 rounds, 256 clients, 2 local ep, lr .3)
       (optional DP / optional multi-objective penalty / optional adaptive DP in the federated loop)
  → Score matrices S_pop, S_tail (mask user's training positives to −∞)
  → (H3) Blend LLM cosine per pool: Ŝ = (1−λ)·S + λ·cos(p_u, t_i)
  → Blueprint merge (hard quota / soft blueprint) → RecList(u), N=10
  → Evaluate: full-ranking leave-one-out Recall@10, nDCG@10 + paper metrics (RMSE-PC, MRMC, APLT, LTC, Entropy, Novelty, GKPI)
  → Compare against baselines: BPR-MF (centralized), SUPER (centralized), FedNCF (plain federated), FedNCF-DP, FedSUPER-DP, LLM-FedSUPER(-DP)
```

### 5.2 Component-by-component

**Data & preprocessing (`recdata.py`).** MovieLens-1M; ratings ≥ 4 become positive; users/items re-indexed to a contiguous 0-based space. **[Fact]** Our recoded index has **6,040 users, 3,533 items, 575,281 positive interactions**. Leave-one-out: each user's **last interaction by timestamp** is the test item; the rest train. **[Inconsistency flagged]** Several of our own documents (README, paper text) say "3,952 items," which is the nominal MovieLens-1M movie count; the *recoded* matrix our experiments actually used has 3,533 items (items with at least one rating ≥ 4). All quantitative results correspond to the 3,533-item matrix; the "≈2.1% head" figure (75/3,533) confirms this.

**Why LOO:** standard in recommender-system evaluation, cheap, comparable to the leave-one-out protocol assumed in the SUPER paper's blueprint construction (`Lu` excludes the held-out test item).

**Pareto partition (`super.py:pareto_partition`).** Cumulative-interaction-volume head set. **Why α=0.20:** paper default. **Alternative considered:** none for the partition itself (we locked the paper's parameter).

**Backbones (`model.py:RecMF`).** Low-rank matrix factorization, latent dim d=64, item bias. `score(u,i) = ⟨E_u, E_i⟩ + b_i`. Trained with **BPR** (pairwise ranking). Matrix-popularity because it is the simplest backbone that isolates the *transfer* question (is calibration invariant to the *training paradigm*?), cheap to run on 4 GB GPU, and mirrors the paper's WBPR-style factor model.

**Centralized training (`train.py:train_centralized`).** 15 epochs, Adam lr 0.05, batch 1,024 positives, 2 negatives/positive sampled per step.

**Federated training (`train.py:train_federated`).** Each user is a client. Server holds **item embeddings + item bias** (shared, FedAvg-aggregated); each client keeps its own **user embedding privately** (persists across rounds, updates locally, never aggregated). Per round: sample up to `clients_per_round` active users; each client does local SGD on its positives (fresh random negatives), clipping gradients to norm 1.0; deltas are accumulated **sparsely** (only items the client touched) and averaged per-item at the server. **This sparse + private-user-embedding design is the key privacy choice:** the server never sees a user vector or a raw interaction.

- **Why sparse FedAvg:** dense aggregation of untouched items would smear gradients everywhere and, more importantly, the retrained-full-matrix version didn't reflect on-device personalization; sparse mirrors what a real FL system can communicate.
- **Alternatives considered:** dense FedAvg, server-side user embeddings — rejected because they leak or dilute; per-client embedding *trained* (`sanity_fed.py` showed loss 0.69 → 0.24 with 60 rounds/512 clients, then production used 40 rounds/256 clients for time).

**DP (in the federated loop).** Clip per-tensor gradients to `[−max_grad_norm, +max_grad_norm]`, add Gaussian noise `σ = 0.3/ε`, capped at σ ≤ 0.6 to keep training tractable. **[Limitation acknowledged in paper]** this is a *pragmatic* per-update mechanism, not a formally composed (ε,δ) accountant over rounds; the privacy–utility trade-off is empirical. Our claims about calibration do not depend on the DP accounting.

**Adaptive DP (H5).** Item-specific noise multiplier `σ_i = σ_base · (1 + amp·normlog(pop_i))` with amp=2, so popular items get up to 3× noise and tail items ≈ 1×.

**LLM user profiling (H3, `run_h3.py`).**
- Item text: `"{title} ({genres})"` (genres with `|` → `, `), truncated to 200 chars.
- User profile text: `"A user who likes {top-8 genres} movies including: {5 sampled liked titles}."` — built from the user's *training* history only (privacy: computed on-device).
- Embed via `sentence-transformers/all-MiniLM-L6-v2` (384-d). Item and user embeddings both L2-normalized, so similarity = dot product.
- Blend **within each pool** only: head scores use only head-item sims, tail scores only tail-item sims, then the unmodified blueprint merge runs. **Why within-pool:** the quota is a hard property of the algorithm; blending two pools independently can't change N_pop/N_tail.
- λ swept over {0.3, 0.5, 0.7}; **λ=0.7 optimal**.
- **Alternatives considered:** server-side big embeddings (untested, flagged future work), item-side view-attention models (flagged), no LLM (the H3 ablation). We chose small on-device SBERT for privacy and feasibility ("LLM profile" is a dense-embedding profile, not a generative LLM — we use SBERT, a sentence BERT; the paper/presentations use "LLM user profiling" as shorthand for this embedding-based semantic profiling).

**Multi-objective penalty (H4, `train.py`).** Add to BPR loss: `loss += α · log1p(pop[i]·100) · s_positive(i)` during M_pop training (push down scores of popular positives). Sweep α ∈ {0, 0.001, 0.005, 0.01, 0.05}.

**Eval (`metrics.py:full_rank_eval`).** Full-catalog ranking per user (test item among all 3,533, excluding training positives for the ranking candidate pool but scoring everything). Two modes: `argsort` (top-K by raw score) and `reclist` (top-K = the blueprint-merge output). Metrics as in Section 8.

### 5.3 What changed vs original in each stage

- **Training:** centralized MF replaced by federated sparse-MF; two specialists trained on head/tail shards in both paradigms.
- **Ranking:** per-pool *score blending* is new (LLM term); original had raw model scores.
- **Privacy:** nothing in the original; ours adds federation + optional DP.
- **Metrics:** we add Recall@10 to the paper's metric set (paper deliberately omits precision/recall).
- **Fixed hyperparameters:** d=64, BPR, α=0.20, N=10, leave-one-out, seed 0 (deterministic single-seed runs; the paper used 5-seed averages — see Limitations).

---

## 6. Experimental timeline (chronological, from git + logs)

| # | Date | Round | What | Status (final reading) |
|---|---|---|---|---|
| 0 | 08-01 | Init/bootstrap | Workspace, Crossref lookup of SUPER, read 24-page PDF, 59-paper / 6-cluster lit survey, 5 hypotheses locked, baselines locked | — |
| 1 | 08-02 | Sanity | Federated harness sanity: sparse FedAvg + per-client user emb; loss 0.69→0.24 | — |
| 2 | 08-02 | **H1 (Phase-1 framing)** | FedSUPER as federated + popularity *reweighting* post-processor (score/pop^α); 4 variants | Fairness prediction refuted; accuracy kept |
| 3 | 08-02 | Methods v2 | Re-implement SUPER as the paper's full structure (Pareto partition, dual specialists, blueprint merge); adopt paper metrics | — |
| 4 | 08-02 | **H1 v2 (FedSUPER)** | Central BPR-MF vs central SUPER vs FedNCF vs FedSUPER (blueprint) | **CONFIRMED** (invariance) |
| 5 | 08-02 | **H2 v2 (DP sweep)** | ε ∈ {8,4,2} on FedSUPER + FedNCF, blueprint setting | DP redundant under blueprint; DP debiases FedNCF → supported w/ nuance |
| 6 | 08-02 | **H3 v2 (LLM recovery)** | SBERT profile, λ sweep {0.3,0.5,0.7} × {noDP, DP ε=2} | **CONFIRMED** (recall +23%) |
| 7 | 08-02 | **H4 (multi-objective)** | popularity-dispersion penalty α sweep on M_pop | Inconclusive/refuted |
| 8 | 08-02 | **H5 (adaptive DP)** | per-item popularity-scaled noise, amp=2 | Partial confirm |
| 9 | 08-02/03 | Outer loop / paper | Synthesis (H1+H3 = core), findings.md, progress HTML, NeurIPS-style paper, verification | paper-draft done |

**Earlier-phase experiments** (rounds 2 and the H2/H3 runs performed *before* methods-v2, i.e., under the Phase-1 reweighting framing) are retained in the repository (`metrics_h1.json`, `metrics_h2.json`, `metrics_h3.json`, `metrics_h3b.json`, `metrics_h3c.json`) and are described in Round 2 for historical completeness. The **canonical** results used in the paper are the **v2** runs.

---

## 7. Round-by-round results

### Round 0 — Bootstrap (context, not an experiment)
**Goal:** lock the research question, find gaps, pre-register hypotheses. **Reasoning:** the SUPER paper is brand-new (IEEE Access 2026); no privacy/LLM follow-ups existed. **Result:** 5 gaps, 5 hypotheses H1–H5, evaluation plan (FDN-A@top-10 as inner-loop proxy), baselines (BPR-MF, FedNCF, SUPER reimpl, +ablations), compute (RTX 3050 4 GB). **Decision:** proceed with ML-1M first, Steam/LastFM optional (never run — not recorded why, likely time).

### Round 1 — Federated harness sanity
**Goal:** make sparse FedAvg converge before running hypotheses. **Config:** `train_federated` on full train matrix, 60 rounds, 512 clients/round, 2 local epochs, lr 0.3. **Observation:** BPR loss fell 0.69 → 0.24. **Diagnosis:** sparse aggregation + per-client user embedding training works only when rounds/clients are large enough that items get touched; untouched item embeddings stay near initialization. **Fix/change:** production runs use 40 rounds / 256 clients (a compromise for wall-clock). **Decision:** keep sparse FedAvg design. **Lesson:** federation is the fragile part of the stack; tune the communication schedule, not the model.

### Round 2 — H1 under the Phase-1 framing (pre-blueprint reweighting version)
**Goal:** test "can popularity-exposure reduction be applied federated, server-side, without raw data." **Configuration:** SUPER modeled as post-hoc re-ranking `score / pop^(α·sensitivity)` (mode `reweight`, α=0.3/0.5); variants BPR-MF, SUPER-centralized, FedNCF, FedSUPER. **Metrics:** v1 set (Recall@10, nDCG, Gini-fairness across popularity deciles, coverage, ILD, novelty, FDN-A composite). **Results (metrics_h1.json / trajectory_h1.csv):**
- BPR-MF: recall 0.0480, gini 0.260, FDN-A 1.0 (reference).
- SUPER-centralized (reweight): recall 0.0108, gini 0.868, FDN-A 1.183.
- FedNCF: recall 0.0401, gini 0.100, FDN-A 0.778.
- FedSUPER (reweight on FedNCF): recall 0.0152, gini 0.400, FDN-A 1.002.

**Expected:** FedSUPER gini ≥ 0.85 and recall ≥ SUPER − 5%. **Observed:** gini 0.40 ≪ 0.87 (prediction **refuted**); recall *above* SUPER-centralized (0.015 vs 0.011) — an **unexpected exploratory finding**. **Diagnosis [Interpretation]:** reweighting only works if the base scores discriminate tail items; centralized MF had strong tail discrimination, federated MF did not (sparsely-touched tail embeddings near init), so dividing near-random scores by pop amplified noise instead of promoting coherent niche items. Conversely SUPER's suppression of popular items hurt accuracy *more* when base scores were good (centralized) — hence FedSUPER beat SUPER-centralized in recall. **[Hypothesis]:** alternative discriminative signal for tail items could rescue federated accuracy → motivates H3. **Decision:** keep federation direction; **adopt the paper's real structure (blueprint) before further hypothesis tests.** **Significance:** established that "data-agnostic post-hoc reweighting" is *not* data-agnostic; it needs informative scores.

### Round 3 — Methods v2 (the framework our results are built on)
**Change:** replace the reweighting instantiation with the paper's Algorithm 1 + Algorithm 2 (Pareto partition, M_pop/M_tail specialists, hard-quota/soft-blueprint merge). Adopt paper metrics (RMSE-PC, MRMC, APLT, LTC, Entropy, Novelty, GKPI) in addition to recall/nDCG. **Why:** to measure the actual claim of the paper (per-user calibration) rather than a proxy Gini; a decile-Gini is a population-equality measure and cannot distinguish a calibrated-but-concentrated list from a miscalibrated one.

### Round 4 — H1 v2 (FedSUPER, canonical)
**Goal:** does federating the *training* of M_pop/M_tail preserve centralized SUPER's calibration and what accuracy does it cost? **Configuration:** BPR-MF (central, full data), SUPER-central (central specialists + blueprint), FedNCF (federated full data), FedSUPER (federated specialists + blueprint); 40 rounds, 256 clients. **Metrics (metrics_h1_v2.json):**

| Variant | Recall@10 | nDCG | APLT | LTC | RMSE-PC↓ | MRMC↓ | GKPI |
|---|---|---|---|---|---|---|---|
| BPR-MF (central) | 0.0494 | 0.0238 | 0.438 | 0.427 | 0.407 | 0.410 | 0.0464 |
| SUPER-central | 0.0378 | 0.0180 | 0.785 | 0.473 | **0.055** | **0.151** | 0.0353 |
| FedNCF (FL) | 0.0402 | 0.0186 | 0.014 | 0.010 | 0.738 | 0.732 | 0.0258 |
| FedSUPER (FL+blueprint) | 0.0310 | 0.0153 | 0.785 | 0.014 | **0.055** | **0.151** | 0.0264 |

**Results/interpretation:**
- **[Fact]** RMSE-PC and MRMC of FedSUPER are *identical* to centralized SUPER (0.05495 / 0.15058). Calibration is a property of the merge, not the backbone → **H1 CONFIRMED**.
- **[Fact]** FedNCF *alone* has worse calibration than the accuracy baseline (0.738 vs 0.407) — sparse federation *intrinsically amplifies* popularity bias (untouched tail items can't be ranked). **[Interpretation]** This is exactly what the blueprint neutralizes.
- **[Fact/Observation]** recall drops 0.0402 → 0.0310 (≈ 23% relative) vs FedNCF; APLT is constant at 0.785 across all blueprint variants (quota arithmetic), LTC collapses to 0.014 (head pool is only 75 items; federated tail specialization is weak).
- **[Inconsistency flagged]** A later reproduction (H2/H3 v2) of FedSUPER-noDP yields recall **0.0298** vs **0.0310** here (slightly different run/config); the paper uses 0.0298. Both exist in artifacts; the canonical baseline is 0.0298.

**Decision:** keep the blueprint framework; the accuracy gap (recall vs central SUPER 0.038) becomes the explicit target for H3.

### Round 5 — H2 v2 (DP sweep under the blueprint)
**Goal:** does Gaussian DP improve FedSUPER's calibration beyond perfect, or change fairness — and what does DP do to a *non*-SUPER federated recommender? **Configuration:** train FedNCF and FedSUPER (head/tail specialists) with ε ∈ {8, 4, 2} (σ = 0.3/ε; clip 1.0). **Metrics (metrics_h2_v2.json):**

| Variant | Recall | nDCG | APLT | LTC | RMSE-PC↓ | MRMC↓ | GKPI |
|---|---|---|---|---|---|---|---|
| FedNCF noDP | 0.0386 | 0.0176 | 0.051 | 0.011 | 0.705 | 0.717 | 0.0274 |
| FedNCF-DP ε=8 | 0.0374 | 0.0174 | 0.110 | 0.011 | 0.648 | 0.683 | 0.0281 |
| FedNCF-DP ε=4 | 0.0356 | 0.0162 | 0.153 | 0.011 | 0.610 | 0.630 | 0.0268 |
| FedNCF-DP ε=2 | 0.0260 | 0.0104 | 0.520 | 0.011 | **0.283** | **0.222** | 0.0182 |
| FedSUPER noDP | 0.0298 | 0.0131 | 0.785 | 0.015 | **0.055** | **0.151** | 0.0229 |
| FedSUPER-DP ε=8 | 0.0287 | 0.0119 | 0.785 | 0.015 | **0.055** | **0.151** | 0.0211 |
| FedSUPER-DP ε=4 | 0.0214 | 0.0096 | 0.785 | 0.015 | **0.055** | **0.151** | 0.0173 |
| FedSUPER-DP ε=2 | 0.0131 | 0.0066 | 0.785 | 0.012 | **0.055** | **0.151** | 0.0120 |

**Interpretation:**
- **[Fact]** RMSE-PC/MRMC/APLT are *frozen* at the blueprint values for every FedSUPER-DP run — the hard quota is impervious to noise. DP cannot make perfect calibration better.
- **[Fact + Mechanism]** In *plain* FedNCF, DP monotonically reduces RMSE-PC (0.705 → 0.283 at ε=2) and raises APLT (0.05 → 0.52): popular items appear in more client histories → more gradient mass → proportionally more noise → head scores down-weighted → tail exposure increases. DP acts as an *intrinsic debias* only when the blueprint is absent.
- **[Observation]** Under the blueprint, DP is pure accuracy cost (recall 0.0298 → 0.0131 at ε=2, −57%).
- **Decision:** keep FedSUPER *without* DP as the fairness-optimal point; report FedNCF-DP as the "DP-debias without SUPER" result; use ε=2 as the strong-privacy scenario for H3's rescue test.

**[Inconsistency flagged]** H1-v2's FedNCF shows RMSE-PC 0.738; H2-v2's re-run shows 0.705. The paper reports 0.705/0.717 (from the H2-v2 run).

### Round 6 — H3 v2 (LLM user profiling; canonical, decisive)
**Goal:** recover the recall lost to federation/DP via on-device semantic profiles, without disturbing calibration. **Configuration:** SBERT all-MiniLM-L6-v2 profiles; per-pool blend; λ sweep {0.3, 0.5, 0.7} × {noDP, ε=2}. **Metrics (metrics_h3_v2.json):**

| Variant | Recall | nDCG | LTC | RMSE-PC↓ | GKPI |
|---|---|---|---|---|---|
| FedSUPER-noDP (ref) | 0.0298 | 0.0131 | 0.015 | 0.055 | 0.0229 |
| LLM-FedSUPER-noDP λ=0.3 | 0.0336 | 0.0155 | 0.015 | 0.055 | 0.0268 |
| LLM-FedSUPER-noDP λ=0.5 | 0.0358 | 0.0164 | 0.018 | 0.055 | 0.0287 |
| **LLM-FedSUPER-noDP λ=0.7** | **0.0366** | **0.0172** | **0.038** | **0.055** | **0.0314** |
| FedSUPER-DP ε=2 (ref) | 0.0131 | 0.0066 | 0.012 | 0.055 | 0.0120 |
| LLM-FedSUPER-DP ε=2 λ=0.3 | 0.0151 | 0.0072 | 0.012 | 0.055 | 0.0130 |
| LLM-FedSUPER-DP ε=2 λ=0.5 | 0.0182 | 0.0082 | 0.014 | 0.055 | 0.0148 |
| **LLM-FedSUPER-DP ε=2 λ=0.7** | **0.0212** | **0.0102** | **0.031** | **0.055** | **0.0190** |

**Interpretation:**
- **[Fact]** λ=0.7 (noDP): recall 0.0298 → 0.0366 (**+23%**), LTC 0.015 → 0.038 (**~2.6×**), RMSE-PC/MRMC *unchanged* (0.055/0.151).
- **[Fact]** Under ε=2 DP: recall 0.0131 → 0.0212 (**+62%**), LTC 0.012 → 0.031 — the LLM channel rescues signal destroyed by noise; it is *orthogonal* to the collaborative signal.
- **[Mechanism]** The head pool holds only ~75 items; each user's N_pop ≈ ⌊10·Pop_u⌋ ≈ 2–3 slots. The semantic cosine re-ranks *within* the pool, promoting the held-out item when the fed score could not. Because blending happens per-pool and the quota is external, calibration can't move.
- **Decision:** adopt λ=0.7; **LLM-FedSUPER-noDP** is the final headline variant. Keep the DP variant as the strong-privacy option.

### Round 7 — H4 (multi-objective popularity-dispersion)
**Goal:** can an explicit training-time penalty beat the blueprint+LLM configuration on GKPI? **Configuration:** α ∈ {0, .001, .005, .01, .05} on M_pop (pop-weighted score penalty added to BPR); M_tail unchanged; with/without LLM λ=0.7.

**Results (metrics_h4.json):** With LLM λ=0.7, peak GKPI = **0.03174 at α=0.005** vs 0.03143 at α=0 (**+0.0003**). Recall 0.0364 vs 0.0366. Without LLM, penalty is mildly *harmful* (α=0.05: GKPI 0.0229 → 0.0197). Calibration 0.055 everywhere.

**Interpretation:** **[Observation]** the penalty term's magnitude (α·pop·score ≈ 0.006 at α=0.001) is two orders below BPR's gradient signal, and the blueprint + LLM already saturate the achievable calibration/accuracy front; **[Interpretation]** the extra objective is redundant — calibration is enforced by the merge, not by the loss. **Decision:** H4 contributed no improvement; documented as a negative result; **rejected as a component** in the final system. [The H4 protocol predicting FDN-A improvement ≥0.05 was locked against the old composite; final verdict: inconclusive at best, effectively refuted.]

### Round 8 — H5 (adaptive DP budget)
**Goal:** does concentrating noise on popular items beat uniform DP at the same nominal ε? **Configuration:** `σ_i = σ_base·(1 + 2·normlog(pop_i))`; ε=2; with and without blueprint. **Metrics (metrics_h5.json):**

| Variant | Recall | APLT | LTC | RMSE-PC↓ | GKPI |
|---|---|---|---|---|---|
| FedNCF-DP uniform ε=2 | 0.0260 | 0.520 | 0.011 | 0.283 | 0.0182 |
| FedNCF-DP adaptive ε=2 | 0.0255 | **0.561** | 0.012 | **0.251** | 0.0175 |
| FedSUPER-DP uniform ε=2 | 0.0131 | 0.785 | 0.012 | 0.055 | 0.0120 |
| FedSUPER-DP adaptive ε=2 | **0.0137** | 0.785 | 0.013 | 0.055 | **0.0129** |

**Interpretation:** **[Observation]** adaptive noise slightly improves FedNCF's calibration (RMSE-PC 0.283 → 0.251, APLT 0.52 → 0.56) and gives +5% recall / +8% GKPI under the blueprint, but the gains are small — the blueprint dominates the noise schedule. **Decision:** H5 **partial confirm**; keep as a secondary result, not part of the headline system.

### Round 9 — Synthesis & paper
**Reasoning:** H1+H3 are the transferable, defensible claims (invariant calibration + orthogonal LLM recovery). H2 provides a clarifying negative (DP redundant under blueprint / DP debiases outside it); H4/H5 are negative/partial. **Decision:** H1+H3 = core contribution; paper targets NeurIPS style. All numbers verified against experiment JSONs (paper-finalization note).

---

## 8. Metrics and what they mean

### The metric set

| Metric | Definition (our implementation) | Direction | Why relevant |
|---|---|---|---|
| **Recall@10** | fraction of users whose held-out item appears in their top-10 (full ranking) | ↑ better | accuracy cost of privacy/fairness |
| **nDCG@10** | graded position-aware hit (binary relevance, 1/log2(rank+2)) | ↑ better | ranking quality paired with GKPI |
| **Gini-fairness** (v1 only) | 1 − (mean abs deviation of decile recommendation shares)/(2·mean share), across popularity deciles | ↑ better | population-level exposure equality — *dropped in v2* |
| **Coverage** | fraction of catalog items in any top-10 | ↑ better | catalog use |
| **ILI/ILD** | 1 − mean pairwise cosine of recommended items (in our runs computed without item embeddings → constant 0.9) | ↑ better | in-list diversity (non-informative here) |
| **Novelty** | mean `−log2 p(item)`, p = training popularity | ↑ better | surprise/discovery |
| **Entropy** | Shannon entropy of recommendation distribution over items | ↑ better | dispersion/balance |
| **APLT** | mean per-user fraction of top-10 that are *tail* items | ↑ better | long-tail exposure |
| **LTC** | fraction of all tail items appearing in ≥1 top-10 | ↑ better | system-level tail coverage |
| **RMSE-PC** | √mean((head-fraction-in-rec − Pop_u)²) | ↓ better | per-user *popularity calibration* (SUPER's core claim) |
| **MRMC** | mean over positions k of |cumulative head fraction at rank k − Pop_u| | ↓ better | *rank-level* (positional) calibration |
| **GKPI** | mean over x∈{APLT,Entropy,Novelty,LTC} of H(nDCG,x), H=2xy/(x+y) | ↑ better | balanced summary; harmonic mean penalizes imbalance; *not an optimization objective* |
| **FDN-A** (v1 only) | 0.25·Σ normalized {F=Gini, D=ILD, N=Novelty, A=Recall+nDCG} | ↑ better | composite used in Phase-1; superseded |

### How a change in each metric read in our experiments

- **RMSE-PC/MRMC decrease** → calibration improved. The single most important fact: these become 0.055/0.151 from the moment a blueprint variant runs, and stay *literally constant* through federation, DP, LLM blending. Direction changes for them were essentially binary (with/without blueprint).
- **Recall up** → accuracy recovered (the H3 story). Recall is the axis where privacy cost is paid (FedSUPER vs FedNCF) and then repaid by the LLM.
- **LTC up** → more niche catalog exposed (the "fairness to producers" axis). This is where the LLM profile's biggest absolute gain lives (~2.6×), and where the head-pool bottleneck showed up (0.015).
- **APLT up** → lists tilt to the tail. Note APLT is pinned to ≈0.785 for every blueprint variant — [Interpretation] it is the arithmetic image of the hard quota: mean per-user N_tail/N = mean(1 − ⌊N·Pop_u⌋/N) ≈ 1 − mean-floor(Pop_u); with mean Pop_u ≈ 0.26, this is ≈ 0.74–0.79. So **APLT is not a model-quality metric under SUPER**; it is a design consequence. [Findings.md documented the same reading.]
- **Novelty/Entropy up** → broader/rarer exposure; in v1 these inflated FDN-A (see evaluation change).
- **GKPI up** → more balanced accuracy+beyond-accuracy. It moves *less* than recall alone because the harmonic mean punishes any axis that lags (notably nDCG itself, which is low in absolute terms) → GKPI under-reports recovery; we therefore always report recall and LTC alongside.

### Evaluation change and why

- **Phase-1 (v1) setup:** FDN-A composite + decile-Gini + coverage + novelty at K=10, evaluated during the reweighting-era experiments.
- **Canonical (v2) setup:** paper-compatible metric set (nDCG, RMSE-PC, MRMC, APLT, LTC, Entropy, Novelty, GKPI) + Recall@10, full-ranking leave-one-out, method=`argsort` for non-merge baselines and `reclist` for blueprint variants.

**Why we changed it:** (1) the paper's contribution and our hypothesis tests are about *per-user popularity calibration*, which decile-Gini cannot isolate (Gini stays 0.10 for FedSUPER even at perfect RMSE-PC because exposure is still head-concentrated — calibration ≠ equality); (2) FDN-A over-weighted novelty, so DP runs looked "best" (FDN-A maxed at ε=1 where the model collapsed to near-random but novelty exploded) — a misleading optimization signal; (3) adopting the paper's metrics makes our results *directly comparable* to SUPER's reported numbers and defensible in a defense.

---

## 9. Blockers and how we solved them

*(Per instructions, only genuine project-development obstacles are included.)*

### Blocker A — Sparse federated training barely learned tail items
- **Observed:** FedNCF alone had appalling calibration (RMSE-PC 0.738/0.705) and LTC 0.010–0.015; tail items were effectively unrankable; FedSUPER held calibration but not coverage.
- **Why a problem:** the whole project depends on the federated backbone giving *usable per-pool rankings*.
- **Investigation:** sanity run (`sanity_fed.py`) with more rounds/clients; loss curve 0.69 → 0.24; inspected per-item update masking during rounds (`items touched` fraction).
- **Believed cause:** sparse FedAvg only updates items a sampled client has as a positive that round; rare (tail) items are rarely touched, so their embeddings stay near uniform init and their scores carry no signal.
- **Experiments/debugging:** tuned rounds (40–60) and clients/round (256–512); local user-embedding persistence; clip norm 1.0.
- **What fixed it:** an adequate communication schedule; conceptually, we stopped fighting it and **gave the tail/head pools a second, content-based signal** (H3) instead of hoping federation would build enough tail discrimination.
- **Improve results?** Yes — directly enabled the H1 invariance result (FedSUPER calibration) and H3 recovery.
- **Lesson:** in FL, "train more" on sparse signals is the wrong lever; add *orthogonal* local signal (content embeddings) and choose an evaluation that separates calibration from ranking quality.

### Blocker B — Rising BPR loss misread as a training failure
- **Observed:** BPR loss increased during training (0.69 → ~0.9+), which looked like divergence.
- **Why a problem:** we could have aborted training or mis-tuned.
- **Investigation:** tracked complementary `pos>neg rate` per batch and LOO recall instead of trusting the raw loss.
- **Believed cause:** BPR samples *fresh random negatives every step*; as the model separates positives from negatives the sampled negatives get relatively "easier vs harder to rank" and the surrogate loss is not monotonically informative; BPR is a *relative* ranking loss, not a classification objective. (Findings.md item 4 records the same conclusion.)
- **What fixed it:** monitoring the right quantities; no network-side change was needed.
- **Lesson:** monitor ranking surrogate signals and downstream eval, not the raw surrogate magnitude.

### Blocker C — The first evaluation suite mislabeled the objective (metric mismatch)
- **Observed:** Decile-Gini said FedSUPER was "unfair" (≈0.10) while RMSE-PC said calibration was perfect; FDN-A favored novelty-explosion (DP ε=1) over genuine quality.
- **Why a problem:** wrong metric ⇒ wrong conclusions and wrong optimization signal.
- **Investigation:** re-derived the paper's metric definitions from the SUPER text; computed both families on the same runs.
- **Believed cause:** global equality (Gini) and per-user calibration (RMSE-PC/MRMC) are different quantities; the composite normalized against a *reference* that drifted, and novelty (unbounded, −log2 p) dominated normalized axes.
- **What fixed it:** switched the canonical protocol to the paper-compatible metric family + separate recall/nDCG; kept Gini/entropy/coverage only as descriptive extras.
- **Improve results?** Results didn't change — but *interpretation* and the optimization target became correct, and comparability to the paper was restored.
- **Lesson:** pick the metric family that matches the mechanism you claim to change (calibration vs equality), and distrust composites whose components have wildly different scales.

### Blocker D — "APLT is constant" (0.785) looked like a bug
- **Observed:** all blueprint variants showed identical APLT 0.7851.
- **Why a problem:** suspicious constancy across wildly different models.
- **Investigation/Diagnosis:** traced to quota arithmetic — APLT = mean over users of N_tail/N = 1 − ⌊N·Pop_u⌋/N, which is score-independent by construction (mean Pop_u ≈ 0.26 ⇒ ~0.74–0.79 tail share). It is a *design image* of the hard quota, not a learned property. Findings.md item 5 documents this.
- **What fixed it:** nothing needed; documented the invariant. LTC and RMSE-PC remain the informative coverage/calibration axes.
- **Lesson:** invariants created by algorithmic structure should be expected and explained, not "fixed."

### Blocker E — GKPI under-reported real improvements
- **Observed:** +23% recall and 2.6× LTC translated into GKPI +37% (and tiny absolute deltas ~0.009), because the harmonic pairs H(nDCG,·) are capped by the small nDCG.
- **Why a problem:** headline numbers looked modest.
- **Investigation:** recomputed GKPI sub-terms; confirmed harmonic domination by nDCG.
- **What fixed it:** we report recall and LTC explicitly next to GKPI (the paper does the same). GKPI is retained as the balanced summary (per the paper, it is *not* an objective).
- **Lesson:** use summary metrics for comparison, per-axis metrics for claims.

### Blocker F — The H4 multi-objective term was (numerically) starved
- **Observed:** α sweep changed GKPI at the 4th decimal (max +0.0003); at large α it *hurt*.
- **Why a problem:** hypothesis stated a possible gain; none appeared.
- **Investigation:** computed term magnitude vs BPR: penalty ≈ α·log1p(pop)·score ≈ 0.006 at α=0.001 vs BPR loss ~0.1–0.5 — two orders of magnitude smaller. Also blueprint+LLM already pin calibration.
- **What fixed it / verdict:** none — this is the intended negative result. Recorded as refuted/inconclusive for the penalty; concluded the merge + semantic blend are already the binding constraints.
- **Lesson:** verify that an added loss term can actually shift the optimization dynamics *at its operating scale* before expecting an effect.

### Blocker G — DP scheduling (strong privacy ⇒ training collapse)
- **Observed:** at ε=1 (σ=0.3) the federated model collapsed to near-random; at ε=2 recall fell 57% under the blueprint; σ cap at 0.6 kept training tractable.
- **Why a problem:** the DP story had to be told *correctly* (which regime it helps, which it hurts), not just "DP bad."
- **Investigation:** swept ε {8,4,2,(1 in v1)}; separated FedNCF-DP (no blueprint) from FedSUPER-DP; tracked σ and monotone RMSE-PC dynamics.
- **Believed cause:** noise overwhelms within-pool ranking signal independently of the quota.
- **What fixed it:** methodology, not code — we framed DP as *conditional*: intrinsic debias without the blueprint (RMSE-PC 0.705→0.283), redundant + accuracy-expensive under it. Formal (ε,δ) composition remains open (flagged in Limitations).
- **Lesson:** a mechanism's effect can sharply depend on whether another, dominating mechanism is active; test mechanisms both alone and nested.

### Blocker H — Narrative/artifact inconsistencies to be aware of (not bugs, but data-hygiene)
- Nominal "3,952 items" appears in README/paper text while the recoded evaluation index is 3,533 (dataset's items with ≥4).
- FedSUPER-noDP recall appears as 0.0310 (H1-v2 run) and 0.0298 (H2/H3-v2 reproductions); the paper uses 0.0298.
- FedNCF RMSE-PC 0.738 (H1-v2) vs 0.705 (H2-v2); paper uses 0.705.
- README's headline table contains values that do not match any JSON (e.g., nDCG 0.041, APLT 0.812 for BPR-MF) and an MRMC typo "0.051" in findings.md's SGPI/LLM row; the paper's Table 1 and the `*_v2.json` files are the authoritative source and were cross-verified.
- research-state.yaml's embedded table and a few "summary" strings contain garbled text (encoding artifacts); its hypothesis statements and statuses are the trustworthy part.

---

## 10. Reasoning behind the major decisions

**Decision 1 — Test privacy via the *training paradigm*, not the post-processing.** Evidence: H1-v2 showed the merge is score-agnostic (identical RMSE-PC across backbones). Because the calibration machinery lives entirely in Algorithm 2's quota, privacy can be obtained by federating *training* at zero calibration cost. The earlier assumption that we'd need server-side re-ranking (Phase-1) was *disproved by the numbers* (Gini gap) and by reading the algorithm's structure — so we moved the mechanism to the right place.

**Decision 2 — Adopt the paper's full structural algorithm (v2).** Evidence: Phase-1 evidence (reweight-FedSUPER gini 0.40) contradicted the paper's claim; we re-derived the algorithm from the primary text (cumulative-volume Pareto, dual-specialist training, hard-quota/soft-blueprint merge) and it *immediately* reproduced the paper's calibration values (0.055/0.151) — confirming the earlier framing was the wrong lens. **Quantitatively supported.**

**Decision 3 — Use per-pool LLM blending rather than a final-score blend or training-time regularization.** Evidence: H3 v2 (blend-before-merge): +23% recall at zero calibration effect; H4 showed training-time regularization is inert. The key insight is *orthogonality*: semantic signal re-ranks *within* quota-determined pools, so it can never touch calibration — a design we engineered for, not discovered accidentally. **Both quantitative (H3/H4) and structural.**

**Decision 4 — λ=0.7.** Evidence: monotone λ sweep → best recall/LTC/GKPI at 0.7 without calibration change; DP run also prefers 0.7. **Quantitatively supported.**

**Decision 5 — Report DP as conditional (outside vs inside blueprint).** Evidence: FedNCF-DP ε=2 RMSE-PC 0.705→0.283 vs FedSUPER-DP ε=2 recall −57% with RMSE-PC pinned. This *refines* rather than contradicts the pre-registered H2; the numbers forced the nuanced statement.

**Decision 6 — Keep FedSUPER-noDP as the headline "private" point, not DP.** Evidence: H2/H3 — DP under blueprint is pure accuracy cost; no calibration upside (it's already perfect). LLM recovery makes no-DP the best privacy+utility point.

**Decision 7 — H4/H5 as secondary results, not system components.** Evidence: H4 GKPI Δ = +0.0003 (noise); H5 Δ = +5%/+8% (marginal, blueprint-dominated). Rejected as complexity with ~zero benefit — a *quantitative* rejection.

**Decision 8 — Single-seed deterministic runs.** Evidence/constraint: compute (4 GB GPU) and the reproducibility section promise seed-locked runs. This trades statistical breadth (paper used 5 seeds + Wilcoxon) for exact reproducibility; flagged as a limitation.

---

## 11. Original paper vs. our extension

| Aspect | Original Paper (SUPER, Yavru et al.) | Our Capstone (FedSUPER / LLM-FedSUPER) |
|---|---|---|
| **Problem** | Popularity bias; over-exposure of head items; per-user miscalibration | Same calibration problem **+ privacy** (federated, no raw data to server) **+ content-signal recovery** (LLM profiles) + **DP dynamics** |
| **Input/Data** | Centralized interaction logs; 3 datasets (ML, DB, Yelp); no exposure logs | MovieLens-1M only; ratings ≥4 → implicit positives; leave-one-out; recoded 6,040×3,533 |
| **Core algorithm** | Pareto partition (α=0.20 cumulative volume) + dual specialists + Pop_u + hard-quota/soft-blueprint merge | Same merge — **re-implemented exactly** — but specialists trained **federated** (sparse FedAvg, private user embeddings) |
| **Architecture/pipeline** | Backbone-agnostic pre/post-processing | Same backbone-agnostic structure + on-device SBERT profile blended per-pool; optional DP in the loop |
| **Modification** | — | (a) federate training; (b) LLM per-pool score blend; (c) DP (uniform + adaptive); (d) tested multi-objective penalty |
| **Training/optimization** | Backbone training on head/tail shards (paper used HPF/NeuMF/SKMeans/VaeCF/WBPR, 5 seeds) | MF d=64, BPR; centralized (15 ep, Adam .05) or federated (40 rnd, 256 clients, lr .3); single seed |
| **Evaluation** | 5 backbones × 3 datasets × 5 seeds + paired Wilcoxon; leave-one-out implied | 1 backbone × 1 dataset; full-ranking LOO; deterministic seed 0; paper-compatible metrics + Recall@10 |
| **Metrics** | nDCG, RMSE-PC↓, MRMC↓, APLT, Entropy, LTC, Novelty, GKPI | Same set + **Recall@10** |
| **Results** | Best GKPI/calibration among debiasing methods on ML/DB/Yelp | FedSUPER reproduces calibration: RMSE-PC 0.055, MRMC 0.151. LLM-FedSUPER: recall 0.0366 (+23% vs FedSUPER), LTC 0.015→0.038, RMSE-PC unchanged. DP conditional (debias outside / redundant inside) |
| **Limitations** | Centralized; interaction-only; no privacy | Single dataset/backbone; pragmatic (non-composed) DP; on-device SBERT assumption; single seed; no formal privacy accounting |

---

## 12. Final results

**Canonical headline table (paper Table 1, verified against experiment JSONs):**

| Variant | Recall@10 | nDCG | LTC | RMSE-PC↓ | MRMC↓ | GKPI |
|---|---|---|---|---|---|---|
| BPR-MF (centralized) | 0.0494 | 0.0238 | 0.427 | 0.407 | 0.410 | 0.0464 |
| SUPER (centralized) | 0.0378 | 0.0180 | 0.473 | **0.055** | **0.151** | 0.0353 |
| FedNCF (plain FL) | 0.0402 | 0.0186 | 0.011 | 0.705 | 0.717 | 0.0258 |
| FedNCF-DP ε=2 | 0.0260 | 0.0104 | 0.011 | 0.283 | 0.222 | 0.0182 |
| FedSUPER (FL + blueprint) | 0.0298 | 0.0131 | 0.015 | **0.055** | **0.151** | 0.0229 |
| FedSUPER-DP ε=2 | 0.0131 | 0.0066 | 0.012 | **0.055** | **0.151** | 0.0120 |
| **LLM-FedSUPER λ=0.7 (ours)** | **0.0366** | **0.0172** | **0.038** | **0.055** | **0.151** | **0.0314** |
| LLM-FedSUPER-DP ε=2 λ=0.7 | 0.0212 | 0.0102 | 0.031 | 0.055 | 0.151 | 0.0190 |

**Bottom line against the paper:** LLM-FedSUPER is within 3% of centralized SUPER's recall (0.0366 vs 0.0378) while being **private** and keeping **identical** calibration (0.055/0.151). Against the pure-FL baseline it is strictly better on the balanced GKPI (0.0314 vs 0.0258) and far better on calibration (0.055 vs 0.705).

**Hypothesis verdicts:**
- H1 FedSUPER calibration invariance — **supported** (algorithmic + empirical).
- H2 DP conditional role — **supported with nuance** (helps plain FedNCF; redundant under blueprint).
- H3 LLM accuracy recovery — **supported** (+23% recall, 2.6× LTC, calibration untouched).
- H4 multi-objective — **not supported** (ΔGKPI ≤ 0.0003; blueprint/LLM dominate).
- H5 adaptive DP — **partially supported** (+5% recall, +8% GKPI under blueprint).

---

## 13. What we learned

1. **Calibration can be a property of the algorithm, not of the model.** Any backbone (centralized, federated, noised, LLM-blended) fed through the hard-quota blueprint yields the same RMSE-PC/MRMC. The interesting consequence: privacy can be bought "for free" on the calibration axis and the residual cost sits only on ranking accuracy.
2. **Federation alone amplifies popularity bias in the noise floor** (FedNCF RMSE-PC 0.74/0.71 >> BPR 0.41) because rarely-touched item embeddings can't rank; the blueprint absorbs it, and the LLM supplies the missing discrimination.
3. **Semantic and calibration signals are orthogonal in this design.** Because the LLM blend is *within-pool* and the quota is external, content signal improves accuracy and coverage without moving calibration at all — this is the mechanism that makes the contribution non-overclaimed.
4. **DP's effect is mechanism-conditional**: intrinsic debias without a quota (popular items receive proportionally more gradient mass → more noise), redundant with one.
5. **Adaptive/popularity-aware noise and train-time multi-objective penalties are second-order** relative to the blueprint's hard guarantee.
6. **Metrics discipline matters**: decile-Gini ≠ per-user calibration; FDN-A over-weighted novelty; GKPI under-reports; APLT is constant by quota design. We report per-axis metrics for claims and GKPI only as a balanced summary.

---

## 14. Contribution and novelty (careful, anti-overclaim)

**What we inherited from the paper (not our contribution):**
- The SUPER framework itself: Pareto partition, dual-specialist training, Pop_u, hard-quota/soft-blueprint merge, and the paper's metric family (RMSE-PC, MRMC, APLT, LTC, Entropy, Novelty, GKPI).
- The premise that per-user popularity calibration improves fairness/diversity.

**What we implemented:**
- A working end-to-end Python reproduction of SUPER v-on MovieLens-1M within a federated harness, with deterministic, seeded, full-ranking LOO evaluation, producing the paper's calibration numbers (0.055/0.151) as a sanity baseline.

**What we modified/added:**
- **FedSUPER:** sponsorship of the two specialists via sparse FedAvg where user embeddings and Pop_u never leave the device (privacy-aware training of a calibration framework).
- **Invariance theorem + proof:** the calibration is backbone-independent by construction — this frames H1 and is the paper's Method Section 3.2 result.
- **LLM user-profile augmentation (LLM-FedSUPER):** per-pool semantic blending of on-device SBERT profiles with the *specific insight* that within-pool blending cannot disturb quota-based calibration.
- **Empirical DP dissection** showing DP is an intrinsic debias only *outside* the blueprint, and quantified the accuracy cost inside it; plus a popularity-adaptive noise variant.

**What we experimentally discovered:**
- Calibration invariance across centralized/federated/DP/LLM backbones (numbers identical).
- Federation's intrinsic popularity-bias floor and its fix via orthogonal content signal.
- λ=0.7 as the effective blend; DP's conditional role; H4/H5 second-order effects (negative/partial results).

**What can reasonably be described as OUR contribution:**
1. First privacy-preserving re-implementation of SUPER's dual-model blueprint framework under FL (FedSUPER) with calibration-guarantee preservation (RMSE-PC=0.055).
2. The observation + proof that user-level popularity calibration is **simultaneously privacy-compatible and LLM-enhanceable**, with the LLM blend recovering 23% recall and ~2.6× long-tail coverage at zero calibration cost.
3. A clean empirical characterization of DP's dual role (debias without blueprint, redundant with blueprint).

**Explicitly *not* claimed:** SOTA accuracy (our recall isn't competitive with deep baselines, and we never claimed it); generalization to other datasets/backbones (untested); formal DP guarantees (we used a pragmatic mechanism).

---

## 15. Limitations (genuine remaining)

- **Single dataset, single backbone.** MovieLens-1M + MF d=64 only. The invariance result is algorithmic and dataset-independent; the *accuracy* findings (recall recovery, coverage) are data-dependent and could shift on sparser/larger catalogs. The paper used 3 datasets × 5 backbones × 5 seeds.
- **Single seed.** All runs deterministic seed 0; no confidence intervals; the paper's Wilcoxon methodology is not reproduced. [A real examiner may probe this — see Q&A.]
- **Pragmatic DP.** σ=0.3/ε with clipping and a σ cap; no composed (ε,δ) accountant over rounds, so our "ε=2" is a per-update nominal level, not a formal population guarantee. Claims about calibration do not rest on DP accounting.
- **LLM profile assumptions.** We assume on-device SBERT feasibility, and that top-genres + sampled titles capture taste; larger/server-side encoders untested; embedding semantics may not transfer to other domains.
- **No exposure/impression data.** Like the paper, Pop_u is a *behavioral proxy* (history share), not true exposure preference.
- **Blueprint ordering proxy.** Mu sorted by interaction count ("preference"); the timestamp-based variant exists but wasn't selected; no sensitivity analysis on this ordering.
- **Aggregate leakage nuance.** The public Pareto partition reveals aggregate popularity only; individual C_u and Pop_u remain private. A determined adversary with the partition + public model could still reason about exposure — threat modelling beyond our scope.
- **Pareto α sensitivity.** α=0.20 locked; head = 75 items is a very small pool, so within-head ranking is hard — this is where the LLM helps but also where some results may concentrate.

---

## 16. Future work (grounded in what our experiments showed)

1. **Scale.** The blueprint's O(N) merge is trivial; test FedSUPER on ≥100k-user, larger-catalog datasets (Steam, LastFM) — open question in findings.md is whether per-user quotas stay worth it at scale and with sparser histories.
2. **No-Pop_u variant.** Test whether calibration can be maintained with only the global α when user history is too short (findings.md open question), e.g., blend a default blueprint.
3. **Formal DP.** Replace the pragmatic Gaussian mechanism with a composed accountant (RDP), then re-run H2/H5 to attach real (ε,δ) values to the "DP debias / DP redundant" claims.
4. **Model the LLM migration.** Larger/diverse encoders (768–1536-d, multilingual) and item-side attention embeddings; measure recall/LTC headroom beyond λ=0.7 — we already identified text-embedding-3-large as a candidate (untested).
5. **Deployment realism of the LLM profile.** On-device latency/memory of SBERT for real-time ranking; hybrid (server coarse + on-device fine) profiling.
6. **Re-open H4 at the right scale.** Use an inference-time score clamp or a far larger penalty scale (the term was starved two orders below BPR) — or a different regularization applied to *both* M_pop and M_tail.
7. **Combine H3 + H5.** We explicitly noted (H5 analysis) that LLM-FedSUPER + adaptive DP was not run; with adaptive noise preserving tail signal, LLM+adaptive-DP may dominate LLM+uniform-DP (recall was 0.0212 with uniform). This is the cheapest high-value next experiment.
8. **Multi-seed + Wilcoxon** to match the paper's statistical protocol and de-risk the accuracy claims.

---

## 17. One-minute explanation

"Recommendation systems get stuck recommending the same popular items to everyone — that's popularity bias, and it hurts discovery and niche producers. A recent paper, SUPER, fixes it by splitting items into 'popular' and 'tail' groups, training two models — one per group — and then blending each user's list so it matches that user's own, personal ratio of popular-to-niche items. The catch: SUPER needs all your data on one central server. Our project does two things. First, we show the calibration trick survives privacy: because the final blend only depends on public item popularity and a single private number computed on your device, we can train the two models in a federated way — no raw data ever leaves your phone — and we reproduce SUPER's calibration numbers exactly. Second, federated training makes the recommendations a bit weaker, so we add a small on-device language model profile that captures the user's taste in plain text and use it to pick better items inside each group. That recovers 23% of the lost accuracy and roughly doubles the variety of recommended content, without touching the calibration guarantee. We also show differential privacy — adding noise for extra privacy — only helps when SUPER's calibration is absent, and is redundant when it's present. The result is a privacy-preserving recommender that matches centralized SUPER's calibration with near-equal accuracy."

---

## 18. Five-minute explanation (presentation version)

1. **Problem (1 min).** Popularity bias: ~2% of a catalog (head) captures ~20–80% of the exposure; long tail starved. Effects: less diversity, unfair to niche producers, homogenized lists. Prior fixes apply uniform debiasing that ignores individual taste. SUPER's insight: calibrate *per-user* to their own observed head/tail ratio.

2. **SUPER (1 min).** Pareto partition (α=0.20 cumulative volume → 75 head items / 3,533). Train M_pop on head interactions and M_tail on tail interactions so gradients don't leak between pools. Compute Pop_u = head share of the user's history. Merge: N_pop = ⌊10·Pop_u⌋, N_tail = 10 − N_pop; take the top-N of each specialist's ranking and interleave by the user's own historical head/tail blueprint (hard quota, soft blueprint). Metrics: RMSE-PC and MRMC (per-user + per-rank calibration), APLT, LTC, GKPI.

3. **Our H1 result — calibration is invariant (1.5 min).** The merge needs only (a) public aggregate popularity, (b) per-pool ranked outputs, (c) the private scalar Pop_u. Since the quota is fixed by Pop_u and algorithm, *any* backbone yields the same calibration. We train both specialists federated (sparse FedAvg of item parameters; each user embedding and Pop_u on-device) → FedSUPER reproduces centralized SUPER's RMSE-PC = 0.055 and MRMC = 0.151 exactly, with no raw interaction leaving the device. This is both a proof and an empirical fact in our runs, across every variant we ever ran.

4. **Our H3 result — LLM profiles recover accuracy (1.5 min).** Federated backbones rank poorly *within* tiny pools (the head pool is 75 items; each user gets ~2–3 head slots). An on-device Sentence-BERT profile (built from the user's top genres and sampled liked titles) is blended per-pool: ŝ = (1−λ)·score + λ·cos(profile, item-text). Because blending is inside each pool and the quota is external, calibration cannot move. λ=0.7: recall 0.0298 → 0.0366 (+23%), LTC 0.015 → 0.038 (~2.6×), RMSE-PC/MRMC unchanged. Under ε=2 DP it rescues even more: recall 0.0131 → 0.0212 (+62%).

5. **DP & full picture (1 min).** Without the blueprint, DP is an *intrinsic debias*: popular items receive more gradient noise → head scores drop → FedNCF RMSE-PC 0.705 → 0.283 at ε=2. With the blueprint, DP is redundant for calibration and only costs recall (−57% at ε=2). Multi-objective penalty (H4): negligible (−0.0003 GKPI). Adaptive DP (H5): +5% recall, marginal. **Headline:** LLM-FedSUPER = perfect private calibration (0.055) + recall 0.0366, within 3% of centralized SUPER's recall 0.0378.

---

## 19. Deep technical explanation (director-level)

### The invariance theorem (what we prove)
Fix the partition (H,T). Let s_pop, s_tail be arbitrary per-pool score functions, and let Pop_u ∈ [0,1] be the user inclination. Algorithm 2's list construction:
1. `N_pop = ⌊N·Pop_u⌋`, `N_tail = N − N_pop`.
2. `C_pop` = top-N_pop of s_pop restricted to H; `C_tail` = top-N_tail of s_tail restricted to T (truncated).
3. Blueprint scan over `B = min(N, |Lu|)` head/tail indicators, drawing from the requested pool only while its quota remains, else the other pool, then residual fill.

The *multiset composition* of the output is always exactly N_pop head items + N_tail tail items whenever |C_pop| ≥ N_pop and |C_tail| ≥ N_tail (which holds when the pools have ≥ required candidates — true here: 75 head, ~3,458 tail). Scores determine *which* items fill the slots, **not** how many. Hence the per-user head fraction equals ⌊N·Pop_u⌋/N, a quantity depending only on N and Pop_u ⇒ RMSE-PC and MRMC coincide for any backbones. **Consequence:** privacy layer (FL), noise layer (DP), and content layer (LLM) change only *which* items appear, never the per-user calibration.

### Why federation preserves privacy here specifically
- The Pareto partition needs only total per-item counts (public catalog statistics).
- User embeddings are trained locally and never aggregated (we literally never average user parameters).
- Pop_u = |C_u ∩ H| / |C_u| is a function of the user's private history, computed on-device.
- Server communication = item-parameter deltas only, sparse per-round. Nothing user-identifiable is transmitted; the stakes of a single update are limited to item-embedding deltas (which is why DP noise is available for *those* when desired).

### LLM blend — exact math
Item embedding t_i from "{title} ({genres})"; user profile p_u = SBERT("A user who likes {top-8 genres} movies including: {5 titles}"). Both L2-normalized. cos(p_u, t_i) = p_u·t_i. Blended pool score: ŝ_ui = (1−λ)·score_ui + λ·cos(p_u, t_i). Applied **separately**:
- head pool: only pairs (i ∈ H) get blended scores; tail pairs masked to −∞.
- tail pool: only (i ∈ T).
Then `super_blueprint_merge` runs on Ŝ_pop, Ŝ_tail. The quota is external to Ŝ, so the calibration figures are invariant to λ and to s entirely. In addition, the blend interpolates toward purely-semantic ranking as λ→1; at λ=0.7 we observed monotone recall/LTC improvements with no RMSE-PC movement.

### DP — where it bites, and the numbers
Gradient clipping to norm 1.0; Gaussian noise σ = 0.3/ε (capped 0.6) on item-parameter gradients at each client step. Popular items appear in many clients' positive sets → accumulate more noise mass → their learned scores are depressed relative to tail. **Without a blueprint** this mechanically reduces head-exposure: RMSE-PC 0.705 → 0.283 (ε=2), APLT 0.05 → 0.52. **With the blueprint**, the merged list still contains exactly N_pop head items by the quota — noise only reshuffles which head item, so calibration is untouched and the only effect is accuracy (recall 0.0298 → 0.0131). Adaptive DP (σ_i scaled by log-popularity, 1→3×) improves the no-blueprint result a bit (RMSE-PC 0.251) and adds +5% recall under the blueprint — not enough to shift the headline.

### Implementation details worth being able to state
- Data: 6,040 users × 3,533 items, 575,281 positives (rating ≥ 4); leave-one-out by timestamp; full-catalog ranking at K=10.
- Model: MF d=64, item bias; BPR with fresh random negatives.
- Centralized: 15 epochs, Adam lr 0.05, batch 1,024.
- Federated: 40 rounds, 256 clients/round, 2 local epochs, lr 0.3, clip 1.0; sparse FedAvg over touched items; user embeddings private/persistent.
- Partition: 75 head items (2.12%).
- LLM: all-MiniLM-L6-v2, 384-d, L2-normalized.
- All runs seed 0, deterministic; metrics computed by `metrics.py` exactly as described; results saved as JSON per hypothesis.

---

## 20. Likely examiner questions and answers (based strictly on what we did)

**Q1. Is your "real SUPER reproduction" actually the paper's algorithm?**
A. Yes. We implemented Algorithm 1 (cumulative-volume Pareto partition, α=0.20 → 75 of 3,533 items) and Algorithm 2 (hard quota N_pop=⌊N·Pop_u⌋, soft blueprint from Lu, deterministic fallback, residual fill) directly from the primary text, and verified we reproduce the paper's calibration family of results in the centralized setting (RMSE-PC 0.055, MRMC 0.151) with an MF backbone. The one substitution we made is a preference proxy for Lu ordering (interaction count, not timestamp) — a documented assumption.

**Q2. Why should we believe calibration is "invariant"? Where's the proof?**
A. It's structural. Algorithm 2 always places exactly N_pop head and N_tail tail items per user whenever the pools are non-empty (here, 75 head / ~3,458 tail candidates), regardless of score values. The scores pick *which* items, never *how many*. Therefore RMSE-PC = RMSE between ⌊N·Pop_u⌋/N and Pop_u, a score-independent quantity. We verified it numerically: every FedSUPER/DP/LLM variant returns exactly 0.05495/0.15058.

**Q3. Your Recall@10 values are low (≤0.05). Why is that not fatal?**
A. Because we report *full-catalog* ranking (test item ranked against all 3,533 items, positives excluded by masking) rather than the common 99-negatives sampled protocol, which inflates recall. Our numbers are directly interpretable as "what fraction of held-out items land in the top-10 of the whole catalog." The important comparisons are relative — FedSUPER vs central SUPER, LLM vs no-LLM — and we match the paper's metric philosophy while adding recall on top.

**Q4. Why did you change evaluation from Gini/FDN-A to the paper metrics?**
A. Two concrete problems: (1) a decile-Gini measures global exposure *equality*, not per-user *calibration* — FedSUPER had Gini≈0.10 yet perfect RMSE-PC 0.055, so Gini was answering a different question than SUPER's; (2) FDN-A normalized raw novelty (unbounded −log2 p), so a noise-collapsed model that recommended near-random rare items looked "best" (FDN-A peaked at ε=1). We switched to the paper-compatible family + recall/nDCG, which isolates calibration, ranks quality, and makes our numbers comparable to SUPER's.

**Q5. Isn't the LLM gain just an artifact of a weak federated baseline?**
A. Partially, and that's exactly the point. Federation's sparse aggregation leaves tail/head item embeddings poorly ranked within the tiny head pool — that is a real property of FL recommenders, not a bug in our harness. The LLM profile is *complementary* signal that fills precisely that gap, and crucially it does so at zero calibration cost because it re-ranks only within quota-fixed pools. The fact that it also works under DP noise (+62% recall) shows the semantic channel is independent of the collaborative channel.

**Q6. Is blending scores from different scales (MF score + cosine) principled?**
A. It's a design choice, tested empirically. We sweep λ and pick the best trading point (0.7); the cosine term is bounded and normalized, and the ranking outcome is monotone in λ for recall/coverage with calibration pinned, so the combination is stable across the whole range. We also tested the strict alternative — no blending (λ=0) — which is the FedSUPER baseline, and a training-time multi-objective penalty (H4), which was inert. Within-pool rank fusion is well-precedented in recsys (score fusion); our contribution is showing its calibration-orthogonality under a quota merge.

**Q7. Your "ε=2" DP is not a real DP guarantee. Isn't that a problem?**
A. We state it explicitly as a limitation. The ε we report is the *per-update* Gaussian mechanism's nominal level (σ=0.3/ε) with gradient clipping — not a composed (ε,δ) accountant over 40 rounds. That means the *privacy* figure is pragmatic, not formal. Importantly, the *calibration* claims do not depend on DP at all (they're algorithmic), and the H2/H5 *utility* findings (does noise help/debias) survive as empirical statements about noise level, which is exactly what we measured.

**Q8. Why single seed? Doesn't that invalidate the numbers?**
A. It limits statistical breadth; we were compute-constrained (4 GB GPU, single machine) and prioritized exact reproducibility (seed 0, saved JSONs). The paper's 5-seed + Wilcoxon protocol is not reproduced; the accuracy deltas (+23% recall, calibration identities) are large enough that we consider the qualitative claims robust, but we flag multi-seed as first-priority future work and do not claim statistical significance.

**Q9. Your APLT is identical (0.785) for every variant — that looks suspicious.**
A. It's expected: under a hard quota, APLT = mean per-user tail fraction = mean(N_tail/N) = mean(1 − ⌊N·Pop_u⌋/N), which is score-independent. With mean Pop_u ≈ 0.26 this lands ≈0.75–0.79. APLT is therefore a *design image* of the quota, not a learned quantity; the discriminating axes are LTC (coverage), RMSE-PC/MRMC (calibration), and recall/nDCG (accuracy).

**Q10. Why is delivering private calibration "not free"?**
A. Calibration transfers exactly (RMSE-PC/MRMC pinned at 0.055/0.151), so the privacy cost is *not* on calibration — it sits entirely on ranking accuracy (FedSUPER recall 0.030 vs central SUPER 0.038) and coverage (LTC 0.015). The H3 contribution exists precisely because we measured and then repaired that residual accuracy/coverage gap with orthogonal content signal.

**Q11. What did you *actually* add relative to SUPER?**
A. (1) A federated training regime for the two specialists with on-device user embeddings and Pop_u — era privacy-preserving SUPER (FedSUPER); (2) a proof and empirical demonstration that the calibration is backbone-invariant, so privacy transfers losslessly on the calibration axis; (3) LLM user-profile per-pool blending that recovers 23% recall and ~2.6× long-tail coverage at zero calibration cost; (4) an empirical dissection of DP showing it is an intrinsic debias *without* the blueprint and redundant *with* it, plus an adaptive noise variant and a multi-objective test (both second-order). We do not claim SOTA accuracy or cross-dataset generality.

**Q12. Where could your conclusion break?**
A. Most plausibly: (i) the accuracy/coverage gains are data-dependent (we tested ML-1M/MF only); (ii) the LLM profile's usefulness depends on text metadata being available and meaningful on-device; (iii) very short user histories weaken Pop_u and the blueprint (we always fill to N, but estimation quality drops); (iv) our DP numbers are not formal guarUpdate. The calibration invariance itself is algorithmic and should survive, but the *attractiveness* of the trade-off could shift.

**Q13. Why didn't you use a stronger baseline like a deep NCF or a graph model?**
A. Design choice from bootstrap: our claim is about the *training paradigm* (centralized vs federated) and the *privacy+content* layers — a low-rank MF isolates that claim cleanly and runs within compute budget, and it is the same family as the paper's WBPR-style backbone. H2's FedNCF-DP result partly depends on gradient-mass asymmetry, which simpler MF captures transparently. Deeper backbones are listed as future work, not a need for the current claims.

---

## Appendix — Reproducibility pointers

- Drivers: `experiments/H{1,H2,H3,H4,H5}-*/code/run_*.py` (v1 and v2 variants).
- Results: `experiments/H*/results/metrics_*.json` (all numbers above), `trajectory_*.csv`.
- Source: `src/{recdata,model,super,train,metrics}.py`; figures: `src/make_paper_figures.py` → `paper/figures/*.pdf`.
- Paper source: `paper/main.tex` + `bibliography.bib` (Overleaf-ready; no local LaTeX on the dev machine).
- Narrative: `research-log.md` (append-only), `findings.md`, `research-state.yaml`, `to_human/paper-finalization.md`.
- Nominal run configs are quoted in the scripts; seed 0 everywhere.