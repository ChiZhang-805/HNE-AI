# HNE: implemented personal-reference warning model

Implementation snapshot: 10 November 2026. This document describes `hne_features.py`, `hne_additive.py`, `hne_set.py` and `configs/development.json`. HNE predicts a sustained large fluctuation or directional departure 24–72 hours ahead from personal history (H), indicator relationships (N), and observed environmental/behavioral context (E).

## 1. Original-paper gap

| Primary source | Established method | HNE question left open |
|---|---|---|
| [FCM, 2024, Sections 3.1–3.5](https://arxiv.org/html/2410.12206v1) | Normal-data forecasting and observation/forecast reconstruction produce an advance anomaly score. Its future context is a generated forecast. | Can measured behavioral/environmental context and an individual's reference relationships improve supervised prediction of a persistent future departure? |
| [CATCH, ICLR 2025, Sections 3.2–3.5](https://arxiv.org/html/2410.12261v3) | Frequency patches and channel masks reconstruct observed point/subsequence anomalies. | Does change in a personal reference network predict events before the abnormal target window is observed? |
| [PatchTST, ICLR 2023, Section 3](https://arxiv.org/pdf/2211.14730), [iTransformer, ICLR 2024, Section 3](https://arxiv.org/html/2310.06625v4) | Shared channel encoders and variate attention provide efficient temporal and cross-variable representations. | Which personally normalized temporal and relational departures carry warning information with few labeled people? |
| [LIFT, ICLR 2024, Sections 3–4](https://arxiv.org/html/2401.17548v2) | Estimates locally stationary lead–lag relations to improve forecasting. | Can departure from the person's usual lagged relations serve as a precursor even when those relations are no longer stable? |
| [TimeXer, NeurIPS 2024, Section 3](https://arxiv.org/html/2402.19072v4) | Distinguishes endogenous and exogenous variables in attention-based forecasting. | Is an unexplained departure after adjustment for usual context responses more predictive than the raw departure? |
| [Moirai, ICML 2024, Section 3.1](https://arxiv.org/html/2402.02592v2), [Timer-XL, ICLR 2025, Section 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/d09820e292f915d77fa717d228c425ee-Paper-Conference.pdf) | Any-variate and long-context forecasting address heterogeneous inputs and variable counts. | Do shared personal-relative warning rules transfer between domains with different measured indicators? |
| [TiDE, TMLR 2023, Section 4 and Appendix A](https://arxiv.org/html/2304.08424v4) | A compact dense model uses covariates; a linear analogue has guarantees under stated linear-system assumptions. | A structured low-capacity rule is a credible small-data candidate; its superiority for rare-event warning still requires testing. |

These are task-specific research gaps, not failures proved by the cited authors. The proposed contribution is explicit comparison with personal reference dynamics and their usual context response, followed by a shared prospective risk rule. Neither a graph alone nor variable-count support is claimed as novel. Direct event BCE targets the warning outcome; the implemented candidates do not use an auxiliary trajectory-MSE objective.

## 2. Observations and personal reference

At issue time t, the input contains only records available by t. The separately specified event label Y_it indicates a qualifying onset in (t+24h,t+72h]; unknown outcomes are excluded. The current development adapters reuse the existing labels, sample origins and first 672-hour calibration reference B_i. Labels are never predictor inputs.

For state j, the reference supplies the hour-of-day profile mu_ij(h) and scale s_ij. Define z_ij(s) = [x_ij(s) - mu_ij(hour(s))] / s_ij. Normalize the context with its reference center and scale. Missing observations remain NaN during feature extraction. Recent windows have lengths L in {24,72,168} hours. The same formulas and support thresholds serve both current datasets; state-node count can vary.

The current feature extractor uses one context series. The set head also accepts multiple context descriptors through its shared context encoder, but the current single-context extractor is not an implemented end-to-end multi-context adapter.

### Reference context response

Form c_i(s) = [1,e_i(s),e_i(s-6h),e_i(s-24h)]. For each state node, fit ridge regression using complete context/target reference observations, with at least 24 usable rows:

\[
 \widehat\gamma_{ij}=\arg\min_\gamma
 \sum_{s\in B_i:\mathrm{observed}}
 \left[\operatorname{clip}(z_{ij}(s),-12,12)
       -\bar c_i(s)^T\gamma\right]^2
 +10\|\gamma_{1:}\|_2^2,
\]

where context values in bar c are clipped to [-8,8] and the intercept is unpenalized. Monitoring residuals are u_ij(s) = z_ij(s) - bar c_i(s)^T gamma_ij. The 28-day reference residuals are computed out of fold over four contiguous seven-day blocks: predict each block using a ridge fit on the other three, then use the full-reference fit for monitoring. If context or the fitted response is unsupported, retain the raw residual; record the coefficient norm and reference support fraction. This is observational adjustment, not a causal effect estimate.

### Eight temporal descriptors

For any recent series v_W and its reference v_B, let b = median(v_B), s = max(SD(v_B),0.1), a = max(Q90(v_B)-Q10(v_B),0.2). Compute:

1. Signed bias [mean(v_W)-b]/s and its absolute value.
2. Log amplitude ratio log[1+(Q90(v_W)-Q10(v_W))/a].
3. Log dispersion ratio log[1+SD(v_W)/s].
4. Tail fraction beyond max(Q90(|v_B-b|),0.1).
5. Difference between second-half and first-half means, divided by s.
6. Personal-relative autocorrelation changes at 1h and 24h, using the reliability rule below.

This gives eight scalars. A window needs at least eight observed values and its reference at least 24; the half-window drift needs four observations in each half. For each L, calculate all eight descriptors for both z and u. Concatenate the 48 temporal values with the norm of the three context-response coefficients and reference fitting support, yielding a 50-dimensional node descriptor h_ij.

### Reliability-weighted relational change

For each window, lag ell in {0,6h,24h}, and state pair, estimate ordinary Pearson correlations rho_W and rho_B on observed pairs. Require at least 12 pairs and nonzero variance for both estimates. Use each unordered pair once at lag zero and directed pairs at positive lags. Let n_W be the recent pair count:

\[
 r_W={n_W-3\over n_W-3+24},\qquad
 \Delta\rho=r_W(\rho_W-\rho_B),\qquad
 \widetilde\rho_W=\rho_B+\Delta\rho.
\]

This shrinks a poorly supported recent relation toward that person's reference graph. It does not independently shrink rho_W and rho_B toward zero, which would introduce a change solely from differing support. The same delta rule defines the two autocorrelation-change descriptors.

For each lag, pool available edges into four values: reliability-weighted mean |rho_tilde_W|, weighted mean |delta_rho|, maximum |delta_rho|, and weighted signed mean delta_rho. Three lags times three window lengths produce 36 graph descriptors. Calculate them on z for N and on context-adjusted u for NE. No future graph, attention map, or per-edge learned message is used.

### Current tensor interface

For every scalar node descriptor, `pool_nodes` takes observed mean, standard deviation, maximum and mean of the largest two values (one value if only one is present). All-missing pools remain NaN.

| Input block | Contents | Width |
|---|---|---:|
| H | 24 raw temporal node descriptors, each with four set summaries | 96 |
| N | Raw relation summaries: 3 windows x 3 lags x 4 statistics | 36 |
| E | 24 adjusted temporal descriptors plus coefficient norm/support, pooled into 104 values; 24 context temporal descriptors | 128 |
| NE | Adjusted relation summaries: 3 windows x 3 lags x 4 statistics | 36 |
| Total pooled input | [H,N,E,NE] | 296 |
| Node input | 48 temporal descriptors plus coefficient norm/support | 50 per node |

The pooled width is independent of the number and ordering of state nodes. The current 296-coordinate representation supersedes the earlier unimplemented 138-coordinate proposal.

## 3. Two implemented risk rules

### A. HNE additive

Fit training-only median imputation, standardization, clipping to [-8,8], and quadratic B-splines with three knots and no bias basis. Standardize the spline outputs, then divide every coordinate in block g by sqrt(d_g), where d_g is that block's expanded width. Fit L2 logistic regression with C in {0.01,0.1,1.0}. With beta defined on the unscaled standardized spline basis, this is equivalent to a group-size-weighted ridge penalty proportional to sum_g d_g ||beta_g||². The penalty has one C; there is no search over separate group penalties.

The risk coefficients act on invariant descriptor meanings, not sensor identities. The representation and fitted preprocessing are reusable across different state-node counts. Missing-value imputation is specific to this candidate and is fitted only on training people; it is not described as explicit mask learning.

### B. HNE relational set head

For each of the 50 node coordinates, fit one training-only observed-value scaler shared across nodes. Add its observation mask, giving 100 inputs to a shared MLP: Linear(100,16), GELU, Linear(16,16). Masked mean/max over nodes yields 32 values. Nodes with no observed recent temporal descriptor are excluded from this pool.

Separately encode the 72 N/NE summaries plus masks through a 144-to-16 MLP. Encode each context's 24 temporal descriptors plus masks through a shared 48-to-16 MLP; masked mean/max across contexts and a linear/GELU projection yield 16 values. Concatenate node, relation and context embeddings with their three support fractions. A 67-to-16-to-1 MLP outputs the risk logit. All hidden layers have width 16. The current model has 7,169 trainable parameters.

This is a relational **set head over precomputed graph summaries**, not learned per-edge message passing. Context affects both node residuals and a separate dynamic-context branch. Supported masks and scalers are frozen at inference. For q context variables, the set-head interface expects E = 104 node summaries + 24q context descriptors; its prediction API accepts the target group map.

Train with AdamW, learning rate 0.001, batch size 128, gradient clipping 1.0, at most 60 epochs, and patience eight. Select weight decay from {0.001,0.01}. The two values are the complete neural candidate search.

### Shared supervised weighting and probability correction

Let a_it = 1/n_i give each training person equal total weight. Define the positive multiplier

\[
 c={\sum_{it:Y=0}a_{it}\over\sum_{it:Y=1}a_{it}},\qquad
 w_{it}\propto a_{it}\,c^{Y_{it}},\quad \sum_{it}w_{it}=N.
\]

Both candidates minimize event BCE with these person- and class-balanced weights; A adds its ridge penalty and B uses AdamW weight decay. The probability returned for validation or prediction is

\[
 \widehat p_{it}=\sigma(f_\theta(it)-\log c).
\]

Equivalently, B adds log[p_train/(1-p_train)] to its balanced logit, with p_train computed under a_it. This removes the deliberate training class-prior shift; it is not a guarantee of empirical calibration. B handles a single-class training partition with its constant training-prior prediction.

Epoch and configuration selection use ordinary validation AUROC when both classes occur and unbalanced validation log loss otherwise. B additionally records person-weighted metrics. The config string mentioning `balanced_log_loss_if_one_class` is stale: both implemented selection functions use unbalanced log loss.

## 4. Research questions and evaluation

**RQ1: Which changes precede the event?** Compare reference-relative temporal descriptors and raw/adjusted relation changes; report the predictive contribution of each block and its uncertainty over people. The hypothesis fails if relation changes add no held-person information or improvements are reproduced by missingness alone.

**RQ2: Does combining H, N and E improve warning?** Compare full HNE, no-network (remove N/NE), no-environment (remove E/NE), and H-only ablations, with the same labels and origins. History-LR remains a required strong supervised comparator. History-HGB and HNE-HGB compare nonlinear readouts on history and the complete descriptor table. The research target is at least +0.04 absolute AUROC over the strongest development-selected comparator, not merely improvement over DNB.

**RQ3: Do the rules transfer?** Freeze source-trained preprocessing and risk parameters, then evaluate a held domain using only each target person's unlabeled reference calibration. Separate this from fitting the same code independently in each domain. The set head supports variable node counts; completed zero-shot transfer is an experimental result, not implied by that property.

Current development uses RESILIENT and BrisT1D-Open, whose earlier folds were already viewed. For outer fold k, validate on (k+1) mod 5, train on the other three folds, and evaluate on k. Splits contain whole people. Select A's C or B's decay inside that validation partition. Three seeds are configured (20261110–20261112); the current development runner uses the first seed, so three-seed results require their actual additional runs.

The primary aggregate is fold-wise AUROC weighted by the number of positive-negative origin pairs in each fold; resample complete people for paired uncertainty. Report AP and Brier score as complementary metrics. Only newly reserved cohorts support an independent final test. Record the actual fitting trials and compute: A has three C values and B two decay values, not an asserted equal-compute budget.

Native DNB/FCM/CATCH and the previous HNE remain separately labeled reference comparisons. A same-event-label FCM/CATCH readout is a planned matched-supervision comparison, not yet implemented here. Its encoder must be fitted inside each training split; any label-trained first-stage scores used by a second-stage learner require person-wise out-of-fold generation. The main architectural claim must distinguish that comparison from native unsupervised scores.

## 5. English pseudocode

```text
For each person:
    Read the fixed 28-day clock profile, state scales and context normalization.
    Construct raw personal residuals Z and context lags [0h, 6h, 24h].
    Fit reference ridge context responses for each state.
    Compute reference adjusted residuals using four held-out 7-day blocks.
    Use full-reference coefficients to adjust subsequent observations.
    For each allowed prediction origin:
        For windows [24h, 72h, 168h]:
            Compute eight temporal descriptors for raw and adjusted states.
            Estimate recent/reference correlations at lags [0h, 6h, 24h].
            Shrink the correlation change toward zero using recent pair support.
            Pool raw and adjusted graph summaries separately.
            Compute the observed-context temporal descriptors.
        Emit node descriptors [P, 50] and pooled descriptors [296].

For each outer person fold:
    Select the next fold for validation and the other three for training.
    Fit preprocessing using training people only.
    Train additive candidates for C in [0.01, 0.1, 1.0].
    Train relational-set candidates for decay in [0.001, 0.01].
    Balance people and classes during fitting; correct the prediction prior.
    Select by validation AUROC, or log loss for a single-class validation fold.
    Freeze each selected model and predict the held-person fold.

Aggregate paired fold results; quantify uncertainty by resampling whole people.
For transfer, freeze a source model and apply target personal calibration only.
```

The current method needs enough personal history and observed pairs to estimate a useful reference. Its pooled descriptors also discard some domain-specific sensor semantics; transfer therefore remains a substantive empirical question.
