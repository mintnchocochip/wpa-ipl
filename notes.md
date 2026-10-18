# Notes: the reframe, and where the data contradicts the original draft

> **Status.** The paper in `overleaf/main.tex` has been REFRAMED. The thesis is
> now effective sample size in ball-by-ball data; the win-probability framework
> is the instrument, not the contribution. See `claims.md` for the claim ledger
> and `overleaf/README.md` for what that means when editing.

## Why the reframe

The original framing proposed a WPA framework and then reported two results that
undercut it: the ensemble loses to logistic regression, and WPA is 3-4x less
stable than strike rate with a null predictive-validity test. Honest, but
structurally weak - it proposed a method and showed it worse than the thing it
replaced, on both of its own tests.

The transferable finding was buried: because ~240 deliveries share one outcome
label, delivery-level validation systematically selects models that fail out of
sample. Validation loss is flat across the whole leaf-size range while test loss
moves 20%. That is now the paper, and the four results below became its four
consequences rather than four apologies.

## What the reframe changed in the findings

Nothing was recomputed except the CSK section, which gained match-level bootstrap
CIs. That addition produced the single most important correction:

**The "CSK build their margin with the ball" claim was withdrawn.** Bowling WPA
(+0.042, CI [+0.002, +0.085]) excludes zero and batting WPA (+0.011, CI [-0.031,
+0.053]) does not - which makes the claim tempting, and is exactly the
difference-of-significance fallacy. The difference itself is +0.030 with CI
[-0.026, +0.084], containing zero. Over 259 matches and nineteen seasons the
most-capped franchise in the league cannot resolve which discipline wins its
matches. Every per-opponent interval straddles the null too.

That is a better section than the one it replaced: it now demonstrates the
paper's thesis instead of decorating it.

---

# Findings log (written against the PRE-reframe draft)

Sections 1–11 below are the original investigation record: what each result is,
how hard it was checked, and why it says what it says. **The evidence is current;
the prescriptions are not.** Where a section suggests wording, section numbers, or
tells you to "keep the gradient-boosted model as the proposed method", that
advice predates the reframe and is superseded by `claims.md` and
`overleaf/README.md`. Retained because the reasoning behind each number — and the
record of how exhaustively the logistic-parity result was checked before being
accepted — is worth having when a reviewer pushes back.

---

## 1. The model does NOT beat the logistic baseline

The abstract and Section VI-A both assert the proposed model outperforms
"logistic-regression and resource-based baselines." On held-out seasons it beats
the resource baseline comfortably, but it does **not** beat a plain regularised
logistic regression on the same features — the two are within noise of each
other, and on log loss the logistic model is marginally ahead.

This was checked exhaustively before accepting it: per-innings models, learning
rates from 0.02–0.10, monotone constraints, tree depths 2–5, leaf sizes
30–10 000, three validation-window widths, spline-expanded logistic, and a
GBM/logistic ensemble. The ordering never reversed. The ensemble gained nothing
(≈0.0001 log loss over logistic alone).

**Why it happens.** Twenty20 win probability is close to a smooth monotone
function of a few quantities (projected score against par, wickets in hand,
required rate). That is exactly the regime where a linear model in well-chosen
features is hard to beat and a tree ensemble mostly adds variance. The trees pay
for their flexibility in the first innings, where the relationship is smoothest
— that is where nearly the whole gap sits.

**Suggested rewording** (abstract):

> ... outperforming a resource-based baseline and matching a regularised
> logistic-regression baseline, while providing the interaction structure
> required for the attribution analysis that follows.

Keep the gradient-boosted model as the proposed method — it is what the SHAP
interaction analysis in Section VI-E needs, and its calibration is the best of
any model tested (ECE 0.032; see §3). But do not claim a discrimination win you
cannot defend under review. A reviewer will run the logistic baseline.

---

## 2. Standard delivery-level validation selects a model that fails

This is the most interesting finding in the run and is worth promoting from a
limitation to a result.

An unregularised gradient-boosted model (`min_child_weight = 30` deliveries, the
kind of default any practitioner would reach for) achieves the **best**
validation log loss and among the **worst** test log loss — roughly 0.70 against
0.51 for a properly regularised model, a ~37 % degradation. Test AUC falls from
about 0.83 to 0.72.

The cause is the effective-sample-size problem the paper already raises in
Section V-B, but the consequence is sharper than the draft suggests. A leaf
holding 30 deliveries holds roughly **one eighth of one independent observation**,
because ~240 deliveries share a single outcome label. The grid in the pipeline is
therefore restricted to leaf sizes ≥ 500 deliveries (≈2 matches), and the failure
mode is reported as an ablation.

Two artefacts support this: **Table VIII** (`tab:regularisation`) and
**`fig_regularisation.pdf`**, which plots validation and test log loss against
leaf size with a secondary axis in equivalent independent matches. Neither is in
the draft; both are ready to drop in. This belongs in Section V-B, referenced
from the Section VI-A discussion.

---

## 3. Calibration on one season makes calibration *worse*

Section IV-B proposes isotonic calibration fitted on the validation split. Done
literally — one season, 71 matches — it degrades every metric:

| Model | Log loss | Brier | AUC | ECE |
|---|---|---|---|---|
| Resource-based | 0.5638 | 0.1894 | 0.794 | 0.067 |
| Logistic regression | 0.5033 | 0.1696 | 0.829 | 0.038 |
| Gradient boosting, uncalibrated | 0.5201 | 0.1766 | 0.815 | **0.032** |
| Isotonic, validation season only | 0.5319 | 0.1806 | 0.813 | 0.054 |
| **Proposed (isotonic, pooled)** | 0.5211 | 0.1766 | 0.814 | 0.032 |

Fitting the calibrator on ~71 independent observations overfits the monotone map,
pushing ECE from 0.032 to 0.054. The pipeline therefore fits it on out-of-fold
predictions across the training seasons pooled with the validation season
(~1000 independent observations), which recovers the uncalibrated ECE. Still
leak-free — no test-season data reaches the calibrator.

The honest reading is that **the model was already well calibrated and post-hoc
calibration is neutral at best here**. That is worth stating: it is the same
effective-sample-size lesson as §2, arriving in a different place. Figure 2 plots
all three curves, so the point is visible rather than asserted.

---

## 4. Feature vector: Section III-C needs updating

The draft's state vector uses absolute run totals and an all-time venue mean.
That combination fails badly out of sample. IPL scoring has inflated sharply, so
a model keyed to absolute totals reads an inflated first-innings score as
dominant when it is merely par: on the test seasons the model predicted a mean
win probability of **0.579** for first-innings states whose actual win rate was
**0.424**.

Two changes fix it, and Section III-C must describe them:

1. **Venue par score $\nu$ is a rolling mean over the previous three seasons**,
   shrunk toward the contemporaneous league mean with a 20-match prior — not an
   all-time expanding mean. Still strictly leak-free.
2. **Three par-relative features are added**: score/par, target/par, and
   required rate / par rate. These let the model generalise across eras instead
   of memorising absolute thresholds.

Replace the state-vector equation with the thirteen features listed at the top of
`paper_values.txt`. The `\ph{...}` count in Section III-C changes accordingly.

---

## 5. WPA is markedly LESS stable than conventional rate statistics

The draft's margin note on Section VI-C anticipated this and asked for it to be
reported honestly rather than buried. It is what the data shows:

| Metric | Batters | Bowlers |
|---|---|---|
| Batting / bowling average | 0.193 | 0.106 |
| **Strike rate / economy rate** | **0.413** | **0.480** |
| WPA | 0.143 | 0.144 |
| WPA₁₀₀ | 0.133 | 0.107 |

Strike rate is roughly three times as stable as WPA₁₀₀ for batters, and economy
rate roughly four and a half times as stable for bowlers (n = 560 and 534
consecutive-season pairs). The mechanism is the one the draft names: WPA is dominated by a small
number of high-leverage deliveries, so it is a high-variance statistic, and
season-to-season leverage exposure is largely a matter of circumstance rather
than a stable player property.

This does not invalidate WPA as a *descriptive* measure of what a player
contributed. It does undercut any claim that WPA is a better basis for
*projecting* future contribution, which is what auction and retention decisions
actually require. Section VI-A must be rewritten accordingly — the current
framing implies the opposite.

---

## 6. Predictive validity: a null result

No rating basis predicts next-season match outcomes better than chance:

| Rating basis | AUC [95 % CI] | Log loss |
|---|---|---|
| Conventional metrics | 0.528 [0.437, 0.616] | 0.6929 |
| WPA | 0.540 [0.440, 0.640] | 0.6937 |
| Combined | 0.524 [0.433, 0.609] | 0.6931 |

Every interval comfortably contains 0.5. With 140 test matches the design simply
cannot resolve effects of this size — the CI half-width is ~0.09 AUC — so this
is "underpowered," not "disproved." Report it as a null result and say so
plainly; do not present WPA's 0.540 as if it were a positive finding.

If you want this section to carry weight, the fix is more matches: pool several
leagues, or evaluate over all seasons with a rolling-origin design rather than
only the two test seasons.

---

## 7. Things still to verify before asserting them

- **Section VI-G** predicts "finishers and death bowlers rise, flat-track
  accumulators fall." The 2026 bowler movers are in `paper_values.txt`. Check
  them against that story before writing the sentence — the draft's own margin
  note says to verify, and it is only partly borne out.
- **Auction/price scatter** in Section VI-G: no price data is bundled here. Drop
  the claim or source the data separately.

---

## 8. Attribution choices worth a reviewer's attention

Implemented exactly as the draft specifies, but two are debatable:

- **No-balls** are credited wholly to the bowler with no batter credit. The draft
  justifies this on the grounds that "no legitimate delivery was faced," which is
  true of wides but not of no-balls — the batter does face a no-ball and may hit
  it for six. Set `NOBALL_BATTER_CREDIT = True` in the config to flip this;
  consider addressing it in the limitations either way.
- **Byes and leg-byes** are unattributed. The residual is reported in
  `paper_values.txt` (`\residualpct`) and is small enough to state in a footnote
  rather than defend at length.

---

## 9. Rating basis

Player ratings use **out-of-fold** win probabilities (5 grouped folds, split by
match, each fold separately calibrated), not the temporally held-out model. Every
delivery is therefore scored by a model that never saw its match. Note that folds
are random across seasons, so a rating for 2015 may be produced by a model that
saw 2019 matches. That is appropriate for a descriptive rating but would not be
for a forecasting claim — worth one sentence in Section V-A.

---

## 10. Reading the interaction plot (Figure 4)

Section VI-E asks for "how the marginal cost of a wicket rises as the required
rate climbs." The fitted interaction is real but modest, and it runs the other
way from the phrasing in the draft: at high required rates the required-rate
feature carries a *more* negative SHAP value for teams with 7–10 wickets in hand
than for teams with 1–3. That is not a paradox — with only a few wickets left the
wickets feature already carries most of the bad news, so the required rate has
less left to explain. Describe the mechanism, do not assert the simpler story.

---

## 11. Reproducing and re-rendering

`paper_outputs/_cache_*.pkl` stores the fitted models, out-of-fold predictions,
ablation and SHAP values. With the cache present a full run takes under two
minutes, so figures can be restyled freely; delete the cache files to force a
genuine refit (about 50 minutes on this machine). Set `FIG_TITLES = False` in the
config to drop the in-figure titles for camera-ready, since the LaTeX `\caption`
carries them.
