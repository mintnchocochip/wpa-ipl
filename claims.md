# Claim ledger

Every load-bearing claim in the paper, where it appears, what supports it, and
how strong that support is. **If you weaken a claim, weaken it in every row of
the "Appears in" column** — most self-contradiction is a claim that survived in
one place after being softened in another.

`verify_paper.py` enforces the mechanical parts of this (numbers, table rows,
banned phrasing, section ordering, small-sample hedging). This file covers what
a script cannot judge: whether the strength of the wording matches the strength
of the evidence.

Legend — **Strength**: `SOLID` = interval excludes the null or the quantity is
exact; `DIRECTIONAL` = point estimate clear, interval not computed or wide;
`NULL` = interval contains the null and the paper says so; `ARGUED` = reasoning,
not measurement.

---

## The four consequences (the spine)

| # | Claim | Appears in | Evidence | Strength |
|---|---|---|---|---|
| 1 | Unregularised ensemble: best validation loss, ~20% worse test loss | Abstract, Contrib 1, §VI-B, Conclusion | `tab:regularisation`, `fig:regularisation` (0.4984 val / 0.6125 test vs 0.5078 / 0.5110) | SOLID (single split; see caveat below) |
| 1a | Validation loss flat over the range test loss moves | Abstract, §VI-B, Conclusion | Val range 0.4880–0.5078 (0.02) vs test 0.5110–0.6125 (0.10) | SOLID |
| 1b | A 30-delivery leaf ≈ ⅛ of one observation | §I, §VI-B | 286,621 / 1,195 ≈ 240 | SOLID (arithmetic) |
| 2 | Isotonic on one season worsens ECE 0.032 → 0.054 | Abstract, Contrib 2, §VI-C, Conclusion | `tab:wp` rows 3 and 4 | SOLID |
| 2a | The ensemble was already well calibrated | §VI-C | ECE 0.032, best in `tab:wp` | SOLID |
| 3 | WPA 3–4× less persistent than strike rate / economy | Abstract, Contrib 2, §VI-D, Conclusion | `tab:stability` (0.133 vs 0.413; 0.107 vs 0.480) | SOLID |
| 3a | 10.0% of deliveries carry 17.9% of \|Δ\| mass | Abstract, §VI-D, Conclusion | `paper_values.txt` leverage block | SOLID |
| 3b | Two mechanisms: sampling variance **and** circumstance | §VI-D, §VII-C | — | ARGUED — **do not** collapse to one; see note below |
| 4 | 140 test matches cannot resolve effects < ~0.09 AUC | Abstract, Contrib 2, §VI-E, Conclusion | `tab:validity` CI widths | SOLID |
| 4a | No rating basis beats chance on next-season outcomes | §VI-E | `tab:validity`, all CIs contain 0.5 | NULL (stated as such) |

**Caveat on #1** worth pre-empting in review: the ablation is one train/val/test
split on one league. The *direction* is unambiguous and the mechanism is
arithmetic, but the 20% figure is a single realisation. §VII-B ("Single
competition") carries this; do not upgrade the wording elsewhere.

**Note on #3b** — this is the subtlest consistency trap in the paper. If the
stability result is described as *purely* sampling noise, then shrinkage would
fix it and WPA would be salvageable as a projection tool, which contradicts
§VI-D's conclusion and §VII-A's recommendation. Both mechanisms must stay named
and distinct. §VII-C ("Future Work") is worded to match: shrinkage mitigates the
sampling component "though not the circumstantial component."

---

## The instrument

| Claim | Appears in | Evidence | Strength |
|---|---|---|---|
| Beats the resource baseline | Abstract (implicit), §VI-A | `tab:wp` (0.521 vs 0.564) | SOLID |
| Does **not** beat logistic regression | §I, §VI-A | `tab:wp` (0.521 vs 0.503; AUC 0.814 vs 0.829) | SOLID — reported as a draw, never as a win |
| Ensemble retained for interactions + calibration, not accuracy | §IV-A, §VI-A | — | ARGUED |
| Era normalisation is necessary | §III-C, §VI-F | Predicted 0.579 vs realised 0.424 first-innings; SHAP 0.836 vs 0.505 | SOLID |
| Team decomposition is exact | §IV-E, §VI-I | Max departure from zero = 0.0 over 1,195 matches | SOLID (identity) |
| Attribution residual is 1.07% | §IV-D | `paper_values.txt` | SOLID |

---

## The CSK demonstration

**This section is where the paper is most exposed**, because it is descriptive
and every quantity is a small-sample mean. Wording must stay inside the
intervals.

| Claim | Evidence | Strength |
|---|---|---|
| Win rate 56.0% | CI [49.8, 62.0] | DIRECTIONAL — interval nearly touches 50 |
| Bowling WPA +0.042 | CI [+0.002, +0.085] | SOLID, barely |
| Batting WPA +0.011 | CI [−0.031, +0.053] | NULL |
| **Bowling contributes more than batting** | diff +0.030, CI [−0.026, +0.084] | **NULL — claim withdrawn** |
| Every per-opponent record distinguishable from even | all CIs straddle 50% | **NULL — none are** |
| Five of six career leaders are bowlers | `tab:cskplayers` | DIRECTIONAL (descriptive, no CI) |

> **The withdrawn claim.** An earlier draft asserted CSK's margin was "built
> roughly four-to-one with the ball." Bowling WPA excludes zero and batting WPA
> does not, which makes the claim tempting — but that is the
> difference-of-significance fallacy. The difference itself is +0.030 [−0.026,
> +0.084] and contains zero. §VI-I now states this explicitly and says the
> earlier claim was drafted before the interval was computed. **Do not
> reintroduce it**, in the abstract, a talk, or a figure caption.

---

## Claims deliberately *not* made

Listed so nobody helpfully adds them back:

- WPA is a better player-valuation metric than strike rate. (§VI-D refutes it.)
- The framework/metric is novel. (WPA dates to 1963; §I says so.)
- The gradient-boosted model is the best available specification. (It is not.)
- The divergence in `tab:divergence` shows players are mispriced. (No price data;
  §VI-H declines the inference.)
- The effective-sample-size result is demonstrated beyond cricket. (§VII-B calls
  the generalisation an argument, not a demonstration.)
- Any individual player in `tab:divergence` is over- or under-rated. (Single
  season, high variance; §VI-H says read as illustrative.)

---

## Before submitting

1. `python -X utf8 wpa_pipeline.py && python -X utf8 check_tex.py && python -X utf8 verify_paper.py`
2. **Verify the three new references** — `kish1965`, `field2007`, `roberts2017`
   were added from memory and their volume/page details have not been checked
   against the sources.
3. **Search for prior cricket/IPL win-probability-added work.** The paper no
   longer claims novelty for WPA, which reduces the risk, but an uncited direct
   predecessor is still the most damaging omission available to a reviewer.
4. Read abstract → contributions → conclusion in one pass. The script checks
   order; only a human catches tone drift.
