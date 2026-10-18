# Overleaf package — The Delivery Is Not the Observation (IEEE conference)

## Uploading

Upload `overleaf_wpa_ipl.zip` via **New Project → Upload Project** on Overleaf.
Everything compiles as-is; no extra files are needed.

- Compiler: **pdfLaTeX** (Menu → Compiler)
- Main document: **`main.tex`** (Overleaf detects it automatically)
- No BibTeX run required — references are a `thebibliography` environment
- `IEEEtran.cls` is preinstalled on Overleaf, so it is not bundled here

## Contents

```
main.tex           the full paper, all values filled in
figures/           seven vector PDFs, referenced via \graphicspath
README.md          this file
```

| File | Figure | Where it appears |
|---|---|---|
| `fig_pipeline.pdf` | 1 | §IV, the win-probability instrument |
| `fig_regularisation.pdf` | 2 | §VI-B, the model-selection failure |
| `fig_reliability.pdf` | 3 | §VI-C, calibration |
| `fig_shap_summary.pdf` | 4 | §VI-F, global attribution |
| `fig_shap_dependence.pdf` | 5 | §VI-F, RRR × wickets interaction |
| `fig_wp_trace.pdf` | 6 | §VI-G, case study |
| `fig_h2h_csk.pdf` | 7 | §VI-I, CSK head-to-head (`figure*`, spans both columns) |

Ten tables are inline in `main.tex`. All floats are numbered automatically by
order of appearance and every cross-reference uses `\ref`, so moving one will
not break numbering.

## What this paper argues

The thesis is **not** "here is a better player metric." It is that in
ball-by-ball data the delivery is not the unit of information — 286,621
deliveries carry 1,195 independent outcomes — and that ignoring the factor of
240 between them corrupts four stages of a pipeline: model selection,
calibration, rating stability, and validation power. The win-probability model
is the instrument used to demonstrate this, not the contribution.

This matters for editing. Several statements that would be natural in a
player-valuation paper are now **actively contradictory**, and the verification
script rejects them. In particular:

- the gradient-boosted model does **not** outperform logistic regression, and
  the paper says so twice on purpose;
- WPA is **less** stable than strike rate, which is a finding, not an admission;
- CSK's bowling contribution is **not** established as larger than their
  batting (difference +0.030, 95% CI [−0.026, +0.084]).

See `claims.md` in the parent directory for the full ledger of what is claimed,
at what strength, and what is deliberately not claimed.

## Before you submit

Three placeholders remain. They render **in red**:

1. Manikandan S — e-mail address
2. Vijayaraj J — e-mail address
3. Code repository URL (§V-F, "Implementation")

Two further tasks that no script can do:

4. **Verify three references.** `kish1965`, `field2007` and `roberts2017` were
   added from memory; their volume and page details have not been checked
   against the sources.
5. **Search for prior cricket/IPL win-probability-added work.** The paper no
   longer claims novelty for WPA, which reduces the exposure, but an uncited
   direct predecessor remains the most damaging omission available to a reviewer.

## Verification

From the parent directory:

```bash
python -X utf8 wpa_pipeline.py && python -X utf8 check_tex.py && python -X utf8 verify_paper.py && python -X utf8 test_guardrails.py
```

- `check_tex.py` — braces, environments, dangling refs/cites, missing figures
- `verify_paper.py` — 34 numbers and 4 table bodies against the pipeline output,
  plus banned phrasing, abstract/contributions/conclusion ordering, and
  small-sample hedging
- `test_guardrails.py` — deliberately corrupts the paper in memory to confirm
  each guardrail actually fires, then restores it

Figure titles are drawn inside the images *and* repeated in the `\caption`. For
camera-ready set `FIG_TITLES = False` in `wpa_pipeline.py` and re-run (~2 min
with the cache present) to regenerate title-free PDFs.
