"""Cross-check main.tex against the pipeline's own outputs, and enforce the
internal-consistency rules that a reframe is most likely to break.

  1. numbers    -- every headline figure asserted in prose exists in the output
  2. tables     -- table bodies match the generated LaTeX row for row
  3. banned     -- stale framing that survives a reframe ("we propose", "outperform")
  4. spine      -- abstract, contributions and conclusion cover the same findings
                   in the same order
  5. hedging    -- numbers drawn from small samples sit next to an interval

Run:  python -X utf8 verify_paper.py
"""
import io, re, os, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
tex = io.open(os.path.join(HERE, "overleaf", "main.tex"), encoding="utf-8").read()
txt = io.open(os.path.join(HERE, "paper_outputs", "paper_values.txt"), encoding="utf-8").read()
gen = io.open(os.path.join(HERE, "paper_outputs", "paper_values.tex"), encoding="utf-8").read()
fails = []

# ---------------------------------------------------------------- 1. numbers
CHECKS = [
    ("deliveries",        "286{,}621",  "286,621"),
    ("matches",           "1{,}195",    "1,195"),
    ("raw matches",       "1{,}243",    "1243 matches"),
    ("raw deliveries",    "295{,}732",  "295,732"),
    ("season range",      "2007/08--2026", "2007/08--2026"),
    ("gbm logloss",       "0.5211",     "0.5211"),
    ("gbm brier",         "0.1766",     "0.1766"),
    ("gbm ece",           "0.032",      "0.032"),
    ("logistic logloss",  "0.5033",     "0.5033"),
    ("resource logloss",  "0.5638",     "0.5638"),
    ("isotonic-only ll",  "0.5319",     "0.5319"),
    ("phase i1 pp",       "0.6537",     "0.6537"),
    ("phase i2 death",    "0.2088",     "0.2088"),
    ("stab sr bat",       "0.413",      "0.413"),
    ("stab wpa100 bat",   "0.133",      "0.133"),
    ("stab econ bowl",    "0.480",      "0.480"),
    ("stab wpa100 bowl",  "0.107",      "0.107"),
    ("pv wpa auc",        "0.540",      "0.540"),
    ("pv conv auc",       "0.528",      "0.528"),
    ("residual",          "1.07",       "1.07"),
    ("oof logloss",       "0.548",      "0.5476"),
    ("abl leaf30 test",   "0.6125",     "0.6125"),
    ("abl leaf5000 test", "0.5110",     "0.5110"),
    ("shap crr",          "0.496",      "0.4960"),
    ("shap wickets",      "0.363",      "0.3630"),
    ("leverage share",    "10.0",       "10.0%"),
    ("leverage mass",     "17.9",       "17.9%"),
    ("csk matches",       "259",        "259"),
    ("csk winpct",        "56.0",       "56.0"),
    ("csk bat wpa",       "+0.011",     "+0.011"),
    ("csk bowl wpa",      "+0.042",     "+0.042"),
    ("csk diff",          "+0.030",     "+0.030"),
    ("case swing 1",      "20.7",       "20.7"),
    ("case swing 3",      "29.8",       "29.8"),
]
for name, in_tex, in_gen in CHECKS:
    a, b = in_tex in tex, (in_gen in txt or in_gen in gen)
    if not (a and b):
        fails.append(f"[number] {name}: in main.tex={a}, in generated={b}")

# The forward check above only proves an expected number appears SOMEWHERE, so a
# corrupted instance slips through while the original survives elsewhere. Invert
# it: every corpus-scale figure written in the prose must be one the pipeline
# actually produced.
ALLOW_COUNTS = {
    "5000", "2000", "1000",   # hyperparameters and round prose figures
    "17000",                  # "roughly 17,000 deliveries per season", rounded in prose
}
# Compare digits-only on both sides: the generated files write thousands as
# 236{,}047 while the plain-text dump writes 236,047.
haystack = re.sub(r"[{,}]", "", txt + gen)
prose = re.sub(r"(?m)^\s*%.*$", "", tex).split("\\begin{thebibliography}")[0]
for m in re.finditer(r"\d{1,3}\{,\}\d{3}", prose):
    plain = m.group(0).replace("{,}", "")
    if plain in ALLOW_COUNTS or plain in haystack:
        continue
    ctx = re.sub(r"\s+", " ", prose[max(0, m.start()-70):m.end()+70])
    fails.append(f"[number] stray count {m.group(0)!r} not produced by the pipeline"
                 f"\n     ...{ctx}...")

# ----------------------------------------------------------------- 2. tables
def rows_of(src, label):
    m = re.search(r"\\label\{" + label + r"\}(.*?)\\bottomrule", src, re.S)
    if not m:
        return None
    return [re.sub(r"\s+", " ", r).strip()
            for r in m.group(1).split("\\\\")
            if "&" in r and "textbf{" not in r.split("&")[0]]

for label in ["tab:h2h", "tab:cskplayers", "tab:divergence", "tab:regularisation"]:
    rt, rg = rows_of(tex, label), rows_of(gen, label)
    if rt is None or rg is None:
        fails.append(f"[table] {label}: missing from {'main.tex' if rt is None else 'generated'}")
        continue
    norm = lambda rs: [re.sub(r"[\$\\ ]|textbf|\{|\}", "", r).replace("\u2212", "-") for r in rs]
    nt, ng = norm(rt), norm(rg)
    if len(nt) != len(ng):
        fails.append(f"[table] {label}: {len(nt)} rows in main.tex vs {len(ng)} generated")
    else:
        for x, y in zip(nt, ng):
            if x != y:
                fails.append(f"[table] {label} row mismatch\n     tex: {x}\n     gen: {y}")

# ----------------------------------------------------------------- 3. banned
# Framing that belonged to the pre-reframe paper, or that overclaims. Comments
# and the bibliography are excluded: cited titles legitimately contain some.
body = re.sub(r"(?m)^\s*%.*$", "", tex)
body = body.split("\\begin{thebibliography}")[0]
BANNED = [
    (r"we propose a[^.]{0,40}framework", "pre-reframe framing"),
    (r"\bthis paper proposes\b", "pre-reframe framing"),
    (r"\boutperform", "unsupported: logistic beats the GBM"),
    (r"\bsuperior to\b", "overclaim"),
    (r"\bstate-of-the-art\b", "overclaim"),
    (r"\bfirst to\b", "novelty claim not defensible for WPA"),
    (r"\bnovel metric\b", "novelty claim not defensible for WPA"),
    (r"\bProposed \(calibrated\)", "stale table label"),
    (r"\bour proposed model\b", "pre-reframe framing"),
    (r"decisively outperforming", "overclaim"),
]
for pat, why in BANNED:
    for m in re.finditer(pat, body, re.I):
        ctx = re.sub(r"\s+", " ", body[max(0, m.start()-60):m.end()+60])
        fails.append(f"[banned] {m.group(0)!r} ({why})\n     ...{ctx}...")

# ------------------------------------------------------------------ 4. spine
# The four consequences must appear in the same order in the abstract, the
# contributions list and the conclusion. This is what catches drift between
# the three places a reader forms an impression.
TAGS = [
    ("SELECT", r"model.selection|selection failure|silent"),
    # the noun, not the adjective: "a calibrated model" is the instrument,
    # "calibration" is the finding
    ("CALIB",  r"\bcalibration\b"),
    ("STAB",   r"stab(le|ility)|persistent"),
    ("POWER",  r"power|resolve|underpowered|0\.09"),
]
def seq(segment):
    hits = []
    for tag, pat in TAGS:
        m = re.search(pat, segment, re.I)
        if m:
            hits.append((m.start(), tag))
    return [t for _, t in sorted(hits)]

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
contrib = re.search(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", tex, re.S)
concl = re.search(r"\\section\{Conclusion\}(.*?)\\begin\{thebibliography\}", tex, re.S)
segs = {"abstract": abstract, "contributions": contrib, "conclusion": concl}
orders = {}
for nm, m in segs.items():
    if not m:
        fails.append(f"[spine] could not locate {nm}")
        continue
    orders[nm] = seq(m.group(1))
if len(orders) == 3:
    ref = orders["abstract"]
    for nm, o in orders.items():
        if o != ref:
            fails.append(f"[spine] {nm} order {o} != abstract order {ref}")
    missing = [t for t, _ in TAGS if t not in ref]
    if missing:
        fails.append(f"[spine] abstract does not cover: {missing}")

# ---------------------------------------------------------------- 5. hedging
# Any sentence asserting a CSK head-to-head quantity must sit near an interval.
HEDGE = r"\[|\bCI\b|interval|straddl|not distinguishable|underpowered|suggestive|point estimate"
SMALL_N = r"(56\.0|48\.8|66\.7|\+0\.042|\+0\.011|\+0\.030)"
# Scope to the sentence, not a character window: a ±400 char window finds an
# unrelated "interval" elsewhere in the same subsection and passes anything.
prose_flat = re.sub(r"\s+", " ", body)
sentences = re.split(r"(?<=[.:;])\s+(?=[A-Z\\])", prose_flat)
for i, sent in enumerate(sentences):
    if re.search(SMALL_N, sent) and not re.search(HEDGE, sent, re.I):
        nxt = sentences[i + 1] if i + 1 < len(sentences) else ""
        if not re.search(HEDGE, nxt, re.I):
            fails.append(f"[hedge] small-sample figure asserted without an interval"
                         f"\n     {sent.strip()[:170]}")

# ------------------------------------------------------- 6. figure staleness
# The bundled figures are copies. If the pipeline regenerates one and the copy
# is not refreshed, the paper compiles cleanly while a caption describes a plot
# the reader is not looking at -- which is exactly what happened once.
figdir = os.path.join(HERE, "overleaf", "figures")
srcdir = os.path.join(HERE, "paper_outputs")
digest = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for fn in sorted(os.listdir(figdir)) if os.path.isdir(figdir) else []:
    if not fn.endswith(".pdf"):
        continue
    src = os.path.join(srcdir, fn)
    if not os.path.exists(src):
        fails.append(f"[stale] {fn} has no counterpart in paper_outputs/")
    elif digest(src) != digest(os.path.join(figdir, fn)):
        fails.append(f"[stale] {fn} differs from the pipeline output "
                     f"-- re-copy paper_outputs/*.pdf into overleaf/figures/")

# ------------------------------------------------------------------- report
print(f"checked {len(CHECKS)} numbers, 4 tables, {len(BANNED)} banned patterns, "
      f"spine order, small-sample hedging")
if fails:
    print("\nPROBLEMS:")
    for f in fails:
        print("  -", f)
else:
    print("RESULT: paper is internally consistent with the pipeline output")
sys.exit(1 if fails else 0)
