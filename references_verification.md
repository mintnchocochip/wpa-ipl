# Reference verification

Checked against publisher records on 2 September 2026. **One reference was
wrong and has been corrected in `main.tex`.**

---

## Corrected

| Key | Was (wrong) | Now (verified) |
|---|---|---|
| `saikia2016` → **`saikia2013`** | "Perfomance based market valuation of cricketers in Indian Premier League," *Journal of Advances in Management Research*, vol. 13, no. 3, pp. 259–273, 2016 | "Performance based market valuation of cricketers in IPL," *Sport, Business and Management: An International Journal*, vol. 3, no. 2, pp. 127–146, 2013 |

Wrong journal, wrong year, wrong volume, wrong issue, wrong pages, and a typo in
the title ("Perfomance"). The citation key was renamed to `saikia2013` and the
single `\cite` updated. **DOI: [10.1108/20426781311325069](https://doi.org/10.1108/20426781311325069)**

---

## Verified correct as written

| Key | Reference | DOI |
|---|---|---|
| `field2007` | C. A. Field and A. H. Welsh, "Bootstrapping clustered data," *JRSS-B*, vol. 69, no. 3, pp. 369–390, 2007 | [10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x) |
| `roberts2017` | D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017 | [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881) |
| `stern2016` | S. E. Stern, "The Duckworth–Lewis–Stern method…," *JORS*, vol. 67, no. 12, pp. 1469–1480, 2016 | [10.1057/jors.2016.30](https://doi.org/10.1057/jors.2016.30) |
| `davis2015` | J. Davis, H. Perera, T. B. Swartz, "A simulator for Twenty20 cricket," *ANZJS*, vol. 57, no. 1, pp. 55–71, 2015 | [10.1111/anzs.12109](https://doi.org/10.1111/anzs.12109) |
| `lemmer2002` | H. H. Lemmer, "The combined bowling rate…," *SAJRSPER*, vol. 24, no. 2, pp. 37–44, 2002 | no DOI (AJOL: [ajol.info/index.php/sajrs/article/view/25839](https://www.ajol.info/index.php/sajrs/article/view/25839)) |
| `sharma2013` | S. K. Sharma, "A factor analysis approach in performance analysis of T-20 cricket," *J. Reliability and Statistical Studies*, vol. 6, no. 1, pp. 69–76, 2013 | no DOI (ISSN 0974-8024) |

---

## Verified with a caveat

| Key | Status | DOI |
|---|---|---|
| `kumar2018` | **Now fully confirmed.** Authors, title, venue, year *and* pages 343-347 all check out - my original entry was correct. | [10.1109/ICSCCC.2018.8703301](https://doi.org/10.1109/ICSCCC.2018.8703301) |
| `vistro2019` | Authors, title, journal, vol. 8, issue 9 (2019) confirmed. Pages 985-990 still unconfirmed. | none - IJSTR does not issue DOIs |

---

## DOIs now printed in the paper

Six references carry a clickable `doi:` link in the bibliography, via
`hyperref` with `hidelinks`. These are exactly the six verified against
publisher records this session:

`field2007` - `roberts2017` - `stern2016` - `davis2015` - `saikia2013` - `kumar2018`

No DOI was added to any reference I have not verified. Adding a DOI from memory
would be the same class of error as the Saikia entry, so the remaining entries
print without one rather than with a guess.

---

## Not machine-verified — canonical works, high confidence

These were not checked this session. All are well-known and the details are
standard, but if you want a clean bill of health, spot-check volume/page numbers:

`kish1965` (Kish, *Survey Sampling*, Wiley, 1965 — book, no DOI) ·
`lindsey1963` · `tango2007` (book) · `yurko2019` · `cervone2016` ·
`decroos2019` · `dl1998` · `bailey2006` · `chen2016` · `lundberg2017` ·
`brier1950` · `platt1999` · `niculescu2005` · `cricsheet` (web resource)

Known DOIs worth confirming rather than trusting from memory:

- `dl1998` — 10.1057/palgrave.jors.2600524
- `decroos2019` — 10.1145/3292500.3330758
- `chen2016` — 10.1145/2939672.2939785
- `cervone2016` — 10.1080/01621459.2016.1141685
- `yurko2019` — 10.1515/jqas-2018-0010
- `lindsey1963` — 10.1287/opre.11.4.477
- `brier1950` — 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

These seven DOIs are from memory and are **not** in the paper. If you confirm
any of them on the publisher's page, append it to that `\bibitem` in the same
form as the six already there:

```latex
doi: \href{https://doi.org/10.1145/2939672.2939785}{10.1145/2939672.2939785}.
```

One caution: `brier1950`'s DOI contains angle brackets, which need percent-encoding
(`%3C`, `%3E`) inside a URL and each `%` then needs escaping as `\%` in LaTeX.
It is the fiddliest of the set and the least worth the trouble.

---

## Still outstanding

**Search for prior cricket/IPL win-probability-added work.** Not done. The paper
no longer claims novelty for WPA — novelty now rests on the effective-sample-size
finding — so the exposure is much lower than it was. But an uncited direct
predecessor remains the most damaging omission available to a reviewer, and this
is the one task on the list that a literature search rather than a citation check
will settle.
