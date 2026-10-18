# =============================================================================
#  Beyond Averages: A Win-Probability Framework for Context-Aware Player
#  Impact Evaluation in Twenty20 Cricket
#
#  SINGLE-CELL REPRODUCIBLE PIPELINE
#  -----------------------------------------------------------------------
#  Downloads Cricsheet IPL ball-by-ball data, builds the match-state table,
#  trains + calibrates the win-probability model and both baselines, runs the
#  regularisation ablation, attributes Win Probability Added (WPA) to batters
#  and bowlers, runs the stability / predictive-validity / SHAP / divergence
#  analyses, and writes every figure and numeric placeholder the paper needs.
#
#  OUTPUTS (in ./paper_outputs/):
#     fig_pipeline.pdf/.png        -> Figure 1
#     fig_reliability.pdf/.png     -> Figure 2
#     fig_shap_summary.pdf/.png    -> Figure 3
#     fig_shap_dependence.pdf/.png -> Figure 4
#     fig_wp_trace.pdf/.png        -> Figure 5
#     fig_regularisation.pdf/.png  -> extra figure for Section V-B
#     paper_values.tex             -> filled LaTeX tables + \newcommand macros
#     paper_values.txt             -> plain-text placeholder -> value map
#     player_ratings_*.csv         -> derived seasonal rating tables
#
#  NOTE ON TWO DELIBERATE DEVIATIONS FROM THE SUBMITTED DRAFT (both are
#  reported in paper_values.txt and must be reflected in the text):
#   (1) The state vector is extended with venue/era-relative features. IPL
#       scoring has inflated sharply; a model keyed to absolute run totals is
#       badly miscalibrated two seasons out.
#   (2) min_child_weight is expressed in DELIVERIES but the effective sample
#       size is MATCHES (~240 deliveries share one label). The grid is
#       therefore restricted to leaf sizes >= 500 deliveries (~2 matches).
#       The unregularised failure mode is reported as an ablation.
# =============================================================================

# ----------------------------- 0. CONFIG -------------------------------------
CONFIG = dict(
    CRICSHEET_URL   = "https://cricsheet.org/downloads/ipl_male_csv2.zip",
    CACHE_ZIP       = "ipl_male_csv2.zip",   # re-used if already present
    OUT_DIR         = "paper_outputs",
    RANDOM_SEED     = 42,
    N_TEST_SEASONS  = 2,      # last k seasons -> test
    N_VAL_SEASONS   = 1,      # season(s) immediately before test -> validation
    EXCLUDE_DLS     = True,   # drop Duckworth-Lewis-Stern affected matches
    EXCLUDE_SUPEROV = True,   # drop matches decided by Super Over (tie)
    PAR_WINDOW      = 3,      # seasons in the rolling venue par-score window
    PAR_SHRINK      = 20,     # pseudo-matches shrinking venue par -> league par
    QUAL_BALLS_BAT  = 120,    # qualification threshold, deliveries faced
    QUAL_BALLS_BOWL = 120,    # qualification threshold, deliveries bowled
    N_BOOTSTRAP     = 500,    # match-level bootstrap replicates
    N_ECE_BINS      = 10,     # equal-frequency calibration bins
    N_OOF_FOLDS     = 5,      # grouped folds for out-of-fold WPA predictions
    SHAP_SAMPLE     = 20000,  # rows sampled for SHAP (speed)
    CAL_CLIP        = 0.01,   # isotonic output clipped to [c, 1-c]
    NOBALL_BATTER_CREDIT = False,  # paper spec: extras attributed to bowler only
    SAVE_PNG        = True,
    SAVE_PDF        = True,
    FIG_TITLES      = True,   # in-figure titles; set False for camera-ready
    USE_CACHE       = True,   # reuse fitted models so figures can be re-rendered
)                             # (delete paper_outputs/_cache_*.pkl to force a refit)

# ----------------------------- 1. IMPORTS ------------------------------------
import importlib, subprocess, sys, os, io, csv, zipfile, urllib.request, warnings, pickle
warnings.filterwarnings("ignore")

for _pkg, _imp in [("numpy","numpy"),("pandas","pandas"),("scikit-learn","sklearn"),
                   ("matplotlib","matplotlib"),("scipy","scipy"),
                   ("xgboost","xgboost"),("shap","shap")]:
    try:
        importlib.import_module(_imp)
    except ImportError:
        print(f"[setup] installing {_pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

import numpy as np
import pandas as pd
import matplotlib

# Inline display inside Jupyter; headless Agg when run as a plain script, so
# that plt.show() never blocks on a GUI window.
try:
    IN_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # noqa: F821
except NameError:
    IN_NOTEBOOK = False
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb; HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    import shap; HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

rng = np.random.default_rng(CONFIG["RANDOM_SEED"])
OUT = CONFIG["OUT_DIR"]; os.makedirs(OUT, exist_ok=True)

# IEEE single-column figure styling (\columnwidth = 3.5in)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
COL_W, COL_H = 3.5, 2.5
C_MAIN, C_ALT, C_ACC = "#1f4e79", "#c0504d", "#4f8f3f"

def cached(name, fn):
    """Compute fn() once and reuse it on later runs, so that figures can be
    re-rendered without refitting. Delete paper_outputs/_cache_*.pkl to refresh."""
    p = os.path.join(OUT, f"_cache_{name}.pkl")
    if CONFIG["USE_CACHE"] and os.path.exists(p):
        print(f"  [cache] reusing '{name}'")
        with open(p, "rb") as fh: return pickle.load(fh)
    v = fn()
    with open(p, "wb") as fh: pickle.dump(v, fh)
    return v

def savefig(fig, stem):
    """Save under the exact filename the LaTeX source expects, then show inline."""
    out = []
    if CONFIG["SAVE_PDF"]:
        p = os.path.join(OUT, stem + ".pdf"); fig.savefig(p); out.append(p)
    if CONFIG["SAVE_PNG"]:
        p = os.path.join(OUT, stem + ".png"); fig.savefig(p); out.append(p)
    print("  [figure] " + "  ".join(out))
    if IN_NOTEBOOK:
        plt.show()          # inline display in Jupyter
    return out

# ----------------------------- 2. DOWNLOAD -----------------------------------
print("\n[1/12] Acquiring Cricsheet ball-by-ball data ...")
zpath = CONFIG["CACHE_ZIP"]
if not os.path.exists(zpath):
    req = urllib.request.Request(CONFIG["CRICSHEET_URL"],
                                 headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        open(zpath, "wb").write(r.read())
    print(f"  downloaded -> {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB)")
else:
    print(f"  using cached {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB)")

zf = zipfile.ZipFile(zpath); names = zf.namelist()

if any(n.endswith("all_matches.csv") for n in names):
    with zf.open([n for n in names if n.endswith("all_matches.csv")][0]) as f:
        bb = pd.read_csv(f, dtype={"match_id": str}, low_memory=False)
else:
    frames = []
    for n in names:
        b = os.path.basename(n)
        if b.endswith(".csv") and not b.endswith("_info.csv") and b[:-4].isdigit():
            with zf.open(n) as f: d = pd.read_csv(f, low_memory=False)
            d["match_id"] = b[:-4]; frames.append(d)
    bb = pd.concat(frames, ignore_index=True)

def parse_info(text):
    d, teams, players, dates = {}, [], {}, []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] != "info" or len(row) < 3: continue
        k, v = row[1].strip(), row[2].strip()
        if   k == "team": teams.append(v)
        elif k == "date": dates.append(v)
        elif k == "player":
            if len(row) >= 4: players.setdefault(v, []).append(row[3].strip())
        elif k in ("registry", "umpire"): continue
        else: d.setdefault(k, v)
    d["teams"], d["players"], d["dates"] = teams, players, dates
    return d

meta_rows, xis = [], {}
for n in names:
    if not n.endswith("_info.csv"): continue
    mid = os.path.basename(n).replace("_info.csv", "")
    info = parse_info(zf.read(n).decode("utf-8", errors="replace"))
    meta_rows.append(dict(match_id=mid, season=info.get("season"),
        date=info["dates"][0] if info["dates"] else None,
        venue_raw=info.get("venue"), city=info.get("city"),
        team1=info["teams"][0] if len(info["teams"]) > 0 else None,
        team2=info["teams"][1] if len(info["teams"]) > 1 else None,
        winner=info.get("winner"), eliminator=info.get("eliminator"),
        outcome=info.get("outcome"), method=info.get("method")))
    xis[mid] = info["players"]

meta = pd.DataFrame(meta_rows)
N_RAW_MATCHES, N_RAW_BALLS = meta.shape[0], bb.shape[0]
print(f"  raw corpus: {N_RAW_MATCHES} matches, {N_RAW_BALLS:,} delivery records")

# ----------------------------- 3. FILTERING ----------------------------------
print("\n[2/12] Filtering ...")
meta["final_winner"] = meta["winner"].where(meta["winner"].notna(), meta["eliminator"])
n_noresult = int((meta["outcome"].fillna("") == "no result").sum())
n_dls      = int(meta["method"].fillna("").eq("D/L").sum())
n_super    = int(meta["eliminator"].notna().sum())

keep = meta["final_winner"].notna()
if CONFIG["EXCLUDE_DLS"]:     keep &= ~meta["method"].fillna("").eq("D/L")
if CONFIG["EXCLUDE_SUPEROV"]: keep &= meta["eliminator"].isna()
meta = meta[keep].copy()

bb = bb[bb["match_id"].isin(set(meta["match_id"]))]
bb = bb[bb["innings"].isin([1, 2])].copy()            # drop Super-Over innings
both = bb.groupby("match_id")["innings"].nunique(); ok = set(both[both == 2].index)
bb   = bb[bb["match_id"].isin(ok)].copy()
meta = meta[meta["match_id"].isin(ok)].copy()
print(f"  excluded: {n_noresult} no-result, {n_dls} D/L, {n_super} Super-Over")
print(f"  retained: {meta.shape[0]} matches, {bb.shape[0]:,} deliveries")

meta["season_year"] = meta["season"].astype(str).str.slice(0, 4).astype(int)
season_labels = (meta[["season", "season_year"]].drop_duplicates()
                   .sort_values("season_year")["season"].tolist())
season_ord_map = {s: i for i, s in enumerate(season_labels)}
meta["season_ord"] = meta["season"].map(season_ord_map)
NS = len(season_labels)

VENUE_ALIAS = {
    "feroz shah kotla": "arun jaitley stadium",
    "punjab cricket association stadium": "is bindra stadium mohali",
    "punjab cricket association is bindra stadium": "is bindra stadium mohali",
    "sardar patel stadium motera": "narendra modi stadium",
    "aca vdca stadium": "dr y s rajasekhara reddy aca vdca cricket stadium",
    "ma chidambaram stadium chepauk": "ma chidambaram stadium",
}
def norm_venue(v):
    if not isinstance(v, str): return "unknown"
    s = " ".join(v.lower().replace(".", " ").split(",")[0].split())
    return VENUE_ALIAS.get(s, s)
meta["venue_n"] = meta["venue_raw"].map(norm_venue)

# ----------------------------- 4. STATE TABLE --------------------------------
print("\n[3/12] Constructing pre-delivery match-state representation ...")
bb = bb.drop(columns=[c for c in ["season", "venue"] if c in bb.columns])
bb = bb.merge(meta[["match_id", "season", "season_ord", "venue_n", "city",
                    "final_winner", "team1", "team2"]], on="match_id", how="left")
bb["seq"] = np.arange(len(bb))
bb = bb.sort_values(["season_ord", "match_id", "innings", "seq"]).reset_index(drop=True)

for c in ["wides", "noballs", "byes", "legbyes", "penalty", "extras", "runs_off_bat"]:
    bb[c] = pd.to_numeric(bb.get(c), errors="coerce").fillna(0.0)

bb["wkt_type"]   = bb["wicket_type"].astype("object").where(bb["wicket_type"].notna(), "")
bb["is_legal"]   = ((bb["wides"] == 0) & (bb["noballs"] == 0)).astype(int)
bb["total_runs"] = bb["runs_off_bat"] + bb["extras"]

NON_WICKET      = {"retired hurt"}
BOWLER_CREDITED = {"bowled", "caught", "lbw", "caught and bowled", "stumped", "hit wicket"}
bb["is_wicket"] = (bb["wkt_type"].ne("") & ~bb["wkt_type"].isin(NON_WICKET)).astype(int)
bb["is_wicket"] = (bb["is_wicket"] + bb["other_wicket_type"].notna().astype(int)).clip(upper=2)

g = bb.groupby(["match_id", "innings"], sort=False)
bb["balls_bowled"] = g["is_legal"].cumsum()   - bb["is_legal"]     # BEFORE the ball
bb["score"]        = g["total_runs"].cumsum() - bb["total_runs"]
bb["wkts_lost"]    = g["is_wicket"].cumsum()  - bb["is_wicket"]
bb["balls_rem"] = (120 - bb["balls_bowled"]).clip(lower=0)
bb["wkts_rem"]  = (10 - bb["wkts_lost"]).clip(lower=0)
bb["over_idx"]  = (bb["balls_bowled"] // 6).clip(upper=19)
bb["phase"]     = np.where(bb["over_idx"] < 6, 0, np.where(bb["over_idx"] < 15, 1, 2))
bb["crr"]       = np.where(bb["balls_bowled"] > 0,
                           6.0 * bb["score"] / bb["balls_bowled"].replace(0, np.nan), np.nan)

inn1_final = (bb[bb["innings"] == 1].groupby("match_id")["total_runs"].sum()
                .rename("inn1_total"))
bb = bb.merge(inn1_final, on="match_id", how="left")
bb["target"]   = np.where(bb["innings"] == 2, bb["inn1_total"] + 1, np.nan)
bb["runs_req"] = np.where(bb["innings"] == 2, bb["target"] - bb["score"], np.nan)
bb["rrr"]      = np.where((bb["innings"] == 2) & (bb["balls_rem"] > 0),
                          6.0 * bb["runs_req"] / bb["balls_rem"].replace(0, np.nan), np.nan)

# --- venue par score: rolling window over STRICTLY PRECEDING seasons ---------
# Expanding all-time means are unusable here: IPL scoring has inflated sharply,
# so an all-time venue mean is stale by the time it reaches the test seasons.
W, K = CONFIG["PAR_WINDOW"], CONFIG["PAR_SHRINK"]
fi = meta[["match_id", "venue_n", "season_ord"]].merge(inn1_final, on="match_id")
idx = pd.MultiIndex.from_product([sorted(fi["venue_n"].unique()), range(NS)],
                                 names=["venue_n", "season_ord"])
vs = (fi.groupby(["venue_n", "season_ord"])["inn1_total"].agg(["sum", "count"])
        .reindex(idx, fill_value=0).reset_index().sort_values(["venue_n", "season_ord"]))
vs["rs"] = vs.groupby("venue_n")["sum"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
vs["rc"] = vs.groupby("venue_n")["count"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
gg = fi.groupby("season_ord")["inn1_total"].agg(["sum", "count"]).reindex(range(NS), fill_value=0)
gg["gs"] = gg["sum"].rolling(W, min_periods=1).sum().shift(1)
gg["gc"] = gg["count"].rolling(W, min_periods=1).sum().shift(1)
vs = vs.merge(gg[["gs", "gc"]], left_on="season_ord", right_index=True, how="left")
overall = float(fi["inn1_total"].mean())
vs["league_par"] = np.where(vs["gc"] >= 8, vs["gs"] / vs["gc"].replace(0, np.nan), overall)
v_raw = np.where(vs["rc"] >= 8, vs["rs"] / vs["rc"].replace(0, np.nan), vs["league_par"])
lam = vs["rc"].fillna(0) / (vs["rc"].fillna(0) + K)          # shrink to league par
vs["par"] = lam * np.nan_to_num(v_raw, nan=overall) + (1 - lam) * vs["league_par"]
bb = bb.merge(vs[["venue_n", "season_ord", "par"]], on=["venue_n", "season_ord"], how="left")
bb["par"] = bb["par"].fillna(overall)
par_rate = bb["par"] / 20.0

bb["rrr2"] = np.where(bb["innings"] == 2, bb["rrr"], np.nan)
bb["tgt2"] = np.where(bb["innings"] == 2, bb["target"], np.nan)
bb["req2"] = np.where(bb["innings"] == 2, bb["runs_req"], np.nan)
bb["score_par"]  = bb["score"] / bb["par"]
bb["rrr_par"]    = bb["rrr2"] / par_rate
bb["target_par"] = bb["tgt2"] / bb["par"]

bb["y"] = (bb["batting_team"] == bb["final_winner"]).astype(int)

FEATURES = ["innings", "balls_rem", "wkts_rem", "score", "crr", "rrr2", "tgt2",
            "req2", "phase", "par", "score_par", "rrr_par", "target_par"]
FEAT_LABEL = {"innings": "Innings index", "balls_rem": "Balls remaining",
    "wkts_rem": "Wickets in hand", "score": "Current score", "crr": "Current run rate",
    "rrr2": "Required run rate", "tgt2": "Target", "req2": "Runs required",
    "phase": "Innings phase", "par": "Venue par score",
    "score_par": "Score / par", "rrr_par": "Required rate / par rate",
    "target_par": "Target / par"}
X_all = bb[FEATURES].astype(float); y_all = bb["y"].values
print(f"  state table: {X_all.shape[0]:,} rows x {X_all.shape[1]} features")

# ----------------------------- 5. TEMPORAL SPLIT -----------------------------
print("\n[4/12] Chronological match-level split ...")
test_ords = list(range(NS - CONFIG["N_TEST_SEASONS"], NS))
val_ords  = list(range(NS - CONFIG["N_TEST_SEASONS"] - CONFIG["N_VAL_SEASONS"],
                       NS - CONFIG["N_TEST_SEASONS"]))
tr_ords   = [i for i in range(NS) if i not in test_ords + val_ords]
so = bb["season_ord"].values
m_tr, m_va, m_te = np.isin(so, tr_ords), np.isin(so, val_ords), np.isin(so, test_ords)
assert not (m_tr & m_va).any() and not (m_va & m_te).any() and not (m_tr & m_te).any()

def span(o):
    L = [season_labels[i] for i in o]
    return L[0] if len(L) == 1 else f"{L[0]}--{L[-1]}"

split_df = pd.DataFrame(
    [dict(split=n_, seasons=span(o), matches=int(bb.loc[m, "match_id"].nunique()),
          deliveries=int(m.sum()))
     for n_, m, o in [("Train", m_tr, tr_ords), ("Validation", m_va, val_ords),
                      ("Test", m_te, test_ords)]]
    + [dict(split="Total", seasons=span(list(range(NS))),
            matches=int(bb["match_id"].nunique()), deliveries=int(len(bb)))])
print(split_df.to_string(index=False))

Xtr, ytr = X_all[m_tr], y_all[m_tr]
Xva, yva = X_all[m_va], y_all[m_va]
Xte, yte = X_all[m_te], y_all[m_te]
gte = bb.loc[m_te, "match_id"].values
inn_te = bb.loc[m_te, "innings"].values

# ----------------------------- 6. METRICS ------------------------------------
def ece(y, p, k=CONFIG["N_ECE_BINS"]):
    q = pd.qcut(pd.Series(p), k, labels=False, duplicates="drop")
    a = (pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
           .agg(n=("y", "size"), o=("y", "mean"), e=("p", "mean")))
    return float((a["n"] / a["n"].sum() * (a["o"] - a["e"]).abs()).sum())

def all_metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                auc=roc_auc_score(y, p), ece=ece(y, p))

_grp_index = None
def boot_ci(y, p, groups, fn, B=CONFIG["N_BOOTSTRAP"], seed=7):
    """Match-level bootstrap: deliveries within a match share one outcome label."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"g": groups, "i": np.arange(len(y))})
    arrs = [v.values for _, v in df.groupby("g")["i"]]
    out = []
    for _ in range(B):
        ii = np.concatenate([arrs[j] for j in r.integers(0, len(arrs), len(arrs))])
        try: out.append(fn(y[ii], p[ii]))
        except Exception: pass
    return ((float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
            if out else (np.nan, np.nan))

# ----------------------------- 7. BASELINES ----------------------------------
print("\n[5/12] Fitting baselines ...")
num = [f for f in FEATURES if f != "phase"]
lr_pipe = Pipeline([
    ("ct", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["phase"])])),
    ("clf", LogisticRegression(C=1.0, max_iter=3000))]).fit(Xtr, ytr)
p_lr = lr_pipe.predict_proba(Xte)[:, 1]

# Duckworth-Lewis-style resource baseline
d1 = bb[m_tr & (bb["innings"] == 1)].copy()
d1["rem_runs"] = d1["inn1_total"] - d1["score"]
d1["bbin"] = (d1["balls_rem"] // 5).astype(int)
rt = (d1.groupby(["wkts_rem", "bbin"])["rem_runs"].mean().unstack("bbin")
        .reindex(index=range(11), columns=range(25))
        .interpolate(axis=1, limit_direction="both").ffill().bfill().fillna(0.0))
def exp_rem(b, w):
    return rt.values[np.clip(np.asarray(w).astype(int), 0, 10),
                     np.clip((np.asarray(b) // 5).astype(int), 0, 24)]
def resource_margin(df):
    proj = df["score"].values + exp_rem(df["balls_rem"].values, df["wkts_rem"].values)
    return np.where(df["innings"].values == 2,
                    proj - df["tgt2"].fillna(overall).values + 0.5,
                    proj - df["par"].values)
res_tr = pd.DataFrame({"m": resource_margin(bb[m_tr]), "i": bb.loc[m_tr, "innings"].values})
res_te = pd.DataFrame({"m": resource_margin(bb[m_te]), "i": inn_te})
p_res = np.zeros(len(res_te))
for i_ in (1, 2):
    a, b_ = res_tr["i"].values == i_, res_te["i"].values == i_
    p_res[b_] = (LogisticRegression(max_iter=1000).fit(res_tr.loc[a, ["m"]], ytr[a])
                   .predict_proba(res_te.loc[b_, ["m"]])[:, 1])

# ----------------------------- 8. PROPOSED MODEL -----------------------------
def make_gbm(p):
    if HAVE_XGB:
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=CONFIG["RANDOM_SEED"],
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, **p)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=p["n_estimators"],
        max_depth=p["max_depth"], learning_rate=p["learning_rate"],
        min_samples_leaf=p["min_child_weight"], random_state=CONFIG["RANDOM_SEED"])

print("\n[6/12] Grid search (leaf sizes restricted by effective sample size) ...")
GRID = [dict(n_estimators=n, max_depth=d, learning_rate=0.05, min_child_weight=m)
        for n in (300, 600) for d in (3, 4) for m in (500, 2000, 5000)]
def _grid():
    b, bll, bp = None, np.inf, None
    for i, prm in enumerate(GRID, 1):
        mm = make_gbm(prm).fit(Xtr, ytr)
        ll = log_loss(yva, np.clip(mm.predict_proba(Xva)[:, 1], 1e-4, 1 - 1e-4))
        print(f"  [{i:2d}/{len(GRID)}] ne={prm['n_estimators']:>3} d={prm['max_depth']} "
              f"mcw={prm['min_child_weight']:>4}   val logloss={ll:.5f}")
        if ll < bll: bll, b, bp = ll, mm, prm
    return b, bll, bp
best, best_ll, best_p = cached("model", _grid)
print(f"  selected: {best_p}  (val logloss {best_ll:.5f})")

c = CONFIG["CAL_CLIP"]
p_gbm_raw = best.predict_proba(Xte)[:, 1]
p_va_raw  = best.predict_proba(Xva)[:, 1]

# --- calibration ---------------------------------------------------------
# Variant A: isotonic fitted on the validation season alone. This is the
# textbook recipe, but a single season is only ~71 INDEPENDENT observations,
# which is far too few to fit a flexible monotone map.
iso_va = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, yva)
p_cal_va = np.clip(iso_va.predict(p_gbm_raw), c, 1 - c)

# Variant B (pre-specified as "Proposed"): isotonic fitted on out-of-fold
# predictions across the TRAINING seasons pooled with the validation season,
# giving ~1000 independent observations instead of ~71. Still leak-free: no
# test-season information enters the calibrator.
print("\n  Building out-of-fold training predictions for calibration ...")
def _caloof():
    o = np.zeros(int(m_tr.sum()))
    Xtr_r, gtr = Xtr.reset_index(drop=True), bb.loc[m_tr, "match_id"].values
    for k, (a, b_) in enumerate(GroupKFold(4).split(Xtr_r, ytr, gtr), 1):
        o[b_] = make_gbm(best_p).fit(Xtr_r.iloc[a], ytr[a]).predict_proba(Xtr_r.iloc[b_])[:, 1]
        print(f"    calibration fold {k}/4 done")
    return o
oof_tr = cached("caloof", _caloof)
iso_pool = IsotonicRegression(out_of_bounds="clip").fit(
    np.concatenate([oof_tr, p_va_raw]), np.concatenate([ytr, yva]))
p_gbm_cal = np.clip(iso_pool.predict(p_gbm_raw), c, 1 - c)

MODELS = {"Resource-based": p_res, "Logistic regression": p_lr,
          "Gradient boosting (raw)": p_gbm_raw,
          "Isotonic, validation season only": p_cal_va,
          "Proposed (calibrated)": p_gbm_cal}
perf = pd.DataFrame({k: all_metrics(yte, v) for k, v in MODELS.items()}).T
print("\n  Held-out test performance:\n", perf.round(4).to_string())
ci = {k: dict(logloss=boot_ci(yte, np.clip(v, 1e-4, 1 - 1e-4), gte, log_loss),
              auc=boot_ci(yte, v, gte, roc_auc_score)) for k, v in MODELS.items()}

ph_names = {0: "Powerplay", 1: "Middle", 2: "Death"}
te_df = bb.loc[m_te, ["innings", "phase"]].copy(); te_df["y"], te_df["p"] = yte, p_gbm_cal
phase_tbl = (te_df.groupby(["innings", "phase"])
    .apply(lambda d: log_loss(d["y"], np.clip(d["p"], 1e-4, 1 - 1e-4), labels=[0, 1]),
           include_groups=False).unstack("phase").rename(columns=ph_names))
print("\n  Log loss by innings x phase:\n", phase_tbl.round(4).to_string())

# --- regularisation ablation (effective sample size) ------------------------
print("\n  Ablation: leaf size vs held-out performance")
def _ablation():
    rows = []
    for mcw in (30, 100, 500, 2000, 5000):
        mm = make_gbm(dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                           min_child_weight=mcw)).fit(Xtr, ytr)
        pv = mm.predict_proba(Xva)[:, 1]; pt = np.clip(mm.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
        rows.append(dict(mcw=mcw, matches=mcw / 240.0,
                         val=log_loss(yva, np.clip(pv, 1e-4, 1 - 1e-4)),
                         test=log_loss(yte, pt), auc=roc_auc_score(yte, pt)))
        print(f"    min_child_weight={mcw:>5} (~{mcw/240:.1f} matches)  "
              f"val={rows[-1]['val']:.4f}  test={rows[-1]['test']:.4f}  auc={rows[-1]['auc']:.3f}")
    return pd.DataFrame(rows)
abl = cached("ablation", _ablation)

# ----------------------------- 9. OUT-OF-FOLD WP -----------------------------
print("\n[7/12] Out-of-fold win probabilities for unbiased WPA ...")
groups = bb["match_id"].values
def _oof():
    o = np.zeros(len(bb))
    for f, (tri, tei) in enumerate(GroupKFold(CONFIG["N_OOF_FOLDS"]).split(X_all, y_all, groups), 1):
        tm = pd.unique(groups[tri]); r = np.random.default_rng(100 + f); r.shuffle(tm)
        cal = set(tm[:max(1, int(0.25 * len(tm)))])   # ~240 matches per calibrator
        is_cal = np.isin(groups[tri], list(cal)); fit_i, cal_i = tri[~is_cal], tri[is_cal]
        mm = make_gbm(best_p).fit(X_all.iloc[fit_i], y_all[fit_i])
        iso_f = IsotonicRegression(out_of_bounds="clip").fit(
            mm.predict_proba(X_all.iloc[cal_i])[:, 1], y_all[cal_i])
        o[tei] = np.clip(iso_f.predict(mm.predict_proba(X_all.iloc[tei])[:, 1]), c, 1 - c)
        print(f"  fold {f}/{CONFIG['N_OOF_FOLDS']} done")
    return o
oof = cached("oof", _oof)
bb["p"] = oof
oof_ll = log_loss(y_all, bb["p"]); print(f"  OOF log loss over full corpus: {oof_ll:.5f}")

# ----------------------------- 10. WPA ATTRIBUTION ---------------------------
print("\n[8/12] Attributing win-probability increments to players ...")
# bb is already ordered by (season, match, innings, delivery) from Section 4.
# Do NOT re-sort here: the split masks in Section 5 are positional, so any
# reordering silently misaligns them against the rows they select.
bb["p_next"] = bb.groupby(["match_id", "innings"], sort=False)["p"].shift(-1)
bb = bb.merge(bb[bb["innings"] == 2].groupby("match_id")["p"].first().rename("p_inn2_open"),
              on="match_id", how="left")
last = bb["p_next"].isna()
# innings boundary: post-state of the final 1st-innings ball is the COMPLEMENT of
# the 2nd-innings opening state, so the ball that sets the target is credited.
pa = bb["p_next"].to_numpy(dtype=float)
pa = np.where(last & (bb["innings"] == 1), 1.0 - bb["p_inn2_open"], pa)
pa = np.where(last & (bb["innings"] == 2), bb["y"].astype(float), pa)
bb["p_after"] = pa
bb["delta"] = bb["p_after"] - bb["p"]

bb["is_wide"]   = bb["wides"] > 0
bb["is_noball"] = bb["noballs"] > 0
bb["is_bye"]    = (bb["byes"] + bb["legbyes"]) > 0
bb["is_runout"] = bb["wkt_type"].isin(["run out", "obstructing the field", "retired out"])
bb["bowler_wkt"] = bb["wkt_type"].isin(BOWLER_CREDITED)

extras_only  = bb["is_wide"] | (bb["is_noball"] & ~CONFIG["NOBALL_BATTER_CREDIT"])
unattributed = bb["is_bye"] & ~extras_only
bat_player = np.where(bb["is_runout"] & bb["player_dismissed"].notna(),
                      bb["player_dismissed"], bb["striker"])
bat_wpa  = np.where(extras_only | unattributed, 0.0, bb["delta"])
bowl_wpa = np.where(unattributed | bb["is_runout"], 0.0, -bb["delta"])

att = pd.DataFrame({"match_id": bb["match_id"], "season": bb["season"],
    "bat_team": bb["batting_team"], "bowl_team": bb["bowling_team"],
    "bat_player": bat_player, "bowler": bb["bowler"],
    "bat_wpa": bat_wpa, "bowl_wpa": bowl_wpa, "delta": bb["delta"],
    "bat_credited": (~(extras_only | unattributed)).astype(int),
    "bowl_credited": (~(unattributed | bb["is_runout"])).astype(int)})
resid = float(np.abs(bb.loc[unattributed, "delta"]).sum() / np.abs(bb["delta"]).sum())
print(f"  unattributed residual (byes/leg-byes): {100*resid:.2f}% of total |WPA| mass")

# --- conventional statistics + ratings --------------------------------------
outs = (bb[bb["player_dismissed"].notna()].groupby(["season", "player_dismissed"])
          .size().rename("outs").reset_index().rename(columns={"player_dismissed": "player"}))
bat_conv = (bb.groupby(["season", "striker"])
              .agg(runs=("runs_off_bat", "sum"), balls=("is_legal", "sum"))
              .reset_index().rename(columns={"striker": "player"})
              .merge(outs, on=["season", "player"], how="left"))
bat_conv["outs"] = bat_conv["outs"].fillna(0)
bat_conv["bat_avg"] = bat_conv["runs"] / bat_conv["outs"].replace(0, np.nan)
bat_conv["sr"] = 100 * bat_conv["runs"] / bat_conv["balls"].replace(0, np.nan)

bb["bowler_runs"] = bb["runs_off_bat"] + bb["wides"] + bb["noballs"]
bowl_conv = (bb.groupby(["season", "bowler"])
               .agg(runs_conceded=("bowler_runs", "sum"), balls=("is_legal", "sum"),
                    wkts=("bowler_wkt", "sum")).reset_index().rename(columns={"bowler": "player"}))
bowl_conv["bowl_avg"] = bowl_conv["runs_conceded"] / bowl_conv["wkts"].replace(0, np.nan)
bowl_conv["econ"] = 6 * bowl_conv["runs_conceded"] / bowl_conv["balls"].replace(0, np.nan)

bat_r = (att.groupby(["season", "bat_player"]).agg(wpa=("bat_wpa", "sum"), n=("bat_credited", "sum"))
           .reset_index().rename(columns={"bat_player": "player"}))
bowl_r = (att.groupby(["season", "bowler"]).agg(wpa=("bowl_wpa", "sum"), n=("bowl_credited", "sum"))
            .reset_index().rename(columns={"bowler": "player"}))
for d in (bat_r, bowl_r): d["wpa100"] = 100 * d["wpa"] / d["n"].replace(0, np.nan)

BAT  = bat_r.merge(bat_conv, on=["season", "player"], how="left")
BOWL = bowl_r.merge(bowl_conv, on=["season", "player"], how="left")
for d in (BAT, BOWL): d["season_ord"] = d["season"].map(season_ord_map)
BAT.to_csv(os.path.join(OUT, "player_ratings_batting.csv"), index=False)
BOWL.to_csv(os.path.join(OUT, "player_ratings_bowling.csv"), index=False)

# --- head-to-head team analysis, centred on Chennai Super Kings --------------
# Because WPA is denominated in match outcome, a team's per-match total is
# nearly fixed by the result (+0.5 won / -0.5 lost). The informative quantity is
# therefore the SPLIT between batting and bowling: it says how a rivalry was won.
print("\n[8b/12] Head-to-head analysis (Chennai Super Kings) ...")
TEAM_ALIAS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
SHORT = {"Chennai Super Kings": "CSK", "Mumbai Indians": "MI",
         "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR",
         "Delhi Capitals": "DC", "Punjab Kings": "PBKS", "Rajasthan Royals": "RR",
         "Sunrisers Hyderabad": "SRH", "Deccan Chargers": "Deccan",
         "Gujarat Titans": "GT", "Lucknow Super Giants": "LSG",
         "Rising Pune Supergiant": "RPS", "Gujarat Lions": "GL",
         "Pune Warriors": "PW", "Kochi Tuskers Kerala": "KTK"}
canon = lambda t: TEAM_ALIAS.get(t, t)
CSK = "Chennai Super Kings"

bb["bat_team_c"]  = bb["batting_team"].map(canon)
bb["bowl_team_c"] = bb["bowling_team"].map(canon)
att["bat_team_c"]  = att["bat_team"].map(canon)
att["bowl_team_c"] = att["bowl_team"].map(canon)

# At team level there is no attribution question -- every increment belongs to
# one side or the other -- so we sum the RAW delta rather than the player-level
# WPA. That keeps the decomposition exact (bat + bowl = outcome - initial WP).
# Using the player-level series here would bias the split, because extras are
# debited to the bowler while the matching credit is given to no batter.
bat_side  = bb.groupby(["match_id", "bat_team_c"])["delta"].sum().rename("wpa_bat")
bowl_side = (-bb.groupby(["match_id", "bowl_team_c"])["delta"].sum()).rename("wpa_bowl")
bat_side.index.names = bowl_side.index.names = ["match_id", "team"]
lev = bb.groupby("match_id")["delta"].apply(lambda s: s.abs().mean()).rename("leverage")

TM = (pd.concat([bat_side, bowl_side], axis=1).fillna(0.0).reset_index()
        .merge(lev, on="match_id", how="left")
        .merge(meta[["match_id", "season", "season_ord", "team1", "team2",
                     "final_winner"]], on="match_id", how="left"))
TM["wpa_total"] = TM["wpa_bat"] + TM["wpa_bowl"]
TM["won"] = (TM["team"] == TM["final_winner"].map(canon)).astype(int)
TM["opponent"] = np.where(TM["team"] == TM["team1"].map(canon),
                          TM["team2"].map(canon), TM["team1"].map(canon))
TM.to_csv(os.path.join(OUT, "team_match_wpa.csv"), index=False)

csk = TM[TM["team"] == CSK]
H2H = (csk.groupby("opponent")
         .agg(matches=("won", "size"), wins=("won", "sum"),
              wpa_bat=("wpa_bat", "mean"), wpa_bowl=("wpa_bowl", "mean"),
              leverage=("leverage", "mean"))
         .reset_index())
H2H["win_pct"] = 100 * H2H["wins"] / H2H["matches"]
H2H["short"] = H2H["opponent"].map(lambda t: SHORT.get(t, t[:12]))
H2H = H2H.sort_values("matches", ascending=False)

# --- uncertainty on every head-to-head figure --------------------------------
# The paper's own thesis is that effective sample size is the number of MATCHES,
# so a per-opponent mean over 10-41 fixtures must carry an interval or it
# contradicts the argument it is meant to illustrate. Each row of `csk` is one
# match, so resampling rows IS the match-level bootstrap prescribed in
# Section V-B.
def boot_stat_ci(vals, fn=np.mean, B=CONFIG["N_BOOTSTRAP"], seed=11):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    draws = [fn(v[r.integers(0, len(v), len(v))]) for _ in range(B)]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

UNDERPOWERED_N = 20          # below this a per-opponent split is not reportable
h2h_ci = {}
for opp, g in csk.groupby("opponent"):
    h2h_ci[opp] = dict(
        win=boot_stat_ci(100 * g["won"].values),
        bat=boot_stat_ci(g["wpa_bat"].values),
        bowl=boot_stat_ci(g["wpa_bowl"].values),
        diff=boot_stat_ci(g["wpa_bowl"].values - g["wpa_bat"].values))
for k in ("win", "bat", "bowl", "diff"):
    H2H[k + "_lo"] = H2H["opponent"].map(lambda o: h2h_ci[o][k][0])
    H2H[k + "_hi"] = H2H["opponent"].map(lambda o: h2h_ci[o][k][1])
H2H["underpowered"] = H2H["matches"] < UNDERPOWERED_N

csk_overall = dict(
    matches=int(len(csk)), wins=int(csk["won"].sum()),
    win_pct=100 * float(csk["won"].mean()),
    wpa_bat=float(csk["wpa_bat"].mean()), wpa_bowl=float(csk["wpa_bowl"].mean()),
    diff=float((csk["wpa_bowl"] - csk["wpa_bat"]).mean()),
    win_ci=boot_stat_ci(100 * csk["won"].values),
    bat_ci=boot_stat_ci(csk["wpa_bat"].values),
    bowl_ci=boot_stat_ci(csk["wpa_bowl"].values),
    diff_ci=boot_stat_ci(csk["wpa_bowl"].values - csk["wpa_bat"].values))
print(f"  CSK overall: {csk_overall['wins']}/{csk_overall['matches']} "
      f"({csk_overall['win_pct']:.1f}%, CI [{csk_overall['win_ci'][0]:.1f}, "
      f"{csk_overall['win_ci'][1]:.1f}])")
print(f"    bat  {csk_overall['wpa_bat']:+.3f} CI [{csk_overall['bat_ci'][0]:+.3f}, "
      f"{csk_overall['bat_ci'][1]:+.3f}]")
print(f"    bowl {csk_overall['wpa_bowl']:+.3f} CI [{csk_overall['bowl_ci'][0]:+.3f}, "
      f"{csk_overall['bowl_ci'][1]:+.3f}]")
print(f"    bowl-bat {csk_overall['diff']:+.3f} CI [{csk_overall['diff_ci'][0]:+.3f}, "
      f"{csk_overall['diff_ci'][1]:+.3f}]  -> "
      + ("bowling edge IS distinguishable"
         if csk_overall['diff_ci'][0] > 0 else
         "NOT distinguishable from zero: withdraw the 'built with the ball' claim"))
print(H2H.head(9)[["short", "matches", "wins", "win_pct", "win_lo", "win_hi",
                   "wpa_bat", "wpa_bowl", "diff_lo", "diff_hi", "underpowered"]]
        .round(3).to_string(index=False))

# --- leverage concentration (the mechanism behind the stability result) ------
_ad = bb["delta"].abs()
_tot = _ad.sum()
_d2 = (bb["innings"] == 2) & (bb["phase"] == 2)
LEV = dict(share_deliveries=100 * float(_d2.mean()),
           share_mass=100 * float(_ad[_d2].sum() / _tot),
           mean_death=float(_ad[_d2].mean()), mean_all=float(_ad.mean()))
print(f"\n  Leverage concentration: 2nd-innings death overs are "
      f"{LEV['share_deliveries']:.1f}% of deliveries but carry "
      f"{LEV['share_mass']:.1f}% of |delta| mass "
      f"({LEV['mean_death']:.4f} vs {LEV['mean_all']:.4f} per ball)")

# CSK's leading career WPA contributors
csk_players = (att[att["bat_team_c"] == CSK].groupby("bat_player")["bat_wpa"].sum()
               .add(att[att["bowl_team_c"] == CSK].groupby("bowler")["bowl_wpa"].sum(),
                    fill_value=0.0).sort_values(ascending=False))
csk_balls = (att[att["bat_team_c"] == CSK].groupby("bat_player")["bat_credited"].sum()
             .add(att[att["bowl_team_c"] == CSK].groupby("bowler")["bowl_credited"].sum(),
                  fill_value=0.0))
csk_top = (pd.DataFrame({"wpa": csk_players, "n": csk_balls})
             .query("n >= 240").sort_values("wpa", ascending=False))
print("\n  CSK career WPA leaders:")
print(csk_top.head(6).to_string())

# ----------------------------- 11. STABILITY ---------------------------------
print("\n[9/12] Season-over-season stability ...")
def stability(df, metrics, qual):
    d = df[df["n"] >= qual]
    j = d.assign(k=d["season_ord"]).merge(d.assign(k=d["season_ord"] - 1),
                                          on=["player", "k"], suffixes=("_a", "_b"))
    out = {}
    for m in metrics:
        a, b = j[m + "_a"], j[m + "_b"]
        ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        out[m] = (spearmanr(a[ok], b[ok]).statistic if ok.sum() > 10 else np.nan, int(ok.sum()))
    return out
stab_bat  = stability(BAT,  ["bat_avg", "sr", "wpa", "wpa100"], CONFIG["QUAL_BALLS_BAT"])
stab_bowl = stability(BOWL, ["bowl_avg", "econ", "wpa", "wpa100"], CONFIG["QUAL_BALLS_BOWL"])
print("  batters:", {k: round(v[0], 3) for k, v in stab_bat.items()}, "n =", stab_bat["wpa"][1])
print("  bowlers:", {k: round(v[0], 3) for k, v in stab_bowl.items()}, "n =", stab_bowl["wpa"][1])

# ----------------------------- 12. PREDICTIVE VALIDITY -----------------------
print("\n[10/12] Predictive validity of aggregated squad ratings ...")
bi = BAT.set_index(["season_ord", "player"]); wi = BOWL.set_index(["season_ord", "player"])
def squad(mid, team, prev):
    w = r = k = 0.0
    for p in xis.get(mid, {}).get(team, []):
        if (prev, p) in bi.index:
            row = bi.loc[(prev, p)]
            w += float(row["wpa"]); r += float(np.nan_to_num(row["runs"]))
        if (prev, p) in wi.index:
            row = wi.loc[(prev, p)]
            w += float(row["wpa"]); k += float(np.nan_to_num(row["wkts"]))
    return w, r, k
rows = []
for _, mm in meta.iterrows():
    prev = mm["season_ord"] - 1
    if prev < 0 or not mm["team1"] or not mm["team2"]: continue
    w1, r1, k1 = squad(mm["match_id"], mm["team1"], prev)
    w2, r2, k2 = squad(mm["match_id"], mm["team2"], prev)
    rows.append(dict(match_id=mm["match_id"], season_ord=mm["season_ord"], d_wpa=w1 - w2,
                     d_runs=(r1 - r2) / 100.0, d_wkts=k1 - k2,
                     y=int(mm["final_winner"] == mm["team1"])))
PV = pd.DataFrame(rows).dropna()
pv_tr = PV[PV["season_ord"] <= max(val_ords)]; pv_te = PV[PV["season_ord"].isin(test_ords)]
def pv_eval(cols):
    p = (LogisticRegression(max_iter=1000).fit(pv_tr[cols], pv_tr["y"])
           .predict_proba(pv_te[cols])[:, 1])
    return (roc_auc_score(pv_te["y"], p), log_loss(pv_te["y"], p),
            boot_ci(pv_te["y"].values, p, pv_te["match_id"].values, roc_auc_score))
pv = {"Conventional metrics": pv_eval(["d_runs", "d_wkts"]),
      "WPA (proposed)": pv_eval(["d_wpa"]),
      "Combined": pv_eval(["d_runs", "d_wkts", "d_wpa"])}
for k, v in pv.items():
    print(f"  {k:<22} AUC={v[0]:.3f} [{v[2][0]:.3f}, {v[2][1]:.3f}]  logloss={v[1]:.4f}")

# ----------------------------- 13. SHAP --------------------------------------
print("\n[11/12] Interpretability ...")
samp = cached("shapidx", lambda: rng.choice(
    np.where(m_te)[0], size=min(CONFIG["SHAP_SAMPLE"], int(m_te.sum())), replace=False))
Xs = X_all.iloc[samp]
if HAVE_SHAP and HAVE_XGB:
    def _shap():
        v = shap.TreeExplainer(best).shap_values(Xs)
        return v[1] if isinstance(v, list) else v
    SV = cached("shap", _shap)
    shap_rank = pd.Series(np.abs(SV).mean(0), index=FEATURES).sort_values(ascending=False)
else:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best, Xs, y_all[samp], n_repeats=5, random_state=0,
                                scoring="neg_log_loss")
    SV, shap_rank = None, pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
print("  top features:", [f"{FEAT_LABEL[k]} ({v:.3f})" for k, v in shap_rank.head(4).items()])

# ============================ FIGURES ========================================
print("\n[12/12] Rendering figures ...")

# ---- Figure 1: pipeline schematic ------------------------------------------
fig, ax = plt.subplots(figsize=(COL_W, 4.4))
ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 15.6); ax.grid(False)
stages = [("Cricsheet ball-by-ball\nIPL corpus", "#dbe5f1"),
          ("Filtering: no-result,\nD/L, Super Over removed", "#dbe5f1"),
          ("Pre-delivery state $s_i$\nballs, wickets, RRR, phase, par", "#e4eeda"),
          ("Gradient-boosted\nwin-probability model", "#fde9d9"),
          ("Isotonic calibration\non validation season", "#fde9d9"),
          ("Delivery attribution\n$\\Delta_i=\\hat{p}_{i+1}-\\hat{p}_i$", "#f2dcdb"),
          ("Seasonal aggregation\nWPA, WPA$_{100}$", "#f2dcdb"),
          ("Validation: calibration,\nstability, predictive validity", "#e5e0ec")]
h, gap = 1.45, 0.5
for i, (t, cbox) in enumerate(stages):
    yb = 15.6 - (i + 1) * h - i * gap
    ax.add_patch(FancyBboxPatch((0.5, yb), 9.0, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                fc=cbox, ec="#404040", lw=0.7))
    ax.text(5.0, yb + h / 2, t, ha="center", va="center", fontsize=6.8, linespacing=1.25)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((5.0, yb), (5.0, yb - gap), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.8, color="#404040"))
if CONFIG["FIG_TITLES"]: ax.set_title("Impact-evaluation pipeline", pad=4)
savefig(fig, "fig_pipeline"); plt.close(fig)

# ---- Figure 2: reliability diagram -----------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
def rel(p, k=CONFIG["N_ECE_BINS"]):
    q = pd.qcut(pd.Series(p), k, labels=False, duplicates="drop")
    a = (pd.DataFrame({"y": yte, "p": p, "q": q}).groupby("q", observed=True)
           .agg(e=("p", "mean"), o=("y", "mean")))
    return a["e"].values, a["o"].values
a1.plot([0, 1], [0, 1], "--", color="#909090", lw=0.9, label="Perfect calibration")
for p, lab, cc, mk in [
        (p_gbm_raw, f"Uncalibrated (ECE = {ece(yte,p_gbm_raw):.3f})", C_ALT, "s"),
        (p_cal_va,  f"Isotonic, 1 season (ECE = {ece(yte,p_cal_va):.3f})", C_ACC, "^"),
        (p_gbm_cal, f"Isotonic, pooled (ECE = {ece(yte,p_gbm_cal):.3f})", C_MAIN, "o")]:
    e, o = rel(p); a1.plot(e, o, mk + "-", color=cc, ms=3.2, label=lab)
a1.set_ylabel("Observed win frequency"); a1.set_xlim(0, 1); a1.set_ylim(0, 1)
a1.legend(loc="upper left", frameon=False)
if CONFIG["FIG_TITLES"]: a1.set_title(f"Reliability, held-out seasons {span(test_ords)}", pad=4)
a2.hist(p_gbm_cal, bins=40, color=C_MAIN, alpha=0.75)
a2.set_xlabel("Predicted win probability (batting team)")
a2.set_ylabel("Deliveries"); a2.set_xlim(0, 1)
savefig(fig, "fig_reliability"); plt.close(fig)

# ---- Figure 3: SHAP summary -------------------------------------------------
if SV is not None:
    # Hand-drawn beeswarm. shap.summary_plot hard-codes its own font sizes and
    # colour-bar geometry, which do not survive a 3.5in IEEE column.
    order = list(shap_rank.index)[::-1]                      # least -> most important
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    nshow = min(4000, SV.shape[0])
    pick = rng.choice(SV.shape[0], nshow, replace=False)
    for row, feat in enumerate(order):
        j = FEATURES.index(feat)
        sv = SV[pick, j]
        fv = Xs[feat].to_numpy()[pick].astype(float)
        ok = np.isfinite(fv)
        colour = np.full(len(fv), np.nan)
        if ok.sum() > 1:                                     # colour by percentile rank
            colour[ok] = pd.Series(fv[ok]).rank(pct=True).to_numpy()
        # vertical jitter proportional to local density, as in a beeswarm
        h, edges = np.histogram(sv, bins=40)
        dens = h[np.clip(np.digitize(sv, edges) - 1, 0, len(h) - 1)]
        jit = (rng.random(len(sv)) - 0.5) * 0.72 * np.sqrt(dens / max(dens.max(), 1))
        ax.scatter(sv, row + jit, c=colour, cmap="coolwarm", vmin=0, vmax=1,
                   s=1.6, alpha=0.55, linewidths=0, rasterized=True)
    ax.axvline(0, color="#909090", lw=0.7, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEAT_LABEL[f] for f in order])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("SHAP value (impact on log-odds of winning)")
    ax.grid(axis="y", alpha=0.15); ax.grid(axis="x", alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("Feature value (percentile)", fontsize=7)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=7)
    if CONFIG["FIG_TITLES"]: ax.set_title("Global feature attribution", pad=4)
else:
    fig, ax = plt.subplots(figsize=(COL_W, 2.8)); r = shap_rank.iloc[::-1]
    ax.barh([FEAT_LABEL[i] for i in r.index], r.values, color=C_MAIN)
    ax.set_xlabel("Permutation importance (increase in log loss)")
savefig(fig, "fig_shap_summary"); plt.close("all")

# ---- Figure 4: SHAP dependence, RRR x wickets -------------------------------
fig, ax = plt.subplots(figsize=(COL_W, COL_H))
k2 = (Xs["innings"].values == 2) & np.isfinite(Xs["rrr2"].values) & (Xs["rrr2"].values < 24)
yv = SV[k2, FEATURES.index("rrr2")] if SV is not None else np.zeros(int(k2.sum()))
rr, wk = Xs["rrr2"].values[k2], Xs["wkts_rem"].values[k2]
sc = ax.scatter(rr, yv, c=wk, cmap="viridis", s=2.0, alpha=0.35, linewidths=0,
                rasterized=True)
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Wickets in hand", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
# median trend per wicket band, so the interaction is legible and not just implied
bins = np.arange(0, 24.1, 1.5)
mid = 0.5 * (bins[:-1] + bins[1:])
for lo, hi, lab, ls in [(1, 3, "1--3 wickets", ":"), (4, 6, "4--6", "--"), (7, 10, "7--10", "-")]:
    sel = (wk >= lo) & (wk <= hi)
    if sel.sum() < 50: continue
    med = [np.median(yv[sel & (rr >= bins[i]) & (rr < bins[i + 1])])
           if ((sel & (rr >= bins[i]) & (rr < bins[i + 1])).sum() >= 15) else np.nan
           for i in range(len(mid))]
    ax.plot(mid, med, ls, color="#202020", lw=1.1, label=lab)
ax.legend(frameon=False, loc="lower left", fontsize=6.5, title="Wickets in hand",
          title_fontsize=6.5)
ax.axhline(0, color="#909090", lw=0.7, ls="--")
ax.set_xlabel("Required run rate (runs per over)")
ax.set_ylabel("SHAP value for required run rate")
if CONFIG["FIG_TITLES"]: ax.set_title("Required run rate $\\times$ wickets in hand", pad=4)
savefig(fig, "fig_shap_dependence"); plt.close(fig)

# ---- Figure 5: win-probability trace of a close match -----------------------
cand = bb[bb["match_id"].isin(bb.loc[m_te, "match_id"].unique())].copy()
cand["chase_p"] = np.where(cand["innings"] == 1, 1 - cand["p"], cand["p"])
def closeness(d):
    t = d[d["innings"] == 2]["chase_p"].tail(30).values
    if len(t) < 10: return -1
    return int(np.sum(np.abs(np.diff((t > 0.5).astype(int))))) * 10 + float(np.std(t[-12:])) * 10
pick = cand.groupby("match_id").apply(closeness, include_groups=False).idxmax()
mt = cand[cand["match_id"] == pick].sort_values(["innings", "seq"]).reset_index(drop=True)
mrow = meta[meta["match_id"] == pick].iloc[0]
mt["x"] = np.arange(len(mt)); brk = int((mt["innings"] == 1).sum())

fig, ax = plt.subplots(figsize=(COL_W, COL_H + 0.35))
ax.plot(mt["x"], mt["chase_p"], color=C_MAIN, lw=1.1)
ax.fill_between(mt["x"], 0.5, mt["chase_p"], where=mt["chase_p"] >= 0.5, color=C_MAIN, alpha=0.15, lw=0)
ax.fill_between(mt["x"], 0.5, mt["chase_p"], where=mt["chase_p"] < 0.5, color=C_ALT, alpha=0.15, lw=0)
ax.axhline(0.5, color="#909090", lw=0.7, ls="--")
ax.axvline(brk, color="#404040", lw=0.8, ls=":")
ax.text(brk, 1.02, "innings break", ha="center", va="bottom", fontsize=6.5, color="#404040")
# three largest swings, kept >=10 deliveries apart so the labels do not collide
sel = []
for i_ in mt["delta"].abs().sort_values(ascending=False).index:
    if all(abs(int(mt.loc[i_, "x"]) - j) >= 10 for j in sel):
        sel.append(int(mt.loc[i_, "x"]))
        if len(sel) == 3: break
top = mt[mt["x"].isin(sel)].sort_values("x")
nb = len(mt)
for _, r in top.iterrows():
    yv0 = float(r["chase_p"]); sgn = "+" if r["delta"] > 0 else "\u2212"
    tx = min(max(float(r["x"]), 0.13 * nb), 0.87 * nb)      # keep labels inside the axes
    ty = yv0 + (0.24 if yv0 < 0.55 else -0.28)
    ax.annotate(f"{str(r['striker']).split()[-1]} v {str(r['bowler']).split()[-1]}\n"
                f"{sgn}{abs(r['delta'])*100:.0f} WP",
                xy=(r["x"], yv0), xytext=(tx, ty), fontsize=6, ha="center", color="#202020",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#606060"))
# the chasing side is whichever team bats in the second innings, not team2
chase_team = mt.loc[mt["innings"] == 2, "batting_team"].iloc[0]
ax.set_xlabel("Delivery number (both innings)")
ax.set_ylabel(f"Win probability: {chase_team}")
ax.set_ylim(-0.02, 1.15); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_xlim(0, len(mt))
if CONFIG["FIG_TITLES"]: ax.set_title(f"{mrow['team1']} v {mrow['team2']}, {mrow['season']}", pad=10)
savefig(fig, "fig_wp_trace"); plt.close(fig)

# ---- Extra figure: regularisation / effective sample size -------------------
fig, ax = plt.subplots(figsize=(COL_W, COL_H))
ax.semilogx(abl["mcw"], abl["val"], "s--", color=C_ACC, ms=3.5,
            label=f"Validation ({span(val_ords)})")
ax.semilogx(abl["mcw"], abl["test"], "o-", color=C_MAIN, ms=3.5,
            label=f"Test ({span(test_ords)})")
ax.set_xlabel("Minimum leaf size (deliveries)")
ax.set_ylabel("Log loss")
sec = ax.secondary_xaxis("top", functions=(lambda v: v / 240.0, lambda v: v * 240.0))
sec.set_xlabel("Equivalent independent matches", fontsize=7.5); sec.tick_params(labelsize=7)
ax.legend(frameon=False, loc="upper right")
if CONFIG["FIG_TITLES"]: ax.set_title("Leaf size vs generalisation", pad=14)
savefig(fig, "fig_regularisation"); plt.close(fig)

# ---- Figure 6: CSK head-to-head profile -------------------------------------
H = H2H[H2H["matches"] >= UNDERPOWERED_N].head(9).iloc[::-1]   # reportable only
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True,
                               gridspec_kw=dict(wspace=0.08))
ypos = np.arange(len(H))
cols = [C_ACC if lo > 50 else (C_ALT if hi < 50 else "#9a9a9a")
        for lo, hi in zip(H["win_lo"], H["win_hi"])]
ax1.barh(ypos, H["win_pct"], color=cols, height=0.68)
ax1.errorbar(H["win_pct"], ypos, xerr=[H["win_pct"] - H["win_lo"],
                                       H["win_hi"] - H["win_pct"]],
             fmt="none", ecolor="#303030", elinewidth=0.8, capsize=1.8)
ax1.axvline(50, color="#404040", lw=0.8, ls="--")
for i, (hi, n) in enumerate(zip(H["win_hi"], H["matches"])):
    ax1.text(min(hi + 2.0, 88), i, f"n={n}", va="center", fontsize=6.5)
ax1.set_yticks(ypos); ax1.set_yticklabels(H["short"])
ax1.set_xlabel("CSK win rate (%)"); ax1.set_xlim(0, 105)
ax1.set_title("(a) Head-to-head record", fontsize=8, pad=4)

ax2.barh(ypos - 0.19, H["wpa_bat"], height=0.36, color=C_MAIN, label="Batting WPA",
         xerr=[H["wpa_bat"] - H["bat_lo"], H["bat_hi"] - H["wpa_bat"]],
         error_kw=dict(ecolor="#303030", elinewidth=0.7, capsize=1.4))
ax2.barh(ypos + 0.19, H["wpa_bowl"], height=0.36, color="#e0a44c", label="Bowling WPA",
         xerr=[H["wpa_bowl"] - H["bowl_lo"], H["bowl_hi"] - H["wpa_bowl"]],
         error_kw=dict(ecolor="#303030", elinewidth=0.7, capsize=1.4))
ax2.axvline(0, color="#404040", lw=0.8)
ax2.set_xlabel("Mean CSK WPA per match")
ax2.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.19))
ax2.set_title("(b) Source of the margin", fontsize=8, pad=4)
if CONFIG["FIG_TITLES"]:
    fig.suptitle("Chennai Super Kings against the field, "
                 f"{season_labels[0]}--{season_labels[-1]}", fontsize=8.5, y=1.02)
savefig(fig, "fig_h2h_csk"); plt.close(fig)

# ---- Divergence tables ------------------------------------------------------
last_season = season_labels[-1]
def divergence(df, col, higher_better, qual):
    d = df[(df["season"] == last_season) & (df["n"] >= qual) & df[col].notna()].copy()
    d["conv_rank"] = d[col].rank(ascending=not higher_better, method="min")
    d["wpa_rank"]  = d["wpa"].rank(ascending=False, method="min")
    d["shift"] = d["conv_rank"] - d["wpa_rank"]
    return d.reindex(d["shift"].abs().sort_values(ascending=False).index)
div_bat  = divergence(BAT,  "sr",   True,  CONFIG["QUAL_BALLS_BAT"])
div_bowl = divergence(BOWL, "econ", False, CONFIG["QUAL_BALLS_BOWL"])
print(f"\n  Largest divergence, {last_season} batters (strike rate vs WPA):")
print(div_bat[["player", "conv_rank", "wpa_rank", "shift", "sr", "wpa"]].head(6).to_string(index=False))

# ============================ LATEX EXPORT ===================================
def f(x, n=3):
    try:
        v = float(x); return "--" if not np.isfinite(v) else f"{v:.{n}f}"
    except Exception: return "--"
def num(x): return f"{int(x):,}".replace(",", "{,}")

L = []; A = L.append
A("% ============ AUTO-GENERATED by wpa_pipeline.py -- paste into the paper ============")
A("")
A("% ---- inline macros ----")
A(f"\\newcommand{{\\Ndeliveries}}{{{num(len(bb))}}}")
A(f"\\newcommand{{\\Mmatches}}{{{num(bb['match_id'].nunique())}}}")
A(f"\\newcommand{{\\seasonrange}}{{{season_labels[0]}--{season_labels[-1]}}}")
A(f"\\newcommand{{\\testlogloss}}{{{f(perf.loc['Proposed (calibrated)','logloss'],4)}}}")
A(f"\\newcommand{{\\testbrier}}{{{f(perf.loc['Proposed (calibrated)','brier'],4)}}}")
A(f"\\newcommand{{\\testauc}}{{{f(perf.loc['Proposed (calibrated)','auc'])}}}")
A(f"\\newcommand{{\\testece}}{{{f(perf.loc['Proposed (calibrated)','ece'])}}}")
A(f"\\newcommand{{\\stabwpa}}{{{f(stab_bat['wpa100'][0])}}}")
A(f"\\newcommand{{\\stabsr}}{{{f(stab_bat['sr'][0])}}}")
A(f"\\newcommand{{\\pvauc}}{{{f(pv['WPA (proposed)'][0])}}}")
A(f"\\newcommand{{\\nbootstrap}}{{{CONFIG['N_BOOTSTRAP']}}}")
A(f"\\newcommand{{\\qualballs}}{{{CONFIG['QUAL_BALLS_BAT']}}}")
A(f"\\newcommand{{\\residualpct}}{{{100*resid:.2f}}}")
A("")

A("% ---- TABLE I (tab:data) ----")
A("\\begin{table}[htbp]\n\\caption{Dataset Composition After Filtering}\n\\label{tab:data}")
A("\\centering\n\\begin{tabular}{lrrr}\n\\toprule")
A("\\textbf{Split} & \\textbf{Seasons} & \\textbf{Matches} & \\textbf{Deliveries} \\\\\n\\midrule")
for _, r in split_df.iterrows():
    if r["split"] == "Total": A("\\midrule")
    A(f"{r['split']} & {r['seasons']} & {num(r['matches'])} & {num(r['deliveries'])} \\\\")
A("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

A("% ---- TABLE II (tab:hyper) ----")
A("\\begin{table}[htbp]\n\\caption{Selected Model Hyperparameters}\n\\label{tab:hyper}")
A("\\centering\n\\begin{tabular}{ll}\n\\toprule\n\\textbf{Parameter} & \\textbf{Value} \\\\\n\\midrule")
PRETTY = {"n_estimators": "Boosting rounds", "max_depth": "Maximum tree depth",
          "learning_rate": "Learning rate", "min_child_weight": "Min.\\ leaf size (deliveries)"}
for k, v in best_p.items(): A(f"{PRETTY.get(k,k)} & {v} \\\\")
A("Row subsample & 0.8 \\\\\nColumn subsample & 0.8 \\\\\n$L_2$ regularisation & 5.0 \\\\")
A("Calibration & Isotonic (validation season) \\\\")
A("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

A("% ---- TABLE III (tab:wp) ----")
A("\\begin{table}[htbp]\n\\caption{Win Probability Model Performance on Held-Out Test Seasons}\n\\label{tab:wp}")
A("\\centering\n\\begin{tabular}{lcccc}\n\\toprule")
A("\\textbf{Model} & \\textbf{Log loss} & \\textbf{Brier} & \\textbf{AUC} & \\textbf{ECE} \\\\\n\\midrule")
CITE = {"Resource-based": " \\cite{dl1998}"}
for k in MODELS:
    r = perf.loc[k]
    nm = ("\\textbf{" + k + "}") if k == "Proposed (calibrated)" else k + CITE.get(k, "")
    A(f"{nm} & {f(r['logloss'],4)} & {f(r['brier'],4)} & {f(r['auc'])} & {f(r['ece'])} \\\\")
A("\\bottomrule\n\\end{tabular}")
A(f"\\\\[2pt]\\footnotesize 95\\% CI (match-level bootstrap, $B={CONFIG['N_BOOTSTRAP']}$) for the "
  f"proposed model: log loss [{f(ci['Proposed (calibrated)']['logloss'][0],4)}, "
  f"{f(ci['Proposed (calibrated)']['logloss'][1],4)}], AUC "
  f"[{f(ci['Proposed (calibrated)']['auc'][0])}, {f(ci['Proposed (calibrated)']['auc'][1])}].")
A("\\end{table}\n")

A("% ---- TABLE IV (tab:phase) ----")
A("\\begin{table}[htbp]\n\\caption{Test Log Loss by Innings and Phase}\n\\label{tab:phase}")
A("\\centering\n\\begin{tabular}{lccc}\n\\toprule")
A("\\textbf{Innings} & \\textbf{Powerplay} & \\textbf{Middle} & \\textbf{Death} \\\\\n\\midrule")
for i_, nm in [(1, "First"), (2, "Second")]:
    r = phase_tbl.loc[i_]
    A(f"{nm} & {f(r['Powerplay'],4)} & {f(r['Middle'],4)} & {f(r['Death'],4)} \\\\")
A("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

A("% ---- TABLE V (tab:stability) ----")
A("\\begin{table}[htbp]\n\\caption{Season-over-Season Spearman Rank Correlation of Player Ratings}\n\\label{tab:stability}")
A("\\centering\n\\begin{tabular}{lcc}\n\\toprule")
A("\\textbf{Metric} & \\textbf{Batters} & \\textbf{Bowlers} \\\\\n\\midrule")
A(f"Batting average / Bowling average & {f(stab_bat['bat_avg'][0])} & {f(stab_bowl['bowl_avg'][0])} \\\\")
A(f"Strike rate / Economy rate & {f(stab_bat['sr'][0])} & {f(stab_bowl['econ'][0])} \\\\")
A(f"\\textbf{{WPA (proposed)}} & {f(stab_bat['wpa'][0])} & {f(stab_bowl['wpa'][0])} \\\\")
A(f"\\textbf{{WPA$_{{100}}$ (proposed)}} & {f(stab_bat['wpa100'][0])} & {f(stab_bowl['wpa100'][0])} \\\\")
A("\\bottomrule\n\\end{tabular}")
A(f"\\\\[2pt]\\footnotesize $n={stab_bat['wpa'][1]}$ batter and $n={stab_bowl['wpa'][1]}$ "
  f"consecutive-season pairs, qualification threshold {CONFIG['QUAL_BALLS_BAT']} deliveries.")
A("\\end{table}\n")

A("% ---- TABLE VI (tab:validity) ----")
A("\\begin{table}[htbp]\n\\caption{Next-Season Match Outcome Prediction from Aggregated Squad Ratings}\n\\label{tab:validity}")
A("\\centering\n\\begin{tabular}{lcc}\n\\toprule")
A("\\textbf{Rating basis} & \\textbf{AUC [95\\% CI]} & \\textbf{Log loss} \\\\\n\\midrule")
for k, v in pv.items():
    nm = ("\\textbf{" + k + "}") if k.startswith("WPA") else k
    A(f"{nm} & {f(v[0])} [{f(v[2][0])}, {f(v[2][1])}] & {f(v[1],4)} \\\\")
A("\\bottomrule\n\\end{tabular}")
A(f"\\\\[2pt]\\footnotesize Evaluated on {len(pv_te)} test-season matches.")
A("\\end{table}\n")

A("% ---- TABLE VII (tab:divergence) ----")
A(f"\\begin{{table}}[htbp]\n\\caption{{Largest Rank Divergence Between WPA and Strike Rate, Season {last_season}}}\n\\label{{tab:divergence}}")
A("\\centering\n\\begin{tabular}{lccc}\n\\toprule")
A("\\textbf{Player} & \\textbf{Conv. rank} & \\textbf{WPA rank} & \\textbf{$\\Delta$} \\\\\n\\midrule")
for _, r in div_bat.head(5).iterrows():
    s = int(r["shift"]); A(f"{r['player']} & {int(r['conv_rank'])} & {int(r['wpa_rank'])} & {'+' if s>0 else ''}{s} \\\\")
A("\\bottomrule\n\\end{tabular}")
A("\\\\[2pt]\\footnotesize Positive $\\Delta$: WPA ranks the player higher than strike rate does.")
A("\\end{table}\n")

A("% ---- TABLE IX (tab:h2h) -- CSK head-to-head ----")
A("\\begin{table}[htbp]\n\\caption{Chennai Super Kings Head-to-Head Record and Mean Per-Match WPA Decomposition}\n\\label{tab:h2h}")
A("\\centering\n\\begin{tabular}{lrrrr}\n\\toprule")
A("\\textbf{Opp.} & \\textbf{M} & \\textbf{Win \\%} & "
  "\\textbf{Bat WPA} & \\textbf{Bowl $-$ Bat} \\\\\n\\midrule")
for _, r in H2H[H2H["matches"] >= UNDERPOWERED_N].head(9).iterrows():
    A(f"{r['short']} & {int(r['matches'])} & "
      f"{r['win_pct']:.1f} [{r['win_lo']:.0f}, {r['win_hi']:.0f}] & "
      f"{r['wpa_bat']:+.3f} & "
      f"{r['diff_lo']:+.3f} to {r['diff_hi']:+.3f} \\\\")
A("\\midrule")
A(f"\\textbf{{All}} & {csk_overall['matches']} & "
  f"{csk_overall['win_pct']:.1f} [{csk_overall['win_ci'][0]:.0f}, "
  f"{csk_overall['win_ci'][1]:.0f}] & {csk_overall['wpa_bat']:+.3f} & "
  f"{csk_overall['diff_ci'][0]:+.3f} to {csk_overall['diff_ci'][1]:+.3f} \\\\")
A("\\bottomrule\n\\end{tabular}")
A(f"\\\\[2pt]\\footnotesize Brackets are 95\\% match-level bootstrap intervals "
  f"($B={CONFIG['N_BOOTSTRAP']}$). Opponents with fewer than {UNDERPOWERED_N} "
  "fixtures are omitted as underpowered. Per-match WPA is denominated in win "
  "probability, so a team's total is close to $+0.5$ when it wins and $-0.5$ when "
  "it loses; the batting/bowling split is the informative quantity, and the final "
  "column gives the interval for that difference.")
A("\\end{table}\n")

A("% ---- TABLE X (tab:cskplayers) -- CSK career WPA leaders ----")
A("\\begin{table}[htbp]\n\\caption{Leading Career WPA Contributors for Chennai Super Kings}\n\\label{tab:cskplayers}")
A("\\centering\n\\begin{tabular}{lrr}\n\\toprule")
A("\\textbf{Player} & \\textbf{Deliveries} & \\textbf{Career WPA} \\\\\n\\midrule")
for nm, r in csk_top.head(6).iterrows():
    A(f"{nm} & {int(r['n']):,} & {r['wpa']:+.2f} \\\\".replace(",", "{,}"))
A("\\bottomrule\n\\end{tabular}")
A("\\\\[2pt]\\footnotesize Deliveries faced plus deliveries bowled while representing "
  "the franchise; minimum 240.")
A("\\end{table}\n")

A("% ---- TABLE VIII (tab:regularisation) -- NEW, supports Section V-B ----")
A("\\begin{table}[htbp]\n\\caption{Effect of Minimum Leaf Size on Generalisation}\n\\label{tab:regularisation}")
A("\\centering\n\\begin{tabular}{rrccc}\n\\toprule")
A("\\textbf{Leaf} & \\textbf{Matches} & \\textbf{Val.\\ LL} & \\textbf{Test LL} & \\textbf{Test AUC} \\\\\n\\midrule")
for _, r in abl.iterrows():
    A(f"{int(r['mcw'])} & {r['matches']:.1f} & {f(r['val'],4)} & {f(r['test'],4)} & {f(r['auc'])} \\\\")
A("\\bottomrule\n\\end{tabular}")
A("\\\\[2pt]\\footnotesize Leaf size in deliveries; ``Matches'' divides by the $\\approx$240 "
  "deliveries that share one outcome label.")
A("\\end{table}\n")
open(os.path.join(OUT, "paper_values.tex"), "w", encoding="utf-8").write("\n".join(L))

# ---- plain-text placeholder map --------------------------------------------
T = []; P = T.append
P("PLACEHOLDER -> VALUE MAP   (paper location in brackets)")
P("=" * 72)
P(f"[Abstract] N deliveries              : {len(bb):,}")
P(f"[Abstract] M matches                 : {bb['match_id'].nunique():,}")
P(f"[Abstract] season range              : {season_labels[0]}--{season_labels[-1]}")
P(f"[Abstract] log loss                  : {perf.loc['Proposed (calibrated)','logloss']:.4f}")
P(f"[Abstract] Brier score               : {perf.loc['Proposed (calibrated)','brier']:.4f}")
P(f"[Abstract] AUC                       : {perf.loc['Proposed (calibrated)','auc']:.3f}")
P(f"[Abstract] ECE                       : {perf.loc['Proposed (calibrated)','ece']:.3f}")
P(f"[Abstract] rank corr WPA_100         : {stab_bat['wpa100'][0]:.3f} bat / {stab_bowl['wpa100'][0]:.3f} bowl")
P(f"[Abstract] rank corr strike rate     : {stab_bat['sr'][0]:.3f}  (economy {stab_bowl['econ'][0]:.3f})")
P(f"[Abstract] squad-rating AUC          : {pv['WPA (proposed)'][0]:.3f}")
P(f"[III-A] raw corpus                   : {N_RAW_MATCHES} matches, {N_RAW_BALLS:,} delivery records")
P(f"[III-B] exclusions                   : {n_noresult} no-result, {n_dls} D/L, {n_super} Super Over "
  f"-> {bb['match_id'].nunique()} retained")
P(f"[IV-A]  grid                         : {len(GRID)} configs; n_estimators{{300,600}} x "
  f"max_depth{{3,4}} x min_child_weight{{500,2000,5000}}, lr=0.05")
P(f"[IV-A]  selected                     : {best_p}")
P(f"[IV-D]  unattributed residual        : {100*resid:.2f}% of total |WPA| mass (byes/leg-byes)")
P(f"[IV-D]  qualification threshold      : {CONFIG['QUAL_BALLS_BAT']} deliveries")
P(f"[V-A]   train/val/test               : {span(tr_ords)} / {span(val_ords)} / {span(test_ords)}")
P(f"[V-B]   bootstrap replicates B       : {CONFIG['N_BOOTSTRAP']} (resampled at match level)")
P(f"[V-C]   ECE bins k                   : {CONFIG['N_ECE_BINS']} (equal frequency)")
P(f"[V-D]   software                     : Python {sys.version.split()[0]}, pandas {pd.__version__}, "
  f"scikit-learn {__import__('sklearn').__version__}"
  + (f", XGBoost {xgb.__version__}" if HAVE_XGB else "") + (f", SHAP {shap.__version__}" if HAVE_SHAP else ""))
P(f"[V-D]   OOF log loss (rating basis)  : {oof_ll:.4f}")
P("")
P("95% CIs, match-level bootstrap:")
for k, v in ci.items():
    P(f"   {k:<26} logloss [{v['logloss'][0]:.4f}, {v['logloss'][1]:.4f}]   "
      f"AUC [{v['auc'][0]:.3f}, {v['auc'][1]:.3f}]")
P("")
P("SHAP mean |value| ranking:")
for k, v in shap_rank.items(): P(f"   {FEAT_LABEL[k]:<26} {v:.4f}")
P("")
P(f"[VI-F] case-study match              : {mrow['team1']} v {mrow['team2']}, "
  f"{mrow['city']}, {mrow['season']}")
for _, r in top.iterrows():
    P(f"   swing {r['delta']*100:+.1f} WP   {r['striker']} vs {r['bowler']}")
P("")
P("[VI-D] leverage concentration (mechanism behind the stability result):")
P(f"   2nd-innings death overs   : {LEV['share_deliveries']:.1f}% of deliveries, "
  f"{LEV['share_mass']:.1f}% of |delta| mass")
P(f"   mean |delta| per ball     : {LEV['mean_death']:.4f} death vs "
  f"{LEV['mean_all']:.4f} overall")
P("")
P("[VI-I] CSK head-to-head, 95% match-level bootstrap CIs:")
P(f"   overall        : {csk_overall['wins']}/{csk_overall['matches']} = "
  f"{csk_overall['win_pct']:.1f}% [{csk_overall['win_ci'][0]:.1f}, {csk_overall['win_ci'][1]:.1f}]")
P(f"   batting WPA    : {csk_overall['wpa_bat']:+.3f} "
  f"[{csk_overall['bat_ci'][0]:+.3f}, {csk_overall['bat_ci'][1]:+.3f}]")
P(f"   bowling WPA    : {csk_overall['wpa_bowl']:+.3f} "
  f"[{csk_overall['bowl_ci'][0]:+.3f}, {csk_overall['bowl_ci'][1]:+.3f}]")
P(f"   bowl - bat     : {csk_overall['diff']:+.3f} "
  f"[{csk_overall['diff_ci'][0]:+.3f}, {csk_overall['diff_ci'][1]:+.3f}]  "
  + ("DISTINGUISHABLE from zero" if csk_overall['diff_ci'][0] > 0
     else "NOT distinguishable from zero"))
P(f"   opponents dropped as underpowered (n < {UNDERPOWERED_N}): "
  + ", ".join(H2H[H2H['underpowered']]['short'].tolist()))
P("")
P(f"[VI-G] divergence, {last_season} bowlers (economy vs WPA):")
for _, r in div_bowl.head(5).iterrows():
    P(f"   {r['player']:<24} econ rank {int(r['conv_rank']):>3} -> WPA rank {int(r['wpa_rank']):>3} "
      f"({int(r['shift']):+d})")
P("")
P("!! TEXT CHANGES REQUIRED (results contradict the draft) -- see notes.md")
txt = "\n".join(T)
open(os.path.join(OUT, "paper_values.txt"), "w", encoding="utf-8").write(txt)
print("\n" + "=" * 72); print(txt); print("=" * 72)
print(f"\nWrote {OUT}/paper_values.tex, {OUT}/paper_values.txt, ratings CSVs and 6 figures.")
