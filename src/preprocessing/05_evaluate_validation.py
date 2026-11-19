
"""Agreement of the original coder with Label Studio validators. Usage: python evaluate_validation.py [export_dir]"""
import json, sys
from itertools import combinations
from pathlib import Path
 
import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
 
ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "labelstudio_batches" / "validation"
EXPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else VAL / "labelstudio" / "export"
OUT = ROOT / "data" / "results"
rng = np.random.default_rng(20261105)
B = 1000
# ordinal position per label; gemischt -> negativ is an extra collapsed run
SCALES = {"stance_3": {"contra": -1, "neutral": 0, "pro": 1},
          "tone": {"negativ": -1, "gemischt": 0, "neutral": 0, "positiv": 1}}
RUNS = [("stance_3", {}), ("tone", {}), ("tone", {"gemischt": "negativ"})]
 
 
def norm(v, var):
    v = str(v).strip().casefold().replace(" ", "_")
    if var == "stance_3" and v.startswith(("pro", "contra")):  # collapse 5-way stance
        v = v.split("_")[0]
    return v if v in SCALES[var] else np.nan  # missing / nicht_beurteilbar / unknown -> NaN
 
 
def load(path):
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype=str).drop_duplicates("code", keep="last").set_index("code")
    rows = {}
    for t in json.loads(path.read_text(encoding="utf-8")):
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if anns:
            a = max(anns, key=lambda a: a.get("updated_at", ""))
            rows[t["data"]["code"]] = {r["from_name"]: r["value"]["choices"][0] for r in a["result"] if "choices" in r.get("value", {})}
    return pd.DataFrame.from_dict(rows, orient="index")
 
 
def kappa(a, b):
    m = pd.notna(a) & pd.notna(b)
    return cohen_kappa_score(a[m], b[m]) if len(set(a[m]) | set(b[m])) > 1 else np.nan
 
 
def alpha(x, level):  # x: items x raters, numeric with NaN
    x = x[(~np.isnan(x)).sum(1) >= 2]
    return krippendorff.alpha(reliability_data=x.T, level_of_measurement=level) if len(np.unique(x[~np.isnan(x)])) > 1 else np.nan
 
 
def ci(stat, n):
    e = np.array([stat(rng.integers(0, n, n)) for _ in range(B)], float)
    return np.nanquantile(e, [0.025, 0.975])
 
 
key = pd.read_csv(VAL / "PRIVATE_key.csv", dtype=str).query("code.str.startswith('V')").set_index("code")
exports = {p.stem: load(p) for p in sorted(EXPORT.glob("*")) if p.suffix in (".csv", ".json")}
raters = ["original", *exports]
key["stance_3"] = key.get("stance_3", key.get("stance"))
if (VAL / "import_main.json").exists():
    key["text"] = key.index.map({t["data"]["code"]: t["data"].get("text") for t in json.loads((VAL / "import_main.json").read_text(encoding="utf-8"))})
lab = {var: pd.DataFrame({"original": key[var], **{r: d[var].reindex(key.index) for r, d in exports.items()}})
             .map(lambda v: norm(v, var)) for var in SCALES}
 
OUT.mkdir(parents=True, exist_ok=True)
rows, conf = [], []
for var, merge in RUNS:
    L = lab[var].replace(merge)
    name = var + (f" ({', '.join(f'{k}->{v}' for k, v in merge.items())})" if merge else "")
    X, n, s = L.to_numpy(object), len(L), SCALES[var]
    add = lambda metric, who, v, c=(np.nan, np.nan), **kw: rows.append(dict(variable=name, metric=metric, raters=who, value=v, ci_low=c[0], ci_high=c[1], **kw))
    for i, j in combinations(range(len(raters)), 2):
        a, b = X[:, i], X[:, j]
        both = pd.notna(a) & pd.notna(b)
        sa, sb = pd.Series(a).map(s), pd.Series(b).map(s)
        polar = (sa * sb) != 0
        add("kappa", f"{raters[i]}-{raters[j]}", kappa(a, b), ci(lambda k: kappa(a[k], b[k]), n),
            n=both.sum(), agree=(a[both] == b[both]).mean(), direction_agree=(sa[polar] == sb[polar]).mean())
        if not merge:
            conf.append(pd.crosstab(a, b).stack().rename("count").reset_index().assign(variable=var, raters=f"{raters[i]}-{raters[j]}"))
    codes = {c: k for k, c in enumerate(s)}
    for who in (raters, raters[1:]):
        for level, m in (("nominal", codes), ("ordinal", s)):
            x = L[who].apply(lambda c: c.map(m)).to_numpy(float)
            add(f"alpha_{level}", "-".join(who), alpha(x, level), ci(lambda k: alpha(x[k], level), n))
    if len(raters) > 2:  # original as reference vs. validators
        V = range(1, len(raters))
        diff = lambda k: np.nanmean([kappa(X[k, 0], X[k, v]) for v in V]) - np.nanmean([kappa(X[k, p], X[k, q]) for p, q in combinations(V, 2)])
        add("kappa_original_minus_validators", "original", diff(np.arange(n)), ci(diff, n))
        cons = L[raters[1:]].notna().all(axis=1) & (L[raters[1:]].nunique(axis=1) == 1) & L.original.notna()
        o, c = L.original[cons].to_numpy(object), L[raters[1]][cons].to_numpy(object)
        add("kappa_original_vs_consensus", "original", kappa(o, c), ci(lambda k: kappa(o[k], c[k]), len(o)), n=len(o), agree=(o == c).mean())
        add("original_matches_any_validator", "original", L[raters[1:]].eq(L.original, axis=0).any(axis=1)[L.original.notna()].mean())
 
res = pd.DataFrame(rows)
res.to_csv(OUT / "agreement_summary.csv", index=False)
pd.concat(conf).to_csv(OUT / "confusion.csv", index=False)
 
# disagreements: severity 3 = opposite direction, 2 = original vs. validator consensus, 1 = any
wide = pd.concat({v: d for v, d in lab.items()}, axis=1)
def severity(r):
    sev = 0
    for var, s in SCALES.items():
        x = r[var].dropna()
        v = x.drop("original", errors="ignore")
        sev += 3 * (len(set(x.map(s)) - {0}) > 1) + 2 * (len(v) > 1 and v.nunique() == 1 and "original" in x and v.iloc[0] != x.original) + (x.nunique() > 1)
    return sev
wide.columns = [f"{v}_{r}" for v, r in wide.columns]
wide.insert(0, "severity", pd.concat({v: d for v, d in lab.items()}, axis=1).apply(severity, axis=1))
wide.join(key.filter(["id", "text"])).query("severity > 0").sort_values("severity", ascending=False).to_csv(OUT / "disagreements.csv")
print(res.round(3).to_string(index=False))
 