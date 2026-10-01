"""Export the data behind the dataset, EDA (section 4) and hypothesis charts as JSON + thumbnails.

Inputs  : raw dataset, cleaned dataset + clean_metadata.csv, near-duplicate reports from the notebook run.
Outputs : frontend/public/data/{eda,dedup,hypotheses}.json and frontend/public/img/*.jpg
"""
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(os.environ.get("EXPORT_ROOT", "."))   # folder holding the dataset, checkpoints and notebook outputs
RAW, CLEAN = ROOT / "project_data", ROOT / "clean"
REPORTS = ROOT / "fc" / "reports"
OUT = Path(__file__).resolve().parents[1] / "frontend" / "public"
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "img").mkdir(parents=True, exist_ok=True)

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
DISPLAY = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No Tumor", "pituitary": "Pituitary"}
SPLITS = ["train", "valid", "test"]
FG = 15
SEED = 42


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def thumb(path, out_name, size=180, quality=82):
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    im.save(OUT / "img" / out_name, quality=quality)
    return f"img/{out_name}"


meta = pd.read_csv(CLEAN / "clean_metadata.csv")
meta["clean_path"] = [str(CLEAN / s / l / f) for s, l, f in zip(meta.split, meta.label, meta.filename)]

# ------------------------------------------------------------------ raw (before) counts
raw_counts = {s: {c: len(list((RAW / s / c).glob("*.jpg"))) for c in CLASSES} for s in SPLITS}
after_counts = {s: {c: int(((meta.split == s) & (meta.label == c)).sum()) for c in CLASSES} for s in SPLITS}

# ------------------------------------------------------------------ per-image table (column arrays = compact)
cols = ["mean_intensity", "std_intensity", "foreground_ratio", "sharpness", "entropy", "file_size_kb",
        "channel_diff", "aspect_ratio"]
images = {"split": meta.split.tolist(), "label": meta.label.tolist(),
          **{c: [r(v, 4) for v in meta[c]] for c in cols}}

# resolution / mode summary (chart 3)
res_counts = (meta.width.astype(str) + "×" + meta.height.astype(str)).value_counts().to_dict()
mode_counts = (meta["mode"] + " / " + np.where(meta.channel_diff < 1, "grayscale content", "colour content")).value_counts().to_dict()

# ------------------------------------------------------------------ chart 10: tissue-pixel density per class
grid = np.arange(FG + 1, 256)
tissue_kde = {}
for c in CLASSES:
    paths = meta[meta.label == c].clean_path.sample(min(60, (meta.label == c).sum()), random_state=SEED)
    pix = np.concatenate([np.asarray(Image.open(p).convert("L").resize((160, 160))).ravel() for p in paths])
    pix = pix[pix > FG].astype(float)
    sub = np.random.default_rng(SEED).choice(pix, min(40000, len(pix)), replace=False)
    kde = stats.gaussian_kde(sub, bw_method=0.8 * stats.gaussian_kde(sub).factor)
    tissue_kde[c] = [r(v, 6) for v in kde(grid)]

# ------------------------------------------------------------------ chart 13: mean image per class + difference
size = 96
means = {}
for c in CLASSES:
    paths = meta[meta.label == c].clean_path.sample(min(200, (meta.label == c).sum()), random_state=SEED)
    means[c] = np.mean([np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32) for p in paths], 0)
overall = np.mean(list(means.values()), 0)
mean_images = {c: {"mean": np.round(m, 1).tolist(), "diff": np.round(m - overall, 1).tolist()} for c, m in means.items()}

# ------------------------------------------------------------------ chart 5: sample grid (also explainer sample images)
samples = {}
for c in CLASSES:
    samp = meta[meta.label == c].sample(6, random_state=SEED)
    samples[c] = [thumb(p, f"sample_{c}_{i}.jpg") for i, p in enumerate(samp.clean_path)]

explainer_samples = []
test = meta[meta.split == "test"]
for c in CLASSES:
    for i, p in enumerate(test[test.label == c].sample(2, random_state=SEED).clean_path):
        name = f"test_{c}_{i}.jpg"
        Image.open(p).save(OUT / "img" / name, quality=95)       # full-size copy: used as an upload sample
        explainer_samples.append({"label": c, "src": f"img/{name}"})

# ------------------------------------------------------------------ chart 14: Spearman correlation
feat = ["mean_intensity", "std_intensity", "foreground_ratio", "sharpness", "entropy", "file_size_kb", "max_intensity", "label_idx"]
corr = meta[feat].corr(method="spearman")

# class x split chi-square (chart 12)
ct = pd.crosstab(meta.label, meta.split).reindex(index=CLASSES, columns=SPLITS)
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)

eda = {
    "classes": CLASSES, "display": DISPLAY, "splits": SPLITS,
    "counts": {"before": raw_counts, "after": after_counts},
    "images": images, "resolution_counts": res_counts, "mode_counts": mode_counts,
    "tissue_kde": {"x": grid.tolist(), "y": tissue_kde},
    "mean_images": mean_images, "samples": samples, "explainer_samples": explainer_samples,
    "correlation": {"features": feat, "matrix": [[r(v, 3) for v in row] for row in corr.values]},
    "chi2_class_split": {"chi2": r(chi2, 2), "dof": int(dof), "p": float(p_chi)},
    "spearman_fg_brightness": r(stats.spearmanr(meta.foreground_ratio, meta.mean_intensity)[0], 3),
    "n_raw": int(sum(sum(v.values()) for v in raw_counts.values())), "n_clean": int(len(meta)),
    "colour_images": int((meta.channel_diff >= 1).sum()),
}
json.dump(eda, open(OUT / "data" / "eda.json", "w"), separators=(",", ":"))

# ------------------------------------------------------------------ de-duplication
pairs = pd.read_csv(REPORTS / "near_duplicate_pairs.csv")
removals = pd.read_csv(REPORTS / "near_duplicate_removals.csv")
groups = pd.read_csv(REPORTS / "duplicate_groups_by_split_combination.csv")
sim = np.load(ROOT / "sim.npy")                      # pairwise pHash similarity (same algorithm as the notebook)
best = np.where(sim > 100, -1, sim).max(axis=1)

pair_summary = pairs.groupby("pair_type").agg(pairs=("similarity_pct", "size"), mean=("similarity_pct", "mean"),
                                              min=("similarity_pct", "min")).reset_index()
removed_by = removals.groupby(["removed_split", "label"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)

examples = []
cross = pairs[pairs.split_a != pairs.split_b]
pick = pd.concat([cross.iloc[[0, 40, 120]], cross[cross.similarity_pct < 97].head(1)])
for i, rw in enumerate(pick.itertuples()):
    a = thumb(RAW / rw.split_a / rw.label_a / rw.file_a, f"dup_{i}_a.jpg", 200)
    b = thumb(RAW / rw.split_b / rw.label_b / rw.file_b, f"dup_{i}_b.jpg", 200)
    examples.append({"a": a, "b": b, "split_a": rw.split_a, "split_b": rw.split_b, "label": rw.label_a,
                     "similarity": r(rw.similarity_pct, 2)})

dedup = {
    "threshold": 95.0, "n_pairs": int(len(pairs)), "n_removed": int(len(removals)),
    "pair_summary": pair_summary.round(2).to_dict("records"),
    "groups": groups.round(2).to_dict("records"),
    "removal_reasons": removals.reason.value_counts().to_dict(),
    "removed_by_split_class": {s: removed_by.loc[s].astype(int).to_dict() if s in removed_by.index else {c: 0 for c in CLASSES} for s in SPLITS},
    "best_match_hist": np.histogram(best, bins=np.arange(40, 100.5, 1))[0].tolist(), "best_match_edges": list(range(40, 101)),
    "examples": examples,
}
json.dump(dedup, open(OUT / "data" / "dedup.json", "w"), separators=(",", ":"))


# ------------------------------------------------------------------ hypotheses H1 / H2 (H3 comes from the model export)
def kruskal(col):
    g = {c: meta.loc[meta.label == c, col].values for c in CLASSES}
    H, p = stats.kruskal(*g.values())
    n, k = len(meta), len(CLASSES)
    post = []
    for a, b in combinations(CLASSES, 2):
        _, pu = stats.mannwhitneyu(g[a], g[b], alternative="two-sided")
        post.append({"a": a, "b": b, "median_a": r(np.median(g[a])), "median_b": r(np.median(g[b])),
                     "p_bonferroni": float(min(pu * 6, 1.0))})
    return {"H": r(H, 2), "p": float(p), "eps2": r((H - k + 1) / (n - k), 3),
            "shapiro_p": {c: float(stats.shapiro(np.random.default_rng(SEED).choice(v, min(500, len(v)), replace=False)).pvalue) for c, v in g.items()},
            "levene_p": float(stats.levene(*g.values()).pvalue), "posthoc": post,
            "medians": {c: r(np.median(v)) for c, v in g.items()}}


hyp = {"H1": {"variable": "mean_intensity", **kruskal("mean_intensity")},
       "H2": {"variable": "foreground_ratio", **kruskal("foreground_ratio")}}
json.dump(hyp, open(OUT / "data" / "hypotheses.json", "w"), separators=(",", ":"))

for f in ["eda", "dedup", "hypotheses"]:
    print(f, round((OUT / "data" / f"{f}.json").stat().st_size / 1e3), "KB")
print("H1", hyp["H1"]["H"], hyp["H1"]["p"], "| H2", hyp["H2"]["H"], hyp["H2"]["p"])
