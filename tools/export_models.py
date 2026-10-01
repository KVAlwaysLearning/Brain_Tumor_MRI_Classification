"""Export model results + explainer assets for the web app (needs TensorFlow; run once, offline).

Outputs
  frontend/public/data/models.json          metrics, curves, histories, grids, CI, efficiency, McNemar (H3)
  frontend/public/data/graph_<model>.json   schematic: every layer, shape, parameters, config, inputs
  backend/assets/predict_mobilenetv2.tflite prediction model (best model)
  backend/assets/<model>_explain.tflite     one output per layer (conv, BN and activation kept separate)
  backend/assets/<model>_graph.json         same graph, used by the server to build the network view
  backend/assets/<model>_weights.npz        line weights (fp16): conv kernels summed over h x w, dense, depthwise, BN scale
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json, glob, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
                             log_loss, roc_curve, precision_recall_curve, confusion_matrix)
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(os.environ.get("EXPORT_ROOT", "."))   # folder holding the dataset, checkpoints and notebook outputs
CLEAN, HIST, MET = ROOT / "clean", ROOT / "hist", ROOT / "fc" / "metrics"
APP = Path(__file__).resolve().parents[1]
DATA, ASSETS = APP / "frontend" / "public" / "data", APP / "backend" / "assets"
DATA.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]; K = len(CLASSES); NO = 2
IMG = 224


def r(x, d=4):
    x = float(x)
    return None if np.isnan(x) else round(x, d)


# ------------------------------------------------------------------ data (training preprocessing, verified identical)
def to_gray3(x):
    x = tf.cast(x, tf.int32)
    g = tf.cast(tf.bitwise.right_shift(x[..., 0] * 19595 + x[..., 1] * 38470 + x[..., 2] * 7471 + 32768, 16), tf.uint8)
    return tf.stack([g, g, g], -1)


def load(p):
    img = to_gray3(tf.io.decode_jpeg(tf.io.read_file(p), channels=3, dct_method="INTEGER_ACCURATE"))
    img = tf.cast(tf.clip_by_value(tf.round(tf.image.resize(img, [IMG, IMG], antialias=True)), 0, 255), tf.uint8)
    return tf.cast(img, tf.float32) / 255.0


def split_ds(split):
    files, y = [], []
    for i, c in enumerate(CLASSES):
        fs = sorted(glob.glob(str(CLEAN / split / c / "*.jpg"))); files += fs; y += [i] * len(fs)
    return tf.data.Dataset.from_tensor_slices(files).map(load).batch(32), np.array(y)


ds_val, y_val = split_ds("valid")
ds_test, y_test = split_ds("test")


# ------------------------------------------------------------------ architectures (identical to the notebook)
def build_cnn(filters=(32, 64, 128, 256), dense_units=256, l2=1e-4, dropout=0.4):
    reg = keras.regularizers.l2(l2); inp = keras.Input((IMG, IMG, 3), name="image"); x = inp
    for b, f in enumerate(filters, 1):
        for c in (1, 2):
            x = layers.Conv2D(f, 3, strides=2 if (b, c) == (1, 1) else 1, padding="same", use_bias=False,
                              kernel_regularizer=reg, name=f"block{b}_conv{c}")(x)
            x = layers.BatchNormalization(momentum=0.9, name=f"block{b}_bn{c}")(x)
            x = layers.Activation("relu", name=f"block{b}_relu{c}")(x)
        x = layers.MaxPooling2D(2, name=f"block{b}_pool")(x)
        if b >= 3:
            x = layers.SpatialDropout2D(0.1, name=f"block{b}_sdrop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, use_bias=False, kernel_regularizer=reg, name="fc")(x)
    x = layers.BatchNormalization(momentum=0.9, name="fc_bn")(x)
    x = layers.Activation("relu", name="fc_relu")(x)
    x = layers.Dropout(dropout, name="fc_dropout")(x)
    return keras.Model(inp, layers.Dense(K, activation="softmax", name="predictions")(x), name="custom_cnn")


def imagenet_layout(constructor):
    """Build a backbone with its ImageNet-specific layers but without downloading weights (checkpoints supply them)."""
    import keras.src.models.functional as F
    mod = __import__(constructor.__module__, fromlist=["file_utils"])
    g, lw = mod.file_utils.get_file, F.Functional.load_weights
    mod.file_utils.get_file = lambda *a, **k: "unused"; F.Functional.load_weights = lambda self, *a, **k: None
    try:
        return constructor(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    finally:
        mod.file_utils.get_file, F.Functional.load_weights = g, lw


def build_tl(arch):
    scale, offset, ctor = {"efficientnetb0": (255.0, 0.0, keras.applications.EfficientNetB0),
                           "mobilenetv2": (2.0, -1.0, keras.applications.MobileNetV2)}[arch]
    inp = keras.Input((IMG, IMG, 3), name="image")
    x = layers.Rescaling(scale, offset=offset, name="to_backbone_range")(inp)
    bb = imagenet_layout(ctor); bb.trainable = False
    x = bb(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="head_dropout1")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)
    return keras.Model(inp, layers.Dense(K, activation="softmax", name="predictions")(x), name=f"{arch}_transfer")


def best_ft(arch):
    g = pd.read_csv(MET / f"{arch}_finetune_grid.csv")
    return int(round(g.sort_values("val_f1_macro", ascending=False).iloc[0].unfreeze_fraction * 100))


CK1, CK2 = ROOT / "ckpt", ROOT / "run2" / "checkpoints"
EFT, MFT = best_ft("efficientnetb0"), best_ft("mobilenetv2")
VARIANTS = {   # variant -> (family, builder, weights, history runs)
    "Custom CNN — baseline": ("Custom CNN", build_cnn, CK2 / "custom_cnn_baseline.weights.h5", ["custom_cnn_baseline"]),
    "Custom CNN — tuned": ("Custom CNN", build_cnn, CK2 / "custom_cnn_tuned.weights.h5", ["custom_cnn_tuned"]),
    "EfficientNetB0 — frozen backbone": ("EfficientNetB0", lambda: build_tl("efficientnetb0"), CK1 / "efficientnetb0_stage1_frozen.weights.h5", ["efficientnetb0_stage1_frozen"]),
    "EfficientNetB0 — fine-tuned": ("EfficientNetB0", lambda: build_tl("efficientnetb0"), CK1 / f"efficientnetb0_finetune_{EFT}pct.weights.h5", ["efficientnetb0_stage1_frozen", f"efficientnetb0_finetune_{EFT}pct"]),
    "MobileNetV2 — frozen backbone": ("MobileNetV2", lambda: build_tl("mobilenetv2"), CK1 / "mobilenetv2_stage1_frozen.weights.h5", ["mobilenetv2_stage1_frozen"]),
    "MobileNetV2 — fine-tuned": ("MobileNetV2", lambda: build_tl("mobilenetv2"), CK1 / f"mobilenetv2_finetune_{MFT}pct.weights.h5", ["mobilenetv2_stage1_frozen", f"mobilenetv2_finetune_{MFT}pct"]),
}
FINAL = {"Custom CNN": "Custom CNN — tuned", "EfficientNetB0": "EfficientNetB0 — fine-tuned", "MobileNetV2": "MobileNetV2 — fine-tuned"}


# ------------------------------------------------------------------ metrics (same definitions as the notebook)
def ece(y, p, bins=10):
    conf, ok = p.max(1), p.argmax(1) == y; e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def metrics(y, p):
    yp = p.argmax(1); oh = np.eye(K)[y]; tumor, healthy = y != NO, y == NO
    return {k: r(v) for k, v in {
        "accuracy": accuracy_score(y, yp), "balanced_accuracy": balanced_accuracy_score(y, yp),
        "precision_macro": precision_score(y, yp, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yp, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yp, average="macro", zero_division=0), "f1_weighted": f1_score(y, yp, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y, p, multi_class="ovr", labels=list(range(K))), "pr_auc_macro": average_precision_score(oh, p, average="macro"),
        "mcc": matthews_corrcoef(y, yp), "cohen_kappa": cohen_kappa_score(y, yp),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1), labels=list(range(K))), "ece": ece(y, p),
        "tumor_miss_rate": np.mean(yp[tumor] == NO), "false_alarm_rate": np.mean(yp[healthy] != NO)}.items()}


def curves(y, p):
    oh = np.eye(K)[y]; out = {"roc": {}, "pr": {}}
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(oh[:, i], p[:, i]); pr, rc, _ = precision_recall_curve(oh[:, i], p[:, i])
        out["roc"][c] = {"x": [r(v, 4) for v in fpr], "y": [r(v, 4) for v in tpr], "auc": r(roc_auc_score(oh[:, i], p[:, i]))}
        out["pr"][c] = {"x": [r(v, 4) for v in rc], "y": [r(v, 4) for v in pr], "ap": r(average_precision_score(oh[:, i], p[:, i]))}
    conf, ok = p.max(1), p.argmax(1) == y; cal = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): cal.append({"conf": r(conf[m].mean()), "acc": r(ok[m].mean()), "n": int(m.sum())})
    out["calibration"] = cal
    out["confidence_hist"] = np.histogram(conf, bins=np.linspace(0, 1, 21))[0].tolist()
    return out


def per_class(y, yp):
    return {m: [r(v) for v in f(y, yp, labels=list(range(K)), average=None, zero_division=0)]
            for m, f in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]}


reference = pd.read_csv(MET / "all_metrics_all_variants.csv", index_col=[0, 1])
out = {"classes": CLASSES, "variants": {}, "final": FINAL}
built = {}
for v, (fam, builder, wpath, runs) in VARIANTS.items():
    keras.utils.set_random_seed(42)
    m = builder(); m.load_weights(str(wpath))
    pv, pt = m.predict(ds_val, verbose=0), m.predict(ds_test, verbose=0)
    mv, mt = metrics(y_val, pv), metrics(y_test, pt)
    ref_v, ref_t = reference.loc[("val", v)], reference.loc[("test", v)]
    assert abs(mv["accuracy"] - ref_v["accuracy"]) < 1e-3 and abs(mt["accuracy"] - ref_t["accuracy"]) < 1e-3, f"{v} does not match the run"
    hist = [{"stage": run, **{k: [r(x, 5) for x in vals] for k, vals in json.load(open(HIST / f"{run}.json")).items()}} for run in runs]
    out["variants"][v] = {"family": fam, "params": int(m.count_params()), "val": mv, "test": mt,
                          "confusion": confusion_matrix(y_test, pt.argmax(1), labels=list(range(K))).tolist(),
                          "confusion_val": confusion_matrix(y_val, pv.argmax(1), labels=list(range(K))).tolist(),
                          "per_class": per_class(y_test, pt.argmax(1)), "curves": curves(y_test, pt), "history": hist}
    print(f"{v:<34} val acc {mv['accuracy']:.4f} | test acc {mt['accuracy']:.4f}  ✓ matches the notebook run")
    if v in FINAL.values():
        built[fam] = m

# ------------------------------------------------------------------ tables from the notebook run
csv = lambda f: pd.read_csv(MET / f).round(5).replace({np.nan: None}).to_dict("records")
out["custom_grid"] = csv("custom_cnn_grid_search.csv")
out["custom_cv"] = csv("custom_cnn_cross_validation.csv")
out["finetune_grids"] = {"EfficientNetB0": csv("efficientnetb0_finetune_grid.csv"), "MobileNetV2": csv("mobilenetv2_finetune_grid.csv")}
out["comparison"] = csv("model_comparison.csv")
out["mcnemar"] = csv("h3_mcnemar.csv")
ci = pd.read_csv(MET / "test_bootstrap_ci.csv")
ci.columns = ["model", "metric", "estimate", "ci_low", "ci_high"]
out["bootstrap_ci"] = ci.round(4).to_dict("records")
out["test_counts"] = np.bincount(y_test, minlength=K).tolist()
out["val_counts"] = np.bincount(y_val, minlength=K).tolist()
json.dump(out, open(DATA / "models.json", "w"), separators=(",", ":"))
print("models.json", round((DATA / "models.json").stat().st_size / 1e3), "KB")


# ------------------------------------------------------------------ explainer assets for the three final models
def layer_inputs(layer):
    try:
        return [t._keras_history.operation.name for t in layer._inbound_nodes[0].input_tensors]
    except (IndexError, AttributeError):
        return []


def layer_config(l):
    c = l.get_config(); keep = {}
    for k in ["filters", "kernel_size", "strides", "padding", "activation", "units", "rate", "pool_size", "epsilon",
              "momentum", "depth_multiplier", "use_bias", "scale", "offset", "target_shape", "axis", "padding"]:
        if k in c and c[k] is not None:
            val = c[k]
            if isinstance(val, (list, tuple)) and len(val) > 6: val = f"{len(val)} values"
            keep[k] = val if isinstance(val, (int, float, str, bool, list, tuple)) else str(val)
    return keep


def kind_of(l):
    if isinstance(l, layers.DepthwiseConv2D): return "depthwise"
    if isinstance(l, (layers.Conv2D, layers.Dense)): return "weight"
    if isinstance(l, layers.BatchNormalization): return "bn"
    if isinstance(l, (layers.Add, layers.Multiply, layers.Concatenate)): return "merge"
    return "pass"


def block_of(name, fam):
    import re
    if fam == "Custom CNN":
        m = re.match(r"(block\d+)", name); return m.group(1) if m else ("input" if name == "image" else "head")
    if fam == "MobileNetV2":
        if name in ("image", "to_backbone_range"): return "input"
        m = re.match(r"(block_\d+)", name)
        if m: return m.group(1)
        if name.startswith("expanded_conv"): return "block_0"
        if name.startswith(("Conv1", "bn_Conv1", "Conv1_relu")): return "stem"
        if name.startswith(("Conv_1", "out_relu")): return "top"
        return "head"
    if name in ("image", "to_backbone_range"): return "input"
    m = re.match(r"(block\d[a-z])", name)
    if m: return m.group(1)
    if name.startswith(("stem", "rescaling", "normalization")): return "stem"
    if name.startswith("top"): return "top"
    return "head"


def explainer_graph(m, fam):
    """Flatten outer + nested backbone layers in execution order; keep true input links for every layer."""
    inp = keras.Input((IMG, IMG, 3), name="image"); t = inp; outs, nodes = [], []
    alias = {}                                   # nested InputLayer -> the outer layer feeding the backbone
    prev_outer = "image"
    for l in m.layers[1:]:
        if isinstance(l, keras.Model):
            alias[l.layers[0].name] = prev_outer
            sub = keras.Model(l.inputs, [ll.output for ll in l.layers[1:]]); vals = sub(t)
            vals = vals if isinstance(vals, list) else [vals]
            for ll, v in zip(l.layers[1:], vals):
                nodes.append((ll, [alias.get(n, n) for n in layer_inputs(ll)])); outs.append(v)
            t = vals[-1]; alias[l.name] = l.layers[-1].name; prev_outer = l.layers[-1].name
        else:
            t = l(t); outs.append(t); nodes.append((l, [alias.get(n, n) for n in layer_inputs(l)])); prev_outer = l.name
    em = keras.Model(inp, outs)
    graph = [{"name": "image", "type": "InputLayer", "kind": "input", "block": "input", "inputs": [], "out": -1,
              "shape": [IMG, IMG, 3], "params": 0, "config": {}}]
    for i, (l, ins) in enumerate(nodes):
        shape = list(outs[i].shape[1:])
        graph.append({"name": l.name, "type": type(l).__name__, "kind": kind_of(l), "block": block_of(l.name, fam),
                      "inputs": ins, "out": i, "shape": [int(s) for s in shape], "params": int(l.count_params()),
                      "config": layer_config(l)})
    return em, graph


def line_weights(graph, m):
    lyr = {}
    for l in m.layers:
        lyr[l.name] = l
        if isinstance(l, keras.Model):
            for ll in l.layers: lyr[ll.name] = ll
    W = {}
    for g in graph:
        l = lyr.get(g["name"])
        if l is None or not l.get_weights(): continue
        w = l.get_weights()
        if g["kind"] == "depthwise": W[g["name"]] = w[0].sum(axis=(0, 1))[:, 0]
        elif isinstance(l, layers.Conv2D): W[g["name"]] = w[0].sum(axis=(0, 1))
        elif isinstance(l, layers.Dense): W[g["name"]] = w[0]
        elif g["kind"] == "bn":
            cfg = l.get_config(); names = [v.path.split("/")[-1] for v in l.weights]; d = dict(zip(names, w))
            gamma = d.get("gamma", np.ones_like(d["moving_variance"]))
            W[g["name"]] = gamma / np.sqrt(d["moving_variance"] + cfg["epsilon"])
    return {k: v.astype(np.float16) for k, v in W.items()}


slug = {"Custom CNN": "custom_cnn", "EfficientNetB0": "efficientnetb0", "MobileNetV2": "mobilenetv2"}
xtest = next(iter(ds_test.unbatch().batch(1)))
for fam, m in built.items():
    s = slug[fam]
    em, graph = explainer_graph(m, fam)
    ref = [np.asarray(v) for v in em(xtest)]
    em.export(str(ROOT / f"em_{s}"), verbose=False)
    open(ASSETS / f"{s}_explain.tflite", "wb").write(tf.lite.TFLiteConverter.from_saved_model(str(ROOT / f"em_{s}")).convert())
    run = Interpreter(str(ASSETS / f"{s}_explain.tflite")).get_signature_runner()
    res = run(image=xtest.numpy()); lite = [res[k] for k in sorted(res, key=lambda k: int(k.split("_")[-1]))]
    worst = max(np.abs(a - b).max() / (np.abs(b).max() + 1e-6) for a, b in zip(lite, ref))
    assert len(lite) == len(graph) - 1 and worst < 1e-3, f"{fam}: explainer mismatch ({worst})"
    W = line_weights(graph, m)
    np.savez_compressed(ASSETS / f"{s}_weights.npz", **W)
    meta = {"family": fam, "slug": s, "layers": graph, "total_params": int(m.count_params()),
            "n_layers": len(graph), "final_variant": FINAL[fam]}
    for target in (ASSETS / f"{s}_graph.json", DATA / f"graph_{s}.json"):
        json.dump(meta, open(target, "w"), separators=(",", ":"))
    print(f"{fam:<15} explainer layers {len(graph) - 1} verified (worst rel diff {worst:.1e}) | weights "
          f"{(ASSETS / f'{s}_weights.npz').stat().st_size / 1e6:.1f} MB | tflite {(ASSETS / f'{s}_explain.tflite').stat().st_size / 1e6:.1f} MB")
    if fam == "MobileNetV2":
        m.export(str(ROOT / "sm_pred"), verbose=False)
        open(ASSETS / "predict_mobilenetv2.tflite", "wb").write(tf.lite.TFLiteConverter.from_saved_model(str(ROOT / "sm_pred")).convert())

cfg = json.load(open(ROOT / "fc" / "models" / "deployment_config.json"))
json.dump({"class_names": CLASSES, "display_names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
           "input_checks": cfg["input_checks"], "img_size": IMG, "foreground_threshold": 15,
           "best_model": "MobileNetV2 — fine-tuned"}, open(ASSETS / "config.json", "w"), indent=1)
print("done")
