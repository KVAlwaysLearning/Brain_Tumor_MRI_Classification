"""Prediction (best model only) and the layer-by-layer explainer, both on LiteRT (no TensorFlow at runtime)."""
import base64
import json
import threading
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CONFIG = json.loads((ASSETS / "config.json").read_text())
CLASSES = CONFIG["class_names"]
MODELS = {"custom_cnn": "Custom CNN", "efficientnetb0": "EfficientNetB0", "mobilenetv2": "MobileNetV2"}
THUMB = 12                                     # feature-map thumbnail size (pixels per side)


class Predictor:
    """MobileNetV2 (the selected best model). Loaded once at start-up."""

    def __init__(self):
        self.it = Interpreter(str(ASSETS / "predict_mobilenetv2.tflite"), num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.lock = threading.Lock()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.lock:
            self.it.set_tensor(self.inp, x)
            self.it.invoke()
            return self.it.get_tensor(self.out)[0].astype(float)


class Explainer:
    """Runs one explainer model at a time (memory stays well under 512 MB on Render's free plan)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = None                     # (slug, runner, graph, weights)

    def _load(self, slug):
        if self.loaded and self.loaded[0] == slug:
            return self.loaded
        self.loaded = None                     # free the previous model before loading the next one
        it = Interpreter(str(ASSETS / f"{slug}_explain.tflite"), num_threads=1)
        it.allocate_tensors()
        sig = it._get_full_signature_list()["serving_default"]
        out_idx = [sig["outputs"][k] for k in sorted(sig["outputs"], key=lambda s: int(s.split("_")[-1]))]
        graph = json.loads((ASSETS / f"{slug}_graph.json").read_text())
        weights = dict(np.load(ASSETS / f"{slug}_weights.npz"))
        self.loaded = (slug, (it, sig["inputs"]["image"], out_idx), graph, weights)
        return self.loaded

    def __call__(self, slug: str, x: np.ndarray, k: int):
        with self.lock:
            _, (it, in_idx, out_idx), graph, weights = self._load(slug)
            it.set_tensor(in_idx, x)
            it.invoke()
            # read one layer output at a time so only a single activation copy is alive at once
            return build_network(graph["layers"], lambda i: it.get_tensor(out_idx[i]), weights, x, k)


def _sig(v, d=4):
    return float(f"{float(v):.{d}g}")


def _thumbnails(fmaps: np.ndarray) -> list[tuple[str, float, float]]:
    """(H, W, n) feature maps -> per map: base64 uint8 thumbnail (<= THUMB x THUMB) block-averaged, min, max."""
    h, w, _ = fmaps.shape
    ys = np.unique(np.linspace(0, h, min(THUMB, h) + 1).astype(int))[:-1]
    xs = np.unique(np.linspace(0, w, min(THUMB, w) + 1).astype(int))[:-1]
    sums = np.add.reduceat(np.add.reduceat(fmaps.astype(np.float64), ys, axis=0), xs, axis=1)
    cnt = np.outer(np.diff(np.append(ys, h)), np.diff(np.append(xs, w)))[..., None]
    small = sums / cnt
    out = []
    for j in range(small.shape[-1]):
        m = small[..., j]; lo, hi = float(m.min()), float(m.max())
        norm = np.zeros_like(m) if hi - lo < 1e-12 else (m - lo) / (hi - lo)
        out.append((base64.b64encode((norm * 255).astype(np.uint8).tobytes()).decode(), _sig(lo), _sig(hi)))
    return out


def build_network(layers, get_output, weights, x, k):
    """For every layer: total node count, the k most-activated nodes (or all if <= k), their real values,
    feature-map thumbnails, and lines with real weights to the nodes shown in the layer's actual input layer(s).
    `get_output(i)` returns layer i's activation; layers are processed one at a time to keep memory low."""
    totals, selected, result, probs = {}, {}, [], None

    for g in layers:
        a = (x if g["out"] < 0 else get_output(g["out"]))[0]   # drop the batch dimension
        spatial = a.ndim == 3
        per_node = a.reshape(-1, a.shape[-1]).mean(0) if spatial else a.reshape(-1)
        total = int(per_node.shape[0])
        totals[g["name"]] = total
        src = g["inputs"][0] if g["inputs"] else None
        if total <= k:
            sel = list(range(total))
        elif g["kind"] in ("depthwise", "bn", "pass", "merge") and src in selected and totals[src] == total:
            sel = selected[src]                             # channel-preserving: keep the same channels in view
        else:
            sel = [int(i) for i in np.argsort(-per_node)[:k]]
        selected[g["name"]] = sel

        nodes = [{"i": i, "v": _sig(per_node[i])} for i in sel]
        if spatial and a.shape[0] * a.shape[1] > 1:
            for n, (t, lo, hi) in zip(nodes, _thumbnails(a[..., sel])):
                n["t"], n["lo"], n["hi"] = t, lo, hi

        edges = []
        for inp in g["inputs"]:
            if inp not in selected:
                continue
            s_in, w = selected[inp], weights.get(g["name"])
            if g["kind"] == "weight" and w is not None and w.ndim == 2 and w.shape[0] == totals[inp]:
                edges += [[inp, s, o, _sig(w[s, o], 3)] for s in s_in for o in sel]
            else:                                           # identity links for channel-preserving layers
                shared = set(s_in) & set(sel)
                edges += [[inp, c, c, _sig(w[c], 3) if (w is not None and w.ndim == 1) else None] for c in sorted(shared)]

        result.append({"name": g["name"], "total": total, "nodes": nodes, "edges": edges,
                       "stats": {"min": _sig(a.min()), "max": _sig(a.max()), "mean": _sig(a.mean()),
                                 "zero_frac": _sig((a == 0).mean(), 3)}})
        if g is layers[-1]:
            probs = [float(v) for v in a.reshape(-1)]       # the last layer is the softmax output
        del a
    return {"layers": result, "probabilities": probs, "predicted": int(np.argmax(probs))}

