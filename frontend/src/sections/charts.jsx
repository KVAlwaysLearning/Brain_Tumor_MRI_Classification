// Shared model-result charts used by the Models and Dashboard sections
import Plot, { ax } from "../lib/Plot";
import { C, CLASSES, DISPLAY } from "../lib/theme";

export const SCORE_KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr", "pr_auc_macro", "mcc", "cohen_kappa"];
export const ALL_METRICS = [...SCORE_KEYS, "log_loss", "ece", "tumor_miss_rate", "false_alarm_rate"];
export const LOWER_BETTER = new Set(["log_loss", "ece", "tumor_miss_rate", "false_alarm_rate"]);
export const LABEL = { accuracy: "Accuracy", balanced_accuracy: "Balanced acc.", precision_macro: "Precision (macro)", recall_macro: "Recall (macro)", f1_macro: "F1 (macro)",
  f1_weighted: "F1 (weighted)", roc_auc_ovr: "ROC-AUC", pr_auc_macro: "PR-AUC", mcc: "MCC", cohen_kappa: "Cohen's κ", log_loss: "Log loss", ece: "ECE",
  tumor_miss_rate: "Tumor miss rate", false_alarm_rate: "False alarm rate" };
export const VAR_COLORS = ["#34d399", "#22d3ee", "#fbbf24", "#a78bfa", "#fb7185", "#f0abfc"];

export function HistoryChart({ variant, v, height = 300 }) {
  // concatenate stages (frozen -> fine-tune) and mark the boundary
  const cat = (k) => v.history.flatMap((h) => h[k] || []);
  const bounds = []; let off = 0;
  v.history.forEach((h, i) => { if (i > 0) bounds.push(off + 0.5); off += h.loss.length; });
  const ep = cat("loss").map((_, i) => i + 1);
  const shapes = bounds.map((x) => ({ type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper", line: { color: C.faint, dash: "dash" } }));
  const ann = bounds.map((x) => ({ x, y: 1, yref: "paper", text: "fine-tune →", showarrow: false, xanchor: "left", font: { size: 10, color: C.muted } }));
  return (
    <div className="grid-2 tight">
      {[["accuracy", "Accuracy"], ["loss", "Loss"]].map(([k, t]) => (
        <Plot key={k} height={height} data={[
          { type: "scatter", mode: "lines", name: "train", x: ep, y: cat(k), line: { color: C.emerald2, width: 2 } },
          { type: "scatter", mode: "lines", name: "validation", x: ep, y: cat(`val_${k}`), line: { color: "#22d3ee", width: 2 } },
        ]} layout={{ title: { text: `${t} — ${variant}`, font: { size: 12 } }, shapes, annotations: ann, xaxis: ax({ title: { text: "Epoch" } }), margin: { t: 34, l: 50, r: 10, b: 60 } }} />
      ))}
    </div>
  );
}

export function Confusion({ matrix, title, height = 320 }) {
  const rows = matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  const z = matrix.map((r, i) => r.map((v) => v / rows[i]));
  return (
    <Plot height={height} data={[{ type: "heatmap", x: CLASSES.map((c) => DISPLAY[c]), y: CLASSES.map((c) => DISPLAY[c]), z, zmin: 0, zmax: 1,
      colorscale: [[0, "#030a07"], [1, "#10b981"]], showscale: false,
      text: matrix.map((r, i) => r.map((v, j) => `${z[i][j].toFixed(2)}<br>(${v})`)), texttemplate: "%{text}",
      hovertemplate: "true %{y} → predicted %{x}<br>%{text}<extra></extra>" }]}
      layout={{ title: { text: title, font: { size: 12 } }, xaxis: ax({ title: { text: "Predicted" } }), yaxis: ax({ title: { text: "True" }, autorange: "reversed" }), margin: { t: 34, l: 90, b: 60, r: 10 } }} />
  );
}

export function MetricBars({ variants, models, split = "test", keys = SCORE_KEYS, height = 340, title }) {
  return (
    <Plot height={height} data={variants.map((v, i) => ({ type: "bar", name: v, x: keys.map((k) => LABEL[k]), y: keys.map((k) => models.variants[v][split][k]),
      marker: { color: VAR_COLORS[i % VAR_COLORS.length] }, text: keys.map((k) => models.variants[v][split][k]?.toFixed(3)), textposition: "outside", textfont: { size: 9 } }))}
      layout={{ barmode: "group", title: title ? { text: title, font: { size: 12 } } : undefined, yaxis: ax({ range: [Math.min(0, ...variants.flatMap((v) => keys.map((k) => models.variants[v][split][k] ?? 0))) - 0.05, 1.1] }), margin: { t: title ? 34 : 10, b: 90, l: 50, r: 10 } }} />
  );
}
