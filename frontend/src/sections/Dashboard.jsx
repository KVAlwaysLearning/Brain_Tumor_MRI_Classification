import { useEffect, useState } from "react";
import Plot, { ax } from "../lib/Plot";
import { loadData } from "../lib/data";
import { C, CLASSES, CLASS_COLORS, DISPLAY, MODEL_COLORS, fmt, pct } from "../lib/theme";
import { Card, Loading, Section, Tabs } from "../lib/ui";
import { ALL_METRICS, Confusion, LABEL, LOWER_BETTER, SCORE_KEYS, VAR_COLORS } from "./charts";

export default function Dashboard() {
  const [m, setM] = useState(null);
  const [split, setSplit] = useState("test");
  const [curveModel, setCurveModel] = useState("MobileNetV2");
  useEffect(() => { loadData("models").then(setM); }, []);
  if (!m) return <Section id="dashboard" title="Model comparison dashboard"><Loading /></Section>;

  const V = Object.keys(m.variants);
  const FAM = Object.keys(m.final);
  const finalV = (f) => m.variants[m.final[f]];
  const comp = Object.fromEntries(m.comparison.map((r) => [r.model, r]));

  // heatmap: colour ranks within each column so green is always "better"
  const tbl = V.map((v) => ALL_METRICS.map((k) => m.variants[v][split][k]));
  const shade = tbl.map((row) => row.map((val, j) => {
    const col = tbl.map((r) => r[j]); const lo = Math.min(...col), hi = Math.max(...col);
    if (!(hi > lo)) return 0.5;
    return LOWER_BETTER.has(ALL_METRICS[j]) ? (hi - val) / (hi - lo) : (val - lo) / (hi - lo);
  }));

  const best = FAM.map((f) => ({ f, val: finalV(f).val.f1_macro })).sort((a, b) => b.val - a.val)[0].f;

  return (
    <Section id="dashboard" kicker="Model comparison" title="Every metric, every model"
      intro={`Six trained variants and the three final models. Final selection rule: highest validation macro-F1 → lowest tumor miss rate → lowest latency. Selected: ${best}.`}>

      <Card title="Dashboard 1 · All metrics heatmap" subtitle="Green = best in column (lower is better for log loss, ECE and the two error rates)"
        right={<Tabs value={split} onChange={setSplit} tabs={[{ value: "val", label: "Validation" }, { value: "test", label: "Test" }]} />}>
        <Plot height={430} data={[{ type: "heatmap", x: ALL_METRICS.map((k) => LABEL[k]), y: V, z: shade, zmin: 0, zmax: 1, showscale: false,
          colorscale: [[0, "#4c0519"], [0.5, "#3f3a0b"], [1, "#065f46"]], text: tbl.map((r) => r.map((v) => fmt(v))), texttemplate: "%{text}", textfont: { size: 11 },
          hovertemplate: "%{y}<br>%{x}: %{text}<extra></extra>" }]} layout={{ yaxis: ax({ autorange: "reversed" }), margin: { l: 230, b: 90, t: 10, r: 10 } }} />
      </Card>

      <Card title="Dashboard 2 · Overall quality metrics for all variants" right={<Tabs value={split} onChange={setSplit} tabs={[{ value: "val", label: "Validation" }, { value: "test", label: "Test" }]} />}>
        <Plot height={420} data={V.map((v, i) => ({ type: "bar", name: v, x: SCORE_KEYS.map((k) => LABEL[k]), y: SCORE_KEYS.map((k) => m.variants[v][split][k]), marker: { color: VAR_COLORS[i] } }))}
          layout={{ barmode: "group", yaxis: ax({ range: [0, 1.05] }), margin: { t: 10, b: 110, l: 50, r: 10 } }} />
      </Card>

      <Card title="Dashboard 3 · Per-class precision, recall and F1 (final models, test)">
        <div className="grid-3 tight">
          {[["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]].map(([k, t]) => (
            <Plot key={k} height={300} data={FAM.map((f) => ({ type: "bar", name: f, x: CLASSES.map((c) => DISPLAY[c]), y: finalV(f).per_class[k], marker: { color: MODEL_COLORS[f] }, text: finalV(f).per_class[k].map((v) => v.toFixed(2)), textposition: "outside", textfont: { size: 9 } }))}
              layout={{ barmode: "group", title: { text: t, font: { size: 12 } }, yaxis: ax({ range: [0, 1.12] }), margin: { t: 30, b: 70, l: 40, r: 6 } }} />
          ))}
        </div>
        <p className="note">Test images per class: {CLASSES.map((c, i) => `${DISPLAY[c]} ${m.test_counts[i]}`).join(" · ")}. Glioma values rest on only {m.test_counts[0]} images.</p>
      </Card>

      <Card title="Dashboard 4 · Test confusion matrices of the final models">
        <div className="grid-3 tight">{FAM.map((f) => <Confusion key={f} matrix={finalV(f).confusion} title={f} height={300} />)}</div>
      </Card>

      <Card title="Dashboard 5 · ROC and precision–recall curves per class (test)" right={<Tabs value={curveModel} onChange={setCurveModel} tabs={FAM.map((f) => ({ value: f, label: f }))} />}>
        <div className="grid-2 tight">
          <Plot height={340} data={[...CLASSES.map((c) => ({ type: "scatter", mode: "lines", name: `${DISPLAY[c]} (AUC ${finalV(curveModel).curves.roc[c].auc.toFixed(3)})`, x: finalV(curveModel).curves.roc[c].x, y: finalV(curveModel).curves.roc[c].y, line: { color: CLASS_COLORS[c], width: 2.2 } })),
            { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: C.faint, dash: "dash" }, showlegend: false, hoverinfo: "skip" }]}
            layout={{ title: { text: "ROC", font: { size: 12 } }, xaxis: ax({ title: { text: "False positive rate" } }), yaxis: ax({ title: { text: "True positive rate" } }), margin: { t: 30, b: 90, l: 50, r: 10 } }} />
          <Plot height={340} data={CLASSES.map((c) => ({ type: "scatter", mode: "lines", name: `${DISPLAY[c]} (AP ${finalV(curveModel).curves.pr[c].ap.toFixed(3)})`, x: finalV(curveModel).curves.pr[c].x, y: finalV(curveModel).curves.pr[c].y, line: { color: CLASS_COLORS[c], width: 2.2, shape: "hv" } }))}
            layout={{ title: { text: "Precision–recall", font: { size: 12 } }, xaxis: ax({ title: { text: "Recall" } }), yaxis: ax({ title: { text: "Precision" }, range: [0, 1.05] }), margin: { t: 30, b: 90, l: 50, r: 10 } }} />
        </div>
      </Card>

      <Card title="Dashboard 6 · Clinical error rates for all variants" subtitle="Lower is better">
        <div className="grid-2 tight">
          {[["tumor_miss_rate", "Tumor miss rate — tumor scans predicted 'No Tumor'"], ["false_alarm_rate", "False alarm rate — healthy scans predicted as tumor"]].map(([k, t]) => (
            <Plot key={k} height={330} data={["val", "test"].map((s, i) => ({ type: "bar", orientation: "h", name: s === "val" ? "validation" : "test", y: V, x: V.map((v) => m.variants[v][s][k]), marker: { color: i ? C.amber : "#22d3ee" }, text: V.map((v) => pct(m.variants[v][s][k])), textposition: "outside" }))}
              layout={{ barmode: "group", title: { text: t, font: { size: 12 } }, yaxis: ax({ autorange: "reversed" }), xaxis: ax({ tickformat: ".0%" }), margin: { l: 230, t: 30, r: 40, b: 50 } }} />
          ))}
        </div>
      </Card>

      <Card title="Dashboard 7 · Calibration of the final models (test)" subtitle="On the diagonal = the confidence score means what it says">
        <div className="grid-2 tight">
          <Plot height={330} data={[...FAM.map((f) => ({ type: "scatter", mode: "lines+markers", name: `${f} (ECE ${finalV(f).test.ece.toFixed(3)})`, x: finalV(f).curves.calibration.map((b) => b.conf), y: finalV(f).curves.calibration.map((b) => b.acc), line: { color: MODEL_COLORS[f], width: 2 }, customdata: finalV(f).curves.calibration.map((b) => b.n), hovertemplate: "confidence %{x:.2f} → accuracy %{y:.2f} (n=%{customdata})<extra></extra>" })),
            { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: C.faint, dash: "dash" }, name: "perfect", hoverinfo: "skip" }]}
            layout={{ title: { text: "Reliability diagram", font: { size: 12 } }, xaxis: ax({ title: { text: "Mean confidence in bin" }, range: [0, 1] }), yaxis: ax({ title: { text: "Accuracy in bin" }, range: [0, 1.02] }), margin: { t: 30, b: 90, l: 50, r: 10 } }} />
          <Plot height={330} data={FAM.map((f) => ({ type: "bar", name: f, x: Array.from({ length: 20 }, (_, i) => (i + 0.5) / 20), y: finalV(f).curves.confidence_hist, marker: { color: MODEL_COLORS[f] }, opacity: 0.75 }))}
            layout={{ barmode: "overlay", title: { text: "Distribution of prediction confidence", font: { size: 12 } }, xaxis: ax({ title: { text: "Max class probability" } }), margin: { t: 30, b: 90, l: 50, r: 10 } }} />
        </div>
      </Card>

      <Card title="Dashboard 8 · Learning curves of the final models" subtitle="All training stages concatenated">
        <div className="grid-3 tight">
          {[["val_accuracy", "Validation accuracy"], ["val_loss", "Validation loss"], ["gap", "Train − validation accuracy (over-fitting gap)"]].map(([k, t]) => (
            <Plot key={k} height={300} data={FAM.map((f) => {
              const h = finalV(f).history; const cat = (key) => h.flatMap((s) => s[key]);
              const y = k === "gap" ? cat("accuracy").map((a, i) => a - cat("val_accuracy")[i]) : cat(k);
              return { type: "scatter", mode: "lines", name: f, y, x: y.map((_, i) => i + 1), line: { color: MODEL_COLORS[f], width: 2 } };
            })} layout={{ title: { text: t, font: { size: 12 } }, xaxis: ax({ title: { text: "Epoch" } }), margin: { t: 30, b: 80, l: 50, r: 10 } }} />
          ))}
        </div>
      </Card>

      <Card title="Per-class test recall with 95% bootstrap confidence intervals" subtitle="2,000 resamples of the test set">
        <Plot height={340} data={FAM.map((f) => {
          const rows = CLASSES.map((c) => m.bootstrap_ci.find((r) => r.model === f && r.metric === `recall_${c}`));
          return { type: "bar", name: f, x: CLASSES.map((c, i) => `${DISPLAY[c]} (n=${m.test_counts[i]})`), y: rows.map((r) => r.estimate), marker: { color: MODEL_COLORS[f] },
            error_y: { type: "data", symmetric: false, array: rows.map((r) => r.ci_high - r.estimate), arrayminus: rows.map((r) => r.estimate - r.ci_low), color: "#d1fae5", thickness: 1.4 } };
        })} layout={{ barmode: "group", yaxis: ax({ range: [0, 1.08], title: { text: "Recall" } }) }} />
      </Card>

      <Card title="Efficiency and safety comparison">
        <div className="grid-3 tight">
          {[["params_M", "Parameters (millions)"], ["size_MB", "Model file size (MB)"], ["latency_ms", "Single-image latency (ms, Colab)"], ["throughput_img_s", "Throughput (images/s)"], ["train_min", "Training time (min)"], ["test_tumor_miss_rate", "Tumor miss rate (test)"]].map(([k, t]) => (
            <Plot key={k} height={230} data={[{ type: "bar", x: FAM, y: FAM.map((f) => comp[f]?.[k]), marker: { color: FAM.map((f) => MODEL_COLORS[f]) }, text: FAM.map((f) => (k.includes("rate") ? pct(comp[f]?.[k]) : fmt(comp[f]?.[k], 1))), textposition: "outside" }]}
              layout={{ title: { text: t, font: { size: 12 } }, margin: { t: 30, b: 40, l: 44, r: 6 } }} />
          ))}
        </div>
      </Card>

      <Card title="Final model selection" subtitle="Validation decides; test only reports">
        <table className="tbl">
          <thead><tr><th>Model</th><th>Variant</th><th>Val macro-F1</th><th>Test accuracy</th><th>Test macro-F1</th><th>ROC-AUC</th><th>Tumor miss</th><th>False alarm</th><th>Size (MB)</th><th>Latency (ms)</th></tr></thead>
          <tbody>{FAM.map((f) => { const v = finalV(f); return (
            <tr key={f} className={f === best ? "ok-row" : ""}><td>{f === best ? "🏆 " : ""}{f}</td><td>{m.final[f]}</td><td>{fmt(v.val.f1_macro)}</td><td>{pct(v.test.accuracy)}</td><td>{fmt(v.test.f1_macro)}</td><td>{fmt(v.test.roc_auc_ovr)}</td><td>{pct(v.test.tumor_miss_rate)}</td><td>{pct(v.test.false_alarm_rate)}</td><td>{fmt(comp[f]?.size_MB, 1)}</td><td>{fmt(comp[f]?.latency_ms, 0)}</td></tr>); })}</tbody>
        </table>
      </Card>

      <Card title="Explainability · occlusion sensitivity of the final model" subtitle="Red = region the prediction depends on (from the notebook)">
        <img src="img/occlusion.png" alt="Occlusion sensitivity maps" className="wide-img" loading="lazy" />
        <p className="note">MobileNetV2 focuses on brain tissue rather than the black background or skull outline — no sign of the framing shortcut tested in H2. Pituitary predictions concentrate on the sellar region; the one misclassified glioma came with only 50% confidence, below the app's uncertainty threshold.</p>
      </Card>
      <div className="grid-2">
        <Card title="Reload sanity check" subtitle="Saved model reproduces predictions exactly"><img src="img/sanity.png" alt="" className="wide-img" loading="lazy" /></Card>
        <Card title="Upload checks" subtitle="Colour and non-MRI images are flagged, never silently diagnosed"><img src="img/input_checks.png" alt="" className="wide-img" loading="lazy" /></Card>
      </div>
    </Section>
  );
}
