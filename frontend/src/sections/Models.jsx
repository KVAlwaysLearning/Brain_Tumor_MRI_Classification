import { useEffect, useState } from "react";
import Plot, { ax } from "../lib/Plot";
import { loadData } from "../lib/data";
import { C, fmt } from "../lib/theme";
import { Card, Loading, Section, Tabs } from "../lib/ui";
import { ALL_METRICS, Confusion, HistoryChart, LABEL, LOWER_BETTER, MetricBars } from "./charts";

const ABOUT = {
  "Custom CNN": "Built from scratch: four blocks of two 3×3 convolutions with batch normalization, ReLU and max-pooling (32→64→128→256 filters, stride-2 first convolution), spatial dropout in the deeper blocks, global average pooling, a 256-unit dense layer and a 4-way softmax. Tuned with a learning-rate × dropout grid search and checked with 3-fold cross-validation.",
  EfficientNetB0: "ImageNet-pretrained EfficientNetB0 with a new head (GAP → dropout → dense 256 → dropout → softmax). Stage 1 trains only the head with the backbone frozen; stage 2 fine-tunes the top of the backbone (grid over unfreeze depth × learning rate, BatchNorm kept frozen).",
  MobileNetV2: "ImageNet-pretrained MobileNetV2 (depthwise-separable convolutions, inverted residuals) with the same head and the same two-stage strategy as EfficientNetB0. Lightweight and fast — selected as the final deployed model.",
};

export default function Models() {
  const [m, setM] = useState(null);
  const [fam, setFam] = useState("Custom CNN");
  const [split, setSplit] = useState("test");
  useEffect(() => { loadData("models").then(setM); }, []);
  if (!m) return <Section id="models" title="Model results"><Loading /></Section>;

  const variants = Object.keys(m.variants).filter((v) => m.variants[v].family === fam);
  const [before, after] = variants;

  return (
    <Section id="models" kicker="Section 7 · ML model implementation" title="Model results"
      intro="Every tuning decision used the validation set; the test set (158 unique images) was used only for reporting. Pick a model family to see its training, errors and tuning.">
      <Tabs value={fam} onChange={setFam} tabs={["Custom CNN", "EfficientNetB0", "MobileNetV2"].map((f) => ({ value: f, label: f }))} />
      <p className="about">{ABOUT[fam]}</p>

      {variants.map((v) => (
        <Card key={v} title={`Training history — ${v}`} subtitle={`${m.variants[v].params.toLocaleString()} parameters`}>
          <HistoryChart variant={v} v={m.variants[v]} />
        </Card>
      ))}

      <div className="grid-2">
        {variants.map((v) => <Card key={v} title="Test confusion matrix" subtitle={v}><Confusion matrix={m.variants[v].confusion} title="Row-normalised (count in brackets)" /></Card>)}
      </div>

      <Card title="Evaluation metric score chart" subtitle={`${before} vs ${after}`} right={<Tabs value={split} onChange={setSplit} tabs={[{ value: "val", label: "Validation" }, { value: "test", label: "Test" }]} />}>
        <MetricBars variants={variants} models={m} split={split} />
      </Card>

      <div className="grid-2">
        <Card title="Improvement after tuning" subtitle="Green = better (lower is better for the last four)">
          <table className="tbl compact">
            <thead><tr><th>Metric</th><th>Val before</th><th>Val after</th><th>Test before</th><th>Test after</th><th>Δ test</th></tr></thead>
            <tbody>{ALL_METRICS.map((k) => {
              const d = m.variants[after].test[k] - m.variants[before].test[k];
              const good = LOWER_BETTER.has(k) ? d < 0 : d > 0;
              return <tr key={k}><td>{LABEL[k]}</td><td>{fmt(m.variants[before].val[k])}</td><td>{fmt(m.variants[after].val[k])}</td><td>{fmt(m.variants[before].test[k])}</td><td>{fmt(m.variants[after].test[k])}</td>
                <td style={{ color: Math.abs(d) < 1e-9 ? C.muted : good ? C.emerald2 : C.neg }}>{d > 0 ? "+" : ""}{fmt(d)}</td></tr>;
            })}</tbody>
          </table>
        </Card>
        {fam === "Custom CNN" ? <CustomTuning m={m} /> : <FineTuneGrid rows={m.finetune_grids[fam]} />}
      </div>
    </Section>
  );
}

function CustomTuning({ m }) {
  const g = m.custom_grid;
  const lrs = [...new Set(g.map((r) => r.lr))].sort(); const dos = [...new Set(g.map((r) => r.dropout))].sort();
  const z = lrs.map((lr) => dos.map((d) => g.find((r) => r.lr === lr && r.dropout === d)?.val_f1_macro));
  const cv = m.custom_cv;
  return (
    <Card title="Hyperparameter tuning & cross-validation" subtitle="Grid search (validation macro-F1) · 3-fold stratified CV of the best configuration">
      <div className="grid-2 tight">
        <Plot height={260} data={[{ type: "heatmap", x: dos.map((d) => `dropout ${d}`), y: lrs.map((l) => `lr ${l}`), z, colorscale: [[0, "#052e21"], [1, "#34d399"]], text: z, texttemplate: "%{text:.3f}", showscale: false }]}
          layout={{ margin: { t: 10, l: 80, b: 40, r: 10 } }} />
        <Plot height={260} data={["accuracy", "f1_macro", "recall_macro"].map((k, i) => ({ type: "bar", name: LABEL[k], x: cv.map((f) => `fold ${f.fold}`), y: cv.map((f) => f[k]), marker: { color: ["#34d399", "#22d3ee", "#fbbf24"][i] } }))}
          layout={{ barmode: "group", yaxis: ax({ range: [0.6, 1] }), margin: { t: 10, l: 40, b: 60, r: 10 } }} />
      </div>
      <p className="note">CV macro-F1 {cv.map((f) => f.f1_macro.toFixed(3)).join(" / ")} — mean {(cv.reduce((a, f) => a + f.f1_macro, 0) / cv.length).toFixed(3)}. Stable across folds.</p>
    </Card>
  );
}

function FineTuneGrid({ rows }) {
  return (
    <Card title="Fine-tuning grid search" subtitle="Each candidate starts from the frozen-stage weights · BatchNorm kept frozen">
      <table className="tbl compact">
        <thead><tr><th>Unfreeze</th><th>LR</th><th>Trainable backbone layers</th><th>Val accuracy</th><th>Val macro-F1</th><th>Best val loss</th><th>Epochs</th><th>Minutes</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className={i === rows.reduce((b, x, j) => (x.val_f1_macro > rows[b].val_f1_macro ? j : b), 0) ? "ok-row" : ""}>
            <td>top {Math.round(r.unfreeze_fraction * 100)}%</td><td>{r.lr}</td><td>{r.trainable_backbone_layers}</td><td>{fmt(r.val_accuracy)}</td><td>{fmt(r.val_f1_macro)}</td><td>{fmt(r.best_val_loss)}</td><td>{r.epochs}</td><td>{fmt(r.minutes, 1)}</td>
          </tr>))}</tbody>
      </table>
      <p className="note">Highlighted row = selected on validation macro-F1.</p>
    </Card>
  );
}
