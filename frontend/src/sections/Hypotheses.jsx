import { useEffect, useState } from "react";
import Plot, { ax } from "../lib/Plot";
import { loadData } from "../lib/data";
import { CLASSES, CLASS_COLORS, DISPLAY, sci } from "../lib/theme";
import { Card, Loading, Section } from "../lib/ui";

const TEXT = {
  H1: { title: "H1 · Brightness differs across tumor classes", h0: "Mean image brightness has the same distribution in all four classes.",
    decision: "Brightness carries class information, so brightness/contrast augmentation was limited to ±10% for all three models." },
  H2: { title: "H2 · Framing (foreground ratio) differs across classes", h0: "The share of the frame occupied by the head is the same in all four classes.",
    decision: "Framing is a potential shortcut, so random zoom (±10%) and shifts (±5%) were added for all three models." },
};

export default function Hypotheses() {
  const [h, setH] = useState(null);
  const [m, setM] = useState(null);
  useEffect(() => { loadData("hypotheses").then(setH); loadData("models").then(setM); }, []);
  if (!h || !m) return <Section id="hypotheses" title="Hypothesis testing"><Loading /></Section>;

  return (
    <Section id="hypotheses" kicker="Section 5 · α = 0.05" title="Hypothesis testing"
      intro="Kruskal–Wallis (normality and equal variance fail — see the Shapiro and Levene p-values) with ε² effect size and Bonferroni-corrected Mann–Whitney post-hoc tests; McNemar's test for paired model predictions.">
      <div className="grid-2">
        {["H1", "H2"].map((k) => {
          const r = h[k];
          return (
            <Card key={k} title={TEXT[k].title} subtitle={`H₀: ${TEXT[k].h0}`}>
              <div className="hyp-stats">
                <div><b>{r.H}</b><span>Kruskal–Wallis H</span></div>
                <div><b>{sci(r.p)}</b><span>p-value</span></div>
                <div><b>{r.eps2}</b><span>effect size ε²</span></div>
                <div className="reject"><b>Reject H₀</b><span>distributions differ</span></div>
              </div>
              <Plot height={220} data={[{ type: "bar", x: CLASSES.map((c) => DISPLAY[c]), y: CLASSES.map((c) => r.medians[c]), marker: { color: CLASSES.map((c) => CLASS_COLORS[c]) }, text: CLASSES.map((c) => r.medians[c].toFixed(k === "H1" ? 1 : 3)), textposition: "outside" }]}
                layout={{ yaxis: ax({ title: { text: `median ${r.variable}` } }), margin: { t: 10, b: 36, l: 56, r: 8 } }} />
              <table className="tbl compact">
                <thead><tr><th>Pair</th><th>Median A</th><th>Median B</th><th>p (Bonferroni)</th><th /></tr></thead>
                <tbody>{r.posthoc.map((p) => (
                  <tr key={p.a + p.b}><td>{DISPLAY[p.a]} vs {DISPLAY[p.b]}</td><td>{p.median_a}</td><td>{p.median_b}</td><td>{sci(p.p_bonferroni)}</td><td>{p.p_bonferroni < 0.05 ? "✓" : "–"}</td></tr>
                ))}</tbody>
              </table>
              <p className="note"><b>Modelling decision:</b> {TEXT[k].decision}</p>
            </Card>
          );
        })}
      </div>
      <Card title="H3 · Do the models make significantly different errors? (McNemar's test)" subtitle="H₀: two models have the same error rate on the same 158 test images">
        <table className="tbl">
          <thead><tr><th>Model A</th><th>Model B</th><th>A right / B wrong</th><th>A wrong / B right</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Decision</th></tr></thead>
          <tbody>{m.mcnemar.map((r) => (
            <tr key={r.model_a + r.model_b} className={r.p_value < 0.05 ? "ok-row" : ""}>
              <td>{r.model_a}</td><td>{r.model_b}</td><td>{r.a_right_b_wrong}</td><td>{r.a_wrong_b_right}</td><td>{r.test}</td><td>{r.statistic}</td><td>{r.p_value.toFixed(4)}</td><td>{r.decision}</td>
            </tr>
          ))}</tbody>
        </table>
        <p className="note">MobileNetV2 makes significantly fewer errors than the custom CNN (p ≈ 0.02). Its lead over EfficientNetB0 is likely but not statistically proven on a test set of this size.</p>
      </Card>
    </Section>
  );
}
