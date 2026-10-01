import { useEffect, useState } from "react";
import Plot, { ax } from "../lib/Plot";
import { loadData } from "../lib/data";
import { C, CLASSES, CLASS_COLORS, DISPLAY, SPLIT_COLORS } from "../lib/theme";
import { Card, Loading, Section, Stat } from "../lib/ui";
import CountUp from "../reactbits/CountUp";

export default function Dataset() {
  const [d, setD] = useState(null);
  const [eda, setEda] = useState(null);
  useEffect(() => { loadData("dedup").then(setD); loadData("eda").then(setEda); }, []);
  if (!d || !eda) return <Section id="data" title="Dataset & leakage-free cleaning"><Loading /></Section>;
  const cross = Object.entries(d.removal_reasons).filter(([k]) => k.startsWith("cross")).reduce((a, [, v]) => a + v, 0);
  const splits = ["train", "valid", "test"];

  return (
    <Section id="data" kicker="Sections 1–3" title="Dataset & leakage-free cleaning"
      intro="A Roboflow export of brain MRI scans in four classes. Every image received a 256-bit perceptual hash under all 8 flips/rotations; pairs at ≥ 95% similarity were treated as the same scan. One copy per duplicate group was kept — train first, then validation, then test — and inside a split, the sharpest copy.">
      <div className="stats-row">
        <Stat label="raw images"><CountUp to={eda.n_raw} separator="," /></Stat>
        <Stat label="near-duplicate pairs (≥95%)"><CountUp to={d.n_pairs} /></Stat>
        <Stat label="images removed"><CountUp to={d.n_removed} /></Stat>
        <Stat label="cross-split leaks removed"><CountUp to={cross} /></Stat>
        <Stat label="unique images kept"><CountUp to={eda.n_clean} separator="," /></Stat>
        <Stat label="genuinely colour images" sub="all train / no_tumor">{eda.colour_images}</Stat>
      </div>

      <div className="grid-2">
        <Card title="Near-duplicate pairs by split combination" subtitle="Bar = number of pairs · hover for mean / min similarity">
          <Plot height={320} data={[{ type: "bar", x: d.pair_summary.map((p) => p.pair_type), y: d.pair_summary.map((p) => p.pairs),
            marker: { color: d.pair_summary.map((p) => (p.pair_type.includes("↔") ? C.neg : C.emerald)) }, text: d.pair_summary.map((p) => p.pairs), textposition: "outside",
            customdata: d.pair_summary.map((p) => [p.mean, p.min]), hovertemplate: "%{x}<br>%{y} pairs<br>mean %{customdata[0]:.2f}% · min %{customdata[1]:.2f}%<extra></extra>" }]}
            layout={{ yaxis: ax({ title: { text: "Pairs" } }), margin: { b: 80, t: 10, l: 50, r: 10 } }} />
          <p className="note">Red bars cross split boundaries — each one would have let the model be "tested" on a scan it trained on.</p>
        </Card>
        <Card title="Images removed, by split and class" subtitle="Duplication is concentrated in glioma and no-tumor">
          <Plot height={320} data={CLASSES.map((c) => ({ type: "bar", name: DISPLAY[c], x: splits, y: splits.map((s) => d.removed_by_split_class[s][c]), marker: { color: CLASS_COLORS[c] } }))}
            layout={{ barmode: "stack", yaxis: ax({ title: { text: "Images removed" } }) }} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Each image's highest similarity to any other image" subtitle="Red line = 95% threshold">
          <Plot height={300} data={[{ type: "bar", x: d.best_match_edges.slice(0, -1).map((e) => e + 0.5), y: d.best_match_hist, marker: { color: "#22d3ee" }, hovertemplate: "%{x:.0f}%: %{y} images<extra></extra>" }]}
            layout={{ shapes: [{ type: "line", x0: 95, x1: 95, y0: 0, y1: 1, yref: "paper", line: { color: C.neg, dash: "dash" } }], xaxis: ax({ title: { text: "Similarity to nearest neighbour (%)" } }), yaxis: ax({ title: { text: "Images" } }), bargap: 0.05 }} />
        </Card>
        <Card title="Duplicate groups by the splits they span">
          <table className="tbl">
            <thead><tr><th>Splits</th><th>Groups</th><th>Images</th><th>Mean sim.</th><th>Min sim.</th></tr></thead>
            <tbody>{d.groups.map((g) => (
              <tr key={g.splits} className={g.splits.includes("+") ? "warn-row" : ""}><td>{g.splits}</td><td>{g.groups}</td><td>{g.images ?? "—"}</td><td>{g.mean_similarity}%</td><td>{g.min_similarity}%</td></tr>
            ))}</tbody>
          </table>
          <p className="note">Groups spanning several splits keep only the train copy (or the validation copy if the group has no train image).</p>
        </Card>
      </div>

      <Card title="Examples of cross-split near-duplicates">
        <div className="dup-grid">
          {d.examples.map((e, i) => (
            <div key={i} className="dup-pair">
              <figure><img src={e.a} alt="" loading="lazy" /><figcaption style={{ color: SPLIT_COLORS[e.split_a] }}>{e.split_a}</figcaption></figure>
              <div className="dup-sim">{e.similarity}%</div>
              <figure><img src={e.b} alt="" loading="lazy" /><figcaption style={{ color: SPLIT_COLORS[e.split_b] }}>{e.split_b}</figcaption></figure>
              <div className="dup-label">{DISPLAY[e.label]}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Pre-processing and augmentation" subtitle="Identical in training and in this app (verified pixel-identical)">
        <div className="pipeline">
          {["Decode JPEG", "Grayscale (PIL formula)", "Replicate to 3 channels", "Resize 224×224 (antialiased)", "Scale ÷ 255 → [0, 1]", "Model-specific range inside the model"].map((s, i) => (
            <div key={s} className="pipe-step"><span>{i + 1}</span>{s}</div>
          ))}
        </div>
        <img src="img/augmentation.png" alt="Training-time augmentation" className="wide-img" loading="lazy" />
        <p className="note">Training-time augmentation: flips, ±29° rotation, ±10% zoom, ±5% shifts, ±10% brightness and contrast — mild, because H1 showed brightness carries class information.</p>
      </Card>
    </Section>
  );
}
