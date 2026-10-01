import { useEffect, useMemo, useRef, useState } from "react";
import Plot, { ax } from "../lib/Plot";
import { loadData } from "../lib/data";
import { C, CLASSES, CLASS_COLORS, DISPLAY, SPLIT_COLORS } from "../lib/theme";
import { Card, Insight, Loading, Section } from "../lib/ui";
import GlareHover from "../reactbits/GlareHover";

const D = CLASSES.map((c) => DISPLAY[c]);
const quantile = (arr, q) => { const s = [...arr].sort((a, b) => a - b); const i = (s.length - 1) * q; const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };
const vline = (x, dash = "dash", color = "#ffffff") => ({ type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper", line: { color, dash, width: 1.4 } });

export default function EDA() {
  const [eda, setEda] = useState(null);
  useEffect(() => { loadData("eda").then(setEda); }, []);

  const byClass = useMemo(() => {
    if (!eda) return null;
    const out = {};
    for (const c of CLASSES) out[c] = {};
    const n = eda.images.label.length;
    for (const key of ["mean_intensity", "std_intensity", "foreground_ratio", "sharpness", "entropy", "file_size_kb"])
      for (const c of CLASSES) out[c][key] = [];
    for (let i = 0; i < n; i++) {
      const c = eda.images.label[i];
      for (const key of Object.keys(out[c])) out[c][key].push(eda.images[key][i]);
    }
    return out;
  }, [eda]);

  if (!eda) return <Section id="eda" kicker="Section 4" title="Exploratory data analysis"><Loading /></Section>;
  const I = eda.insights;
  const all = (key) => eda.images[key];

  return (
    <Section id="eda" kicker="Section 4 · UBM rule" title="Exploratory data analysis"
      intro={`All charts use the ${eda.n_clean.toLocaleString()} unique images left after de-duplication. Charts 1–7 are univariate, 8–13 bivariate, 14–15 multivariate. Hover any chart for exact values.`}>

      <div className="grid-2">
        <Card title="Chart 1 · Class distribution per split" subtitle="Before vs after near-duplicate removal">
          <div className="grid-2 tight">
            {["before", "after"].map((w) => (
              <Plot key={w} height={300} data={eda.splits.map((s) => ({
                type: "bar", name: s, x: D, y: CLASSES.map((c) => eda.counts[w][s][c]), marker: { color: SPLIT_COLORS[s] },
                text: CLASSES.map((c) => eda.counts[w][s][c]), textposition: "outside", textfont: { size: 9, color: C.muted },
              }))} layout={{ barmode: "group", title: { text: w === "before" ? `Before (${eda.n_raw})` : `After (${eda.n_clean})`, font: { size: 13 } }, margin: { t: 36, l: 44, r: 8, b: 60 }, yaxis: ax({ title: { text: "Images" } }) }} />
            ))}
          </div>
          <Insight data={I[1]} />
        </Card>

        <Card title="Chart 2 · Share of each class" subtitle="Cleaned dataset">
          <Plot height={330} data={[{
            type: "pie", hole: 0.55, labels: D, values: CLASSES.map((c) => eda.splits.reduce((a, s) => a + eda.counts.after[s][c], 0)),
            marker: { colors: CLASSES.map((c) => CLASS_COLORS[c]), line: { color: C.bg, width: 2 } }, textinfo: "label+percent", sort: false,
          }]} layout={{ showlegend: false, margin: { t: 10, b: 10, l: 10, r: 10 } }} />
          <Insight data={I[2]} />
        </Card>
      </div>

      <Card title="Chart 3 · Resolution, shape and channels">
        <div className="grid-3 tight">
          <Plot height={250} data={[{ type: "bar", x: Object.keys(eda.resolution_counts), y: Object.values(eda.resolution_counts), marker: { color: C.emerald }, text: Object.values(eda.resolution_counts), textposition: "outside" }]}
            layout={{ title: { text: "Resolution (W×H)", font: { size: 12 } }, margin: { t: 30, b: 40, l: 44, r: 8 } }} />
          <Plot height={250} data={[{ type: "histogram", x: all("aspect_ratio"), nbinsx: 20, marker: { color: "#22d3ee" } }]}
            layout={{ title: { text: "Aspect ratio (W / H)", font: { size: 12 } }, margin: { t: 30, b: 40, l: 44, r: 8 } }} />
          <Plot height={250} data={[{ type: "bar", x: Object.keys(eda.mode_counts), y: Object.values(eda.mode_counts), marker: { color: C.amber }, text: Object.values(eda.mode_counts), textposition: "outside" }]}
            layout={{ title: { text: "Stored mode vs actual content", font: { size: 12 } }, margin: { t: 30, b: 60, l: 44, r: 8 } }} />
        </div>
        <Insight data={I[3]} />
      </Card>

      <Card title="Chart 4 · JPEG file-size distribution">
        <div className="grid-2 tight">
          <Plot height={300} data={[{ type: "histogram", x: all("file_size_kb"), nbinsx: 40, marker: { color: "#a78bfa" } }]}
            layout={{ shapes: [vline(quantile(all("file_size_kb"), 0.5), "dash", C.neg)], xaxis: ax({ title: { text: "File size (KB)" } }), yaxis: ax({ title: { text: "Images" } }),
              annotations: [{ x: quantile(all("file_size_kb"), 0.5), y: 1, yref: "paper", text: `median ${quantile(all("file_size_kb"), 0.5).toFixed(0)} KB`, showarrow: false, font: { color: C.neg, size: 11 }, xanchor: "left" }] }} />
          <Plot height={300} data={CLASSES.map((c) => ({ type: "histogram", name: DISPLAY[c], x: byClass[c].file_size_kb, histnorm: "probability density", nbinsx: 35, opacity: 0.5, marker: { color: CLASS_COLORS[c] } }))}
            layout={{ barmode: "overlay", xaxis: ax({ title: { text: "File size (KB) by class" } }) }} />
        </div>
        <Insight data={I[4]} />
      </Card>

      <Card title="Chart 5 · Random MRI samples from each class">
        <div className="sample-grid">
          {CLASSES.map((c) => (
            <div key={c} className="sample-row">
              <div className="sample-label" style={{ color: CLASS_COLORS[c] }}>{DISPLAY[c]}</div>
              <div className="sample-imgs">
                {eda.samples[c].map((src) => (
                  <GlareHover key={src} width="100%" height="auto" background="#000" borderColor="rgba(52,211,153,0.2)" glareColor="#6ee7b7" glareOpacity={0.35} borderRadius="10px">
                    <img src={src} alt={DISPLAY[c]} loading="lazy" className="sample-img" />
                  </GlareHover>
                ))}
              </div>
            </div>
          ))}
        </div>
        <Insight data={I[5]} />
      </Card>

      <div className="grid-2">
        <Card title="Chart 6 · Mean image brightness" subtitle="Dashed = median, dotted = quartiles">
          <Plot height={300} data={[{ type: "histogram", x: all("mean_intensity"), nbinsx: 50, marker: { color: C.neg } }]}
            layout={{ shapes: [[0.25, "dot"], [0.5, "dash"], [0.75, "dot"]].map(([q, d]) => vline(quantile(all("mean_intensity"), q), d)), xaxis: ax({ title: { text: "Mean gray level (0–255)" } }) }} />
          <Insight data={I[6]} />
        </Card>
        <Card title="Chart 7 · Foreground ratio" subtitle="Share of the frame occupied by the head">
          <Plot height={300} data={[{ type: "histogram", x: all("foreground_ratio"), nbinsx: 50, marker: { color: "#d6a77a" } }]}
            layout={{ shapes: [vline(quantile(all("foreground_ratio"), 0.5))], xaxis: ax({ title: { text: "Foreground ratio" } }) }} />
          <Insight data={I[7]} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Chart 8 · Mean brightness by tumor class" subtitle="Numerical – categorical">
          <Plot height={340} data={CLASSES.map((c) => ({ type: "box", name: DISPLAY[c], y: byClass[c].mean_intensity, boxpoints: "all", jitter: 0.45, pointpos: 0, marker: { color: CLASS_COLORS[c], size: 2.5, opacity: 0.45 }, line: { color: CLASS_COLORS[c] }, fillcolor: "rgba(0,0,0,0)" }))}
            layout={{ showlegend: false, yaxis: ax({ title: { text: "Mean gray level" } }) }} />
          <Insight data={I[8]} />
        </Card>
        <Card title="Chart 9 · Contrast by tumor class" subtitle="Std of gray levels">
          <Plot height={340} data={CLASSES.map((c) => ({ type: "violin", name: DISPLAY[c], y: byClass[c].std_intensity, box: { visible: true }, meanline: { visible: true }, line: { color: CLASS_COLORS[c] }, opacity: 0.8, points: false }))}
            layout={{ showlegend: false, yaxis: ax({ title: { text: "Std of gray level" } }) }} />
          <Insight data={I[9]} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Chart 10 · Brain-tissue pixel intensities" subtitle="Background removed (gray > 15)">
          <Plot height={320} data={CLASSES.map((c) => ({ type: "scatter", mode: "lines", name: DISPLAY[c], x: eda.tissue_kde.x, y: eda.tissue_kde.y[c], line: { color: CLASS_COLORS[c], width: 2.5 } }))}
            layout={{ xaxis: ax({ title: { text: "Gray level" } }), yaxis: ax({ title: { text: "Density" } }) }} />
          <Insight data={I[10]} />
        </Card>
        <Card title="Chart 11 · Brightness vs foreground ratio" subtitle={`Numerical – numerical · Spearman ρ = ${eda.spearman_fg_brightness}`}>
          <Plot height={320} data={CLASSES.map((c) => ({ type: "scattergl", mode: "markers", name: DISPLAY[c], x: byClass[c].foreground_ratio, y: byClass[c].mean_intensity, marker: { color: CLASS_COLORS[c], size: 5, opacity: 0.55 } }))}
            layout={{ xaxis: ax({ title: { text: "Foreground ratio" } }), yaxis: ax({ title: { text: "Mean gray level" } }) }} />
          <Insight data={I[11]} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Chart 12 · Class composition of each split" subtitle={`Categorical – categorical · χ² = ${eda.chi2_class_split.chi2}, p = ${eda.chi2_class_split.p.toExponential(2)}`}>
          {(() => {
            const tot = Object.fromEntries(eda.splits.map((s) => [s, CLASSES.reduce((a, c) => a + eda.counts.after[s][c], 0)]));
            const z = CLASSES.map((c) => eda.splits.map((s) => (100 * eda.counts.after[s][c]) / tot[s]));
            return <Plot height={320} data={[{ type: "heatmap", x: eda.splits, y: D, z, colorscale: [[0, "#03140d"], [1, "#10b981"]],
              text: CLASSES.map((c, i) => eda.splits.map((s, j) => `${eda.counts.after[s][c]}<br>(${z[i][j].toFixed(1)}%)`)), texttemplate: "%{text}", hovertemplate: "%{y} in %{x}: %{text}<extra></extra>", colorbar: { title: { text: "% of split" } } }]}
              layout={{ yaxis: ax({ autorange: "reversed" }) }} />;
          })()}
          <Insight data={I[12]} />
        </Card>
        <Card title="Chart 14 · Spearman correlation between image statistics" subtitle="Multivariate">
          <Plot height={360} data={[{ type: "heatmap", x: eda.correlation.features, y: eda.correlation.features, z: eda.correlation.matrix, zmin: -1, zmax: 1,
            colorscale: [[0, "#fb7185"], [0.5, "#050806"], [1, "#10b981"]], text: eda.correlation.matrix, texttemplate: "%{text:.2f}", textfont: { size: 10 } }]}
            layout={{ yaxis: ax({ autorange: "reversed" }), margin: { l: 110, b: 110, t: 10, r: 10 } }} />
          <Insight data={I[14]} />
        </Card>
      </div>

      <Card title="Chart 13 · Average MRI per class" subtitle="Top: mean image · bottom: difference from the overall mean (green brighter, red darker)">
        <div className="mean-grid">
          {CLASSES.map((c) => (
            <div key={c} className="mean-col">
              <div className="mean-title" style={{ color: CLASS_COLORS[c] }}>{DISPLAY[c]}</div>
              <MatrixCanvas matrix={eda.mean_images[c].mean} mode="gray" />
              <MatrixCanvas matrix={eda.mean_images[c].diff} mode="diverging" />
            </div>
          ))}
        </div>
        <Insight data={I[13]} />
      </Card>

      <Card title="Chart 15 · Pair plot of image statistics by class" subtitle="Multivariate · drag to zoom a panel, double-click to reset">
        <Plot height={620} data={CLASSES.map((c) => ({
          type: "splom", name: DISPLAY[c], showupperhalf: false, diagonal: { visible: false },
          dimensions: [["mean_intensity", "brightness"], ["std_intensity", "contrast"], ["foreground_ratio", "foreground"], ["entropy", "entropy"]]
            .map(([k, label]) => ({ label, values: byClass[c][k] })),
          marker: { color: CLASS_COLORS[c], size: 3.5, opacity: 0.5, line: { width: 0 } },
        }))} layout={{ dragmode: "zoom", margin: { l: 60, r: 10, t: 10, b: 60 },
          ...Object.fromEntries([1, 2, 3, 4].flatMap((i) => [[`xaxis${i === 1 ? "" : i}`, ax()], [`yaxis${i === 1 ? "" : i}`, ax()]])) }} />
        <Insight data={I[15]} />
      </Card>
    </Section>
  );
}

// Canvas rendering of a small matrix (mean image / difference map) with hover value readout
export function MatrixCanvas({ matrix, mode = "gray", size = 180 }) {
  const ref = useRef(null);
  const [tip, setTip] = useState(null);
  const n = matrix.length;
  const lim = useMemo(() => Math.max(...matrix.flat().map((v) => Math.abs(v))) || 1, [matrix]);
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const img = ctx.createImageData(n, n);
    matrix.forEach((row, y) => row.forEach((v, x) => {
      const i = (y * n + x) * 4;
      if (mode === "gray") { img.data[i] = img.data[i + 1] = img.data[i + 2] = v; }
      else {
        const t = Math.max(-1, Math.min(1, v / lim));
        img.data[i] = t < 0 ? 251 * -t : 8; img.data[i + 1] = t > 0 ? 185 * t + 20 : 20 + 90 * -t; img.data[i + 2] = t > 0 ? 129 * t : 133 * -t;
      }
      img.data[i + 3] = 255;
    }));
    ctx.putImageData(img, 0, 0);
  }, [matrix, mode, lim, n]);
  return (
    <div className="matrix-wrap" style={{ width: size }}>
      <canvas ref={ref} width={n} height={n} style={{ width: size, height: size }} className="pixelated"
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const x = Math.floor(((e.clientX - r.left) / r.width) * n); const y = Math.floor(((e.clientY - r.top) / r.height) * n); setTip(matrix[y]?.[x] !== undefined ? `(${x}, ${y}) = ${matrix[y][x]}` : null); }}
        onMouseLeave={() => setTip(null)} />
      <div className="matrix-tip">{tip || " "}</div>
    </div>
  );
}
