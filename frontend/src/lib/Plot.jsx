import { useEffect, useRef, useState } from "react";
import { C } from "./theme";

// Plotly is large, so it is loaded once, on demand, and each chart renders only when scrolled into view.
let plotlyPromise = null;
const getPlotly = () => (plotlyPromise ??= import("plotly.js-dist-min").then((m) => m.default || m));

const axis = { gridcolor: C.grid, zerolinecolor: C.line, linecolor: C.line, tickfont: { color: C.muted, size: 11 },
  title: { font: { color: C.muted, size: 12 } }, automargin: true };

export const baseLayout = (extra = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, system-ui, sans-serif", color: C.text, size: 12 },
  margin: { l: 56, r: 18, t: 16, b: 48 },
  legend: { orientation: "h", y: -0.2, font: { color: C.muted, size: 11 } },
  hoverlabel: { bgcolor: "#06130d", bordercolor: C.emerald, font: { color: C.text } },
  xaxis: { ...axis }, yaxis: { ...axis }, bargap: 0.18,
  ...extra,
});
export const ax = (extra = {}) => ({ ...axis, ...extra });

export default function Plot({ data, layout, height = 360, config }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setVisible(true), { rootMargin: "300px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!visible) return;
    let alive = true;
    getPlotly().then((Plotly) => {
      if (!alive || !ref.current) return;
      Plotly.react(ref.current, data, baseLayout({ height, ...layout }),
        { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], ...config });
    });
    return () => { alive = false; };
  }, [visible, data, layout, height, config]);

  useEffect(() => () => { if (ref.current && plotlyPromise) plotlyPromise.then((P) => ref.current && P.purge(ref.current)); }, []);

  return <div ref={ref} className="plot" style={{ minHeight: height }}>{!visible && <div className="plot-skeleton" />}</div>;
}
