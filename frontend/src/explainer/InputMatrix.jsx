import { useEffect, useMemo, useRef, useState } from "react";

// What each model's backbone actually receives, computed from the scaled [0, 1] input
export const RANGE = {
  custom_cnn: { label: "Custom CNN input (scaled 0–1)", f: (v) => v },
  efficientnetb0: { label: "EfficientNetB0 backbone input (×255 → 0–255)", f: (v) => v * 255 },
  mobilenetv2: { label: "MobileNetV2 backbone input (×2 − 1 → −1…1)", f: (v) => v * 2 - 1 },
};
const LENS = 11;

export default function InputMatrix({ gray, model }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 112, y: 112 });
  const [mode, setMode] = useState("scaled");
  const px = useMemo(() => (gray ? Uint8Array.from(atob(gray), (c) => c.charCodeAt(0)) : null), [gray]);

  useEffect(() => {
    if (!px) return;
    const ctx = ref.current.getContext("2d");
    const img = ctx.createImageData(224, 224);
    for (let i = 0; i < px.length; i++) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = px[i]; img.data[i * 4 + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }, [px]);

  if (!px) return <div className="matrix-empty">The 224 × 224 input matrix appears here once an image is analysed.</div>;
  const val = (x, y) => { const v = px[y * 224 + x] / 255; return mode === "scaled" ? v : RANGE[model].f(v); };
  const half = Math.floor(LENS / 2);
  const x0 = Math.min(Math.max(pos.x - half, 0), 224 - LENS), y0 = Math.min(Math.max(pos.y - half, 0), 224 - LENS);
  const digits = mode === "scaled" || model === "mobilenetv2" ? 3 : 1;

  return (
    <div className="input-matrix">
      <div className="im-left">
        <div className="im-canvas-wrap">
          <canvas ref={ref} width={224} height={224} className="im-canvas pixelated"
            onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setPos({ x: Math.floor(((e.clientX - r.left) / r.width) * 224), y: Math.floor(((e.clientY - r.top) / r.height) * 224) }); }} />
          <div className="im-lens-box" style={{ left: `${(x0 / 224) * 100}%`, top: `${(y0 / 224) * 100}%`, width: `${(LENS / 224) * 100}%`, height: `${(LENS / 224) * 100}%` }} />
        </div>
        <div className="im-caption">Shape 224 × 224 × 3 · the three channels are identical (grayscale) · hover to move the lens</div>
      </div>
      <div className="im-right">
        <div className="tabs small">
          <button className={`tab ${mode === "scaled" ? "active" : ""}`} onClick={() => setMode("scaled")}>Scaled input 0–1</button>
          <button className={`tab ${mode === "backbone" ? "active" : ""}`} onClick={() => setMode("backbone")}>What the backbone receives</button>
        </div>
        <div className="im-title">{mode === "scaled" ? "Pixel ÷ 255 (same for all models)" : RANGE[model].label} · rows {y0}–{y0 + LENS - 1}, cols {x0}–{x0 + LENS - 1}</div>
        <div className="im-grid" style={{ gridTemplateColumns: `repeat(${LENS}, 1fr)` }}>
          {Array.from({ length: LENS * LENS }, (_, k) => {
            const x = x0 + (k % LENS), y = y0 + Math.floor(k / LENS); const v = px[y * 224 + x];
            return <div key={k} className={`im-cell ${x === pos.x && y === pos.y ? "cur" : ""}`} style={{ background: `rgba(16,185,129,${(v / 255) * 0.85})`, color: v > 150 ? "#02140c" : "#d1fae5" }}>{val(x, y).toFixed(digits)}</div>;
          })}
        </div>
        <div className="im-caption">Cursor ({pos.x}, {pos.y}): raw pixel {px[pos.y * 224 + pos.x]} → scaled {(px[pos.y * 224 + pos.x] / 255).toFixed(4)}{model !== "custom_cnn" && <> → backbone {RANGE[model].f(px[pos.y * 224 + pos.x] / 255).toFixed(3)}</>}</div>
      </div>
    </div>
  );
}
