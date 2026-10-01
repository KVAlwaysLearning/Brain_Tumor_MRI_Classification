// Colour system: black + emerald. Class colours are distinct so charts stay readable.
export const C = {
  bg: "#020403", text: "#e6fff4", muted: "#8fb3a3", faint: "rgba(143,179,163,0.35)",
  emerald: "#10b981", emerald2: "#34d399", mint: "#6ee7b7", neg: "#fb7185", amber: "#fbbf24",
  grid: "rgba(52,211,153,0.10)", line: "rgba(52,211,153,0.28)",
};
export const CLASS_COLORS = { glioma: "#34d399", meningioma: "#22d3ee", no_tumor: "#a78bfa", pituitary: "#fbbf24" };
export const SPLIT_COLORS = { train: "#10b981", valid: "#22d3ee", test: "#fbbf24" };
export const MODEL_COLORS = { "Custom CNN": "#fbbf24", EfficientNetB0: "#22d3ee", MobileNetV2: "#34d399" };
export const DISPLAY = { glioma: "Glioma", meningioma: "Meningioma", no_tumor: "No Tumor", pituitary: "Pituitary" };
export const CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"];

export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
export const sci = (p) => (p < 1e-4 ? p.toExponential(2) : p.toFixed(4));
