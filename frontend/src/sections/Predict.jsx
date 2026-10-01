import { useEffect, useRef, useState } from "react";
import { loadData } from "../lib/data";
import { onServerStatus, predict, urlToFile, wakeServer } from "../lib/api";
import { CLASSES, CLASS_COLORS, DISPLAY, pct } from "../lib/theme";
import { Card, Section } from "../lib/ui";
import StarBorder from "../reactbits/StarBorder";
import GradientText from "../reactbits/GradientText";

export function ServerStatus() {
  const [s, setS] = useState("idle");
  useEffect(() => onServerStatus(setS), []);
  const text = { idle: "Model server", waking: "Waking the model server (free plan · up to ~60 s)…", ready: "Model server ready", offline: "Model server unreachable — retry in a moment" }[s];
  return <div className={`server-pill ${s}`}><span className="dot" />{text}</div>;
}

export function Dropzone({ onFile, preview, label = "Drop a brain MRI here, or click to choose" }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  const pick = (f) => f && f.type.startsWith("image/") && onFile(f);
  return (
    <StarBorder as="div" color="#34d399" speed="6s" className={`dropzone-wrap ${drag ? "drag" : ""}`}>
      <div className="dropzone" onClick={() => input.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}>
        {preview ? <img src={preview} alt="Uploaded MRI" className="drop-preview" /> : (
          <div className="drop-empty"><div className="drop-icon">⬆</div><div>{label}</div><div className="drop-hint">JPG or PNG · up to 10 MB</div></div>
        )}
        <input ref={input} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files[0])} />
      </div>
    </StarBorder>
  );
}

export function SampleStrip({ onPick, active }) {
  const [samples, setSamples] = useState([]);
  useEffect(() => { loadData("eda").then((e) => setSamples(e.explainer_samples)); }, []);
  return (
    <div className="sample-strip">
      <span className="muted">Or try a test image:</span>
      {samples.map((s) => (
        <button key={s.src} className={`sample-btn ${active === s.src ? "active" : ""}`} onClick={() => onPick(s)} title={DISPLAY[s.label]}>
          <img src={s.src} alt={DISPLAY[s.label]} /><span style={{ color: CLASS_COLORS[s.label] }}>{DISPLAY[s.label]}</span>
        </button>
      ))}
    </div>
  );
}

export default function Predict({ image, setImage }) {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  useEffect(() => { wakeServer(); }, []);

  const run = async (file, url, source) => {
    setImage({ file, url, source }); setRes(null); setErr(null); setBusy(true);
    try { setRes(await predict(file)); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  const onFile = (f) => run(f, URL.createObjectURL(f), "upload");
  const onSample = async (s) => run(await urlToFile(s.src), s.src, s.src);

  return (
    <Section id="predict" kicker="Live prediction" title="Upload an MRI"
      intro="Runs the selected best model — MobileNetV2 (fine-tuned), 90.5% test accuracy — with exactly the notebook's preprocessing.">
      <ServerStatus />
      <div className="predict-grid">
        <div>
          <Dropzone onFile={onFile} preview={image?.url} />
          <SampleStrip onPick={onSample} active={image?.source} />
        </div>
        <Card title="Prediction" className="result-card">
          {busy && <div className="loading"><span className="spinner" />Analysing…</div>}
          {err && <div className="error">⚠ {err}</div>}
          {!busy && !res && !err && <p className="muted">Upload an image or pick a test image to see the model's prediction.</p>}
          {res && (
            <>
              <div className="pred-main">
                <GradientText colors={["#34d399", "#a7f3d0", "#10b981"]} animationSpeed={6} className="pred-label">{DISPLAY[res.predicted]}</GradientText>
                <div className="pred-conf">{pct(res.confidence)} confidence</div>
              </div>
              <div className="prob-bars">
                {CLASSES.map((c) => (
                  <div key={c} className={`prob-row ${c === res.predicted ? "top" : ""}`}>
                    <span className="prob-name">{DISPLAY[c]}</span>
                    <div className="prob-track"><div className="prob-fill" style={{ width: pct(res.probabilities[c]), background: CLASS_COLORS[c] }} /></div>
                    <span className="prob-val">{pct(res.probabilities[c])}</span>
                  </div>
                ))}
              </div>
              {res.warnings.length > 0 && <ul className="warn-list">{res.warnings.map((w) => <li key={w}>⚠ {w}</li>)}</ul>}
              <div className="pred-meta">{res.model} · {res.inference_ms} ms · input {res.original_size.join("×")} {res.original_mode}</div>
              <StarBorder as="button" color="#22d3ee" speed="5s" className="cta small" onClick={() => document.getElementById("explainer")?.scrollIntoView({ behavior: "smooth" })}>
                See inside the network →
              </StarBorder>
            </>
          )}
          <p className="disclaimer">Educational project — not a medical device. Always consult a radiologist.</p>
        </Card>
      </div>
    </Section>
  );
}
