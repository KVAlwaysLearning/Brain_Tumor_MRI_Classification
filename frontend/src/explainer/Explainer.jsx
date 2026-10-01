import { useEffect, useRef, useState } from "react";
import { loadData } from "../lib/data";
import { explain, urlToFile } from "../lib/api";
import { CLASSES, CLASS_COLORS, DISPLAY, pct } from "../lib/theme";
import { Card, Loading, Section } from "../lib/ui";
import StarBorder from "../reactbits/StarBorder";
import { Dropzone, SampleStrip, ServerStatus } from "../sections/Predict";
import InputMatrix from "./InputMatrix";
import NetworkView from "./NetworkView";

const MODELS = [
  { slug: "custom_cnn", name: "Custom CNN", sub: "from scratch · 36 layers" },
  { slug: "efficientnetb0", name: "EfficientNetB0", sub: "transfer learning · 243 layers" },
  { slug: "mobilenetv2", name: "MobileNetV2", sub: "transfer learning · 159 layers · deployed" },
];

export default function Explainer({ shared }) {
  const [model, setModel] = useState("mobilenetv2");
  const [k, setK] = useState(8);
  const [graph, setGraph] = useState(null);
  const [image, setImage] = useState(null);           // { file, url, source }
  const [net, setNet] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const req = useRef(0);

  // default to the image last used in the Predict section
  useEffect(() => { if (shared) setImage(shared); }, [shared]);
  useEffect(() => { setGraph(null); loadData(`graph_${model}`).then(setGraph); }, [model]);

  useEffect(() => {
    if (!image) return;
    const id = ++req.current;
    setBusy(true); setErr(null);
    explain(image.file, model, k)
      .then((r) => id === req.current && setNet(r))
      .catch((e) => id === req.current && setErr(e.message))
      .finally(() => id === req.current && setBusy(false));
  }, [image, model, k]);

  const netForModel = net && net.model === model && net.k === k ? net : null;

  return (
    <Section id="explainer" kicker="Model explainer" title="Inside the network, layer by layer"
      intro="Pick a model. The schematic lists every layer with its output shape and exact parameter count (click a block to expand all details). The network below runs your image through the real model: each column shows a layer's most-activated nodes out of its total, with real activation values, and the lines carry the real trained weights.">
      <div className="model-buttons">
        {MODELS.map((m) => (
          model === m.slug
            ? <StarBorder key={m.slug} as="button" color="#34d399" speed="4s" className="model-btn active" onClick={() => setModel(m.slug)}><b>{m.name}</b><span>{m.sub}</span></StarBorder>
            : <button key={m.slug} className="model-btn" onClick={() => setModel(m.slug)}><b>{m.name}</b><span>{m.sub}</span></button>
        ))}
      </div>

      <div className="explainer-top">
        <Card title="Image">
          <Dropzone onFile={(f) => setImage({ file: f, url: URL.createObjectURL(f), source: "upload" })} preview={image?.url} label="Drop an MRI to explain" />
          <SampleStrip active={image?.source} onPick={async (s) => setImage({ file: await urlToFile(s.src), url: s.src, source: s.src })} />
          <div className="k-select">
            <span className="muted">Nodes shown per layer</span>
            {[4, 8, 12, 16].map((n) => <button key={n} className={`chip ${k === n ? "active" : ""}`} onClick={() => setK(n)}>{n}</button>)}
          </div>
          <ServerStatus />
        </Card>
        <Card title="Transformed, scaled input matrix" subtitle="Exactly what enters the model">
          <InputMatrix gray={netForModel?.input.gray} model={model} />
        </Card>
      </div>

      {netForModel && (
        <div className="explain-result">
          <span className="muted">{MODELS.find((m) => m.slug === model).name} predicts</span>
          <b style={{ color: CLASS_COLORS[CLASSES[netForModel.predicted]] }}>{DISPLAY[CLASSES[netForModel.predicted]]}</b>
          <span>{pct(netForModel.probabilities[netForModel.predicted])}</span>
          <span className="muted">· computed in {netForModel.inference_ms} ms on the server</span>
        </div>
      )}
      {busy && <Loading text="Running the image through every layer…" />}
      {err && <div className="error">⚠ {err}</div>}
      {!image && <p className="note">Upload or pick an image to fill the network with real values. Until then, the structure is shown with empty nodes.</p>}

      <Card className="nv-card-wrap">
        {graph ? <NetworkView graph={graph} net={netForModel} k={k} /> : <Loading text="Loading the model graph…" />}
      </Card>
    </Section>
  );
}
