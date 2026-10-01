import { useEffect, useState } from "react";
import SplitText from "../reactbits/SplitText";
import ShinyText from "../reactbits/ShinyText";
import CountUp from "../reactbits/CountUp";
import StarBorder from "../reactbits/StarBorder";
import { loadData } from "../lib/data";
import { C } from "../lib/theme";

export default function Hero() {
  const [m, setM] = useState(null);
  const [eda, setEda] = useState(null);
  useEffect(() => { loadData("models").then(setM); loadData("eda").then(setEda); }, []);
  const best = m?.variants["MobileNetV2 — fine-tuned"]?.test;
  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  return (
    <header className="hero" id="top">
      <div className="hero-badge"><span className="dot" /> Deep learning · Medical imaging · 4-class MRI classification</div>
      <SplitText text="Brain Tumor MRI Classification" tag="h1" className="hero-title" delay={35} duration={0.9}
        from={{ opacity: 0, y: 30 }} to={{ opacity: 1, y: 0 }} rootMargin="0px" />
      <p className="hero-sub">
        <ShinyText text="Custom CNN · EfficientNetB0 · MobileNetV2 — trained on leakage-free data, explained layer by layer."
          color="#8fb3a3" shineColor="#d1fae5" speed={3} />
      </p>
      <div className="hero-stats">
        <div className="hero-stat"><div className="hero-num">{eda ? <CountUp to={eda.n_raw} separator="," duration={1.4} /> : "…"}</div><div className="hero-lbl">raw images</div></div>
        <div className="hero-arrow">→</div>
        <div className="hero-stat"><div className="hero-num">{eda ? <CountUp to={eda.n_clean} separator="," duration={1.6} /> : "…"}</div><div className="hero-lbl">unique after de-duplication</div></div>
        <div className="hero-stat"><div className="hero-num">{best ? <><CountUp to={+(best.accuracy * 100).toFixed(1)} duration={1.8} />%</> : "…"}</div><div className="hero-lbl">test accuracy (MobileNetV2)</div></div>
        <div className="hero-stat"><div className="hero-num">{best ? best.roc_auc_ovr.toFixed(3) : "…"}</div><div className="hero-lbl">ROC-AUC</div></div>
        <div className="hero-stat"><div className="hero-num">{best ? <><CountUp to={+(best.tumor_miss_rate * 100).toFixed(1)} duration={1} />%</> : "…"}</div><div className="hero-lbl">tumors missed on test</div></div>
      </div>
      <div className="hero-cta">
        <StarBorder as="button" color={C.emerald2} speed="5s" className="cta" onClick={() => go("predict")}>Upload an MRI</StarBorder>
        <StarBorder as="button" color="#22d3ee" speed="7s" className="cta ghost" onClick={() => go("explainer")}>Open the model explainer</StarBorder>
      </div>
      <p className="disclaimer">Educational project — not a medical device. Predictions must not be used for diagnosis.</p>
    </header>
  );
}
