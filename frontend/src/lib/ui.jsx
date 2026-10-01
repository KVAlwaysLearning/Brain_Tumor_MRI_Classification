import { useState } from "react";
import SpotlightCard from "../reactbits/SpotlightCard";
import BlurText from "../reactbits/BlurText";

export function Section({ id, kicker, title, intro, children }) {
  return (
    <section id={id} className="section">
      <div className="section-head">
        {kicker && <div className="kicker">{kicker}</div>}
        <BlurText text={title} className="section-title" delay={60} animateBy="words" direction="top" />
        {intro && <p className="section-intro">{intro}</p>}
      </div>
      {children}
    </section>
  );
}

export function Card({ title, subtitle, children, className = "", right }) {
  return (
    <SpotlightCard className={`card ${className}`} spotlightColor="rgba(16, 185, 129, 0.16)">
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h3 className="card-title">{title}</h3>}
            {subtitle && <div className="card-sub">{subtitle}</div>}
          </div>
          {right}
        </div>
      )}
      {children}
    </SpotlightCard>
  );
}

// Notebook write-up for a chart (why / insight / impact), collapsed by default
export function Insight({ data }) {
  const [open, setOpen] = useState(false);
  if (!data) return null;
  return (
    <div className="insight">
      <p className="insight-main">{md(data.insight)}</p>
      {(data.why || data.impact) && (
        <button className="link-btn" onClick={() => setOpen(!open)}>{open ? "Hide details" : "Why this chart & business impact"}</button>
      )}
      {open && (
        <div className="insight-more">
          {data.why && <p><b>Why this chart.</b> {md(data.why)}</p>}
          {data.impact && <p><b>Business impact.</b> {md(data.impact)}</p>}
        </div>
      )}
    </div>
  );
}

// Minimal markdown: **bold** and *italic*
export function md(text = "") {
  const parts = String(text).split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> : p.startsWith("*") && p.length > 2 ? <i key={i}>{p.slice(1, -1)}</i> : p);
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <button key={t.value} className={`tab ${value === t.value ? "active" : ""}`} onClick={() => onChange(t.value)}>{t.label}</button>
      ))}
    </div>
  );
}

export function Stat({ label, children, sub }) {
  return (
    <div className="stat">
      <div className="stat-value">{children}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export const Loading = ({ text = "Loading…" }) => <div className="loading"><span className="spinner" />{text}</div>;
