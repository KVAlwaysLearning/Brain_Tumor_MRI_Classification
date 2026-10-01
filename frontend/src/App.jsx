import { useState } from "react";
import Aurora from "./reactbits/Aurora";
import Dock from "./reactbits/Dock";
import Hero from "./sections/Hero";
import Dataset from "./sections/Dataset";
import EDA from "./sections/EDA";
import Hypotheses from "./sections/Hypotheses";
import Models from "./sections/Models";
import Dashboard from "./sections/Dashboard";
import Predict from "./sections/Predict";
import Explainer from "./explainer/Explainer";

const icon = (d) => <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{d}</svg>;
const NAV = [
  ["top", "Home", icon(<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z" />)],
  ["data", "Dataset", icon(<><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>)],
  ["eda", "EDA charts", icon(<><path d="M4 20V10" /><path d="M10 20V4" /><path d="M16 20v-7" /><path d="M22 20H2" /></>)],
  ["hypotheses", "Hypotheses", icon(<><path d="M9 3h6" /><path d="M10 3v6L4 19a2 2 0 0 0 1.7 3h12.6A2 2 0 0 0 20 19l-6-10V3" /></>)],
  ["models", "Model results", icon(<><circle cx="12" cy="12" r="3" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4M5 5l3 3M16 16l3 3M5 19l3-3M16 8l3-3" /></>)],
  ["dashboard", "Dashboard", icon(<><rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></>)],
  ["predict", "Predict", icon(<><path d="M12 16V4" /><path d="M7 9l5-5 5 5" /><path d="M4 20h16" /></>)],
  ["explainer", "Explainer", icon(<><circle cx="5" cy="6" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="19" cy="12" r="2" /><circle cx="12" cy="9" r="2" /><circle cx="12" cy="15" r="2" /><path d="M7 6l3 2.5M7 18l3-2.5M7 6l3 8M14 9l3 2M14 15l3-2" /></>)],
];

export default function App() {
  const [image, setImage] = useState(null);     // shared between Predict and the Explainer
  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  return (
    <>
      <div className="bg-aurora"><Aurora colorStops={["#022c22", "#10b981", "#064e3b"]} amplitude={0.9} blend={0.55} /></div>
      <div className="bg-vignette" />
      <main className="page">
        <Hero />
        <Dataset />
        <EDA />
        <Hypotheses />
        <Models />
        <Dashboard />
        <Predict image={image} setImage={setImage} />
        <Explainer shared={image} />
        <footer className="footer">
          Brain Tumor MRI Classification · Deep learning capstone · Models run with LiteRT on the server · UI components from <a href="https://reactbits.dev" target="_blank" rel="noreferrer">React Bits</a>
          <div className="disclaimer">Educational project — not a medical device.</div>
        </footer>
      </main>
      <div className="dock-wrap"><Dock items={NAV.map(([id, label, ic]) => ({ icon: ic, label, onClick: () => go(id) }))} panelHeight={60} baseItemSize={44} magnification={62} /></div>
    </>
  );
}
