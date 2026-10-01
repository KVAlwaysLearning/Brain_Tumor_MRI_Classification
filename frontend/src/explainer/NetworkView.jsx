import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CLASSES, CLASS_COLORS, DISPLAY } from "../lib/theme";

const COL = 136;          // width of one layer column (px)
const ROW = 46;           // vertical spacing between nodes
const TOP = 34;           // space above the first node
const R = 13;             // node radius
const BUFFER = 6;         // extra columns rendered outside the viewport
const PAD_RIGHT = 190;    // room for the class labels next to the output layer

const KIND_COLOR = { input: "#94a3b8", weight: "#34d399", depthwise: "#2dd4bf", bn: "#22d3ee", pass: "#a78bfa", merge: "#fbbf24" };
const KIND_LABEL = { input: "input", weight: "weights", depthwise: "depthwise", bn: "batch-norm", pass: "no weights", merge: "merge" };

const fmtShape = (s) => s.join(" × ");
const sig = (v) => (v === null || v === undefined ? "—" : Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? Number(v).toExponential(2) : Number(v).toPrecision(3));

export default function NetworkView({ graph, net, k }) {
  const scroller = useRef(null);
  const [view, setView] = useState({ left: 0, width: 1200 });
  const [expanded, setExpanded] = useState(() => new Set());
  const [hover, setHover] = useState(null);             // { layer, i }
  const drag = useRef(null);

  const layers = graph.layers;
  const col = useMemo(() => Object.fromEntries(layers.map((l, i) => [l.name, i])), [layers]);
  const netByName = useMemo(() => (net ? Object.fromEntries(net.layers.map((l) => [l.name, l])) : {}), [net]);

  // blocks = consecutive runs of layers with the same block name
  const blocks = useMemo(() => {
    const out = [];
    layers.forEach((l, i) => {
      const last = out[out.length - 1];
      if (last && last.name === l.block) { last.end = i; last.params += l.params; }
      else out.push({ name: l.block, start: i, end: i, params: l.params });
    });
    return out;
  }, [layers]);

  // nodes shown per layer: real selection from the server, or placeholders before an image is analysed
  const shown = useCallback((l) => {
    const n = netByName[l.name];
    if (n) return n.nodes;
    const total = l.shape[l.shape.length - 1];
    return Array.from({ length: Math.min(total, k) }, (_, i) => ({ i, v: null }));
  }, [netByName, k]);
  const maxNodes = Math.max(...layers.map((l) => Math.min(l.shape[l.shape.length - 1], k)));
  const height = TOP + maxNodes * ROW + 40;

  const yOf = useCallback((name, ch) => {
    const idx = shown(layers[col[name]]).findIndex((n) => n.i === ch);
    return idx < 0 ? null : TOP + idx * ROW + R;
  }, [shown, layers, col]);

  const onScroll = () => {
    const el = scroller.current;
    requestAnimationFrame(() => setView({ left: el.scrollLeft, width: el.clientWidth }));
  };
  useEffect(() => { onScroll(); }, [graph]);
  useEffect(() => { scroller.current.scrollLeft = 0; setExpanded(new Set()); }, [graph]);

  const first = Math.max(0, Math.floor(view.left / COL) - BUFFER);
  const last = Math.min(layers.length - 1, Math.ceil((view.left + view.width) / COL) + BUFFER);
  const visible = layers.slice(first, last + 1);

  const toggleBlock = (b) => setExpanded((s) => { const n = new Set(s); n.has(b) ? n.delete(b) : n.add(b); return n; });
  const jump = (i) => scroller.current.scrollTo({ left: Math.max(0, i * COL - 40), behavior: "smooth" });
  const pan = (dx) => scroller.current.scrollBy({ left: dx, behavior: "smooth" });

  // edges into visible target layers, including long skip connections from off-screen sources
  const edges = [];
  const nodeEdges = hover ? new Set() : null;
  for (const l of layers.slice(first, last + 1)) {
    const n = netByName[l.name];
    if (!n) continue;
    const maxW = Math.max(1e-9, ...n.edges.filter((e) => e[3] !== null).map((e) => Math.abs(e[3])));
    for (const [src, s, o, w] of n.edges) {
      if (col[src] === undefined) continue;
      const y1 = yOf(src, s), y2 = yOf(l.name, o);
      if (y1 === null || y2 === null) continue;
      const on = hover && ((hover.layer === l.name && hover.i === o) || (hover.layer === src && hover.i === s));
      edges.push({ key: `${src}-${s}-${l.name}-${o}`, x1: col[src] * COL + COL / 2 + R, y1, x2: col[l.name] * COL + COL / 2 - R, y2, w, maxW, skip: col[l.name] - col[src] > 1, on });
      if (on) nodeEdges.add(`${src}:${s}`).add(`${l.name}:${o}`);
    }
  }

  const predicted = net?.predicted;
  const hoverLayer = hover && layers[col[hover.layer]];
  const hoverNode = hover && shown(hoverLayer).find((n) => n.i === hover.i);

  return (
    <div className="nv">
      {/* overview bar: blocks proportional to their layer count; click to jump */}
      <div className="nv-overview">
        {blocks.map((b) => (
          <button key={b.name + b.start} className="nv-ov-block" style={{ flex: b.end - b.start + 1 }} onClick={() => jump(b.start)} title={`${b.name}: ${b.end - b.start + 1} layers`}>
            <span>{b.name}</span>
          </button>
        ))}
        <div className="nv-ov-window" style={{ left: `${(view.left / (layers.length * COL + PAD_RIGHT)) * 100}%`, width: `${Math.min(100, (view.width / (layers.length * COL + PAD_RIGHT)) * 100)}%` }} />
      </div>
      <div className="nv-toolbar">
        <button className="link-btn" onClick={() => pan(-view.width * 0.8)}>◀ Scroll left</button>
        <span className="muted">{layers.length} layers · {graph.total_params.toLocaleString()} parameters · drag the canvas or use the bar above to move</span>
        <span className="nv-tools-right">
          <button className="link-btn" onClick={() => setExpanded(new Set(blocks.map((b) => b.name)))}>Expand all</button>
          <button className="link-btn" onClick={() => setExpanded(new Set())}>Collapse all</button>
          <button className="link-btn" onClick={() => pan(view.width * 0.8)}>Scroll right ▶</button>
        </span>
      </div>

      <div className="nv-scroll" ref={scroller} onScroll={onScroll}
        onMouseDown={(e) => { if (e.target.closest("button,.nv-card")) return; drag.current = { x: e.clientX, left: scroller.current.scrollLeft }; }}
        onMouseMove={(e) => { if (drag.current) scroller.current.scrollLeft = drag.current.left - (e.clientX - drag.current.x); }}
        onMouseUp={() => (drag.current = null)} onMouseLeave={() => (drag.current = null)}>
        <div className="nv-inner" style={{ width: layers.length * COL + PAD_RIGHT }}>

          {/* pinned header: block band + layer names */}
          <div className="nv-sticky">
            <div className="nv-blocks">
              {blocks.map((b) => (
                <button key={b.name + b.start} className={`nv-block ${expanded.has(b.name) ? "open" : ""}`} style={{ left: b.start * COL, width: (b.end - b.start + 1) * COL }} onClick={() => toggleBlock(b.name)}
                  title={`${b.end - b.start + 1} layers · ${b.params.toLocaleString()} params`}>
                  <span className="nv-block-name">{expanded.has(b.name) ? "▾" : "▸"} {b.name}</span>
                  <span className="nv-block-meta">{b.end - b.start + 1} layers · {b.params.toLocaleString()} params</span>
                </button>
              ))}
            </div>
            <div className="nv-names">
              {visible.map((l) => (
                <div key={l.name} className="nv-name" style={{ left: col[l.name] * COL, width: COL }} title={l.name}>
                  <div className="nv-name-main">{l.name}</div>
                  <div className="nv-name-sub"><span style={{ color: KIND_COLOR[l.kind] }}>●</span> {l.type} · {l.shape[l.shape.length - 1].toLocaleString()} nodes</div>
                </div>
              ))}
            </div>
          </div>

          {/* schematic: one card per layer; blocks expand to full details */}
          <div className="nv-schematic">
            {visible.map((l) => {
              const open = expanded.has(l.block);
              const n = netByName[l.name];
              return (
                <div key={l.name} className={`nv-card ${open ? "open" : ""}`} style={{ left: col[l.name] * COL + 4, width: COL - 8, borderColor: KIND_COLOR[l.kind] + "66" }}>
                  <div className="nv-chip" style={{ color: KIND_COLOR[l.kind] }}>{KIND_LABEL[l.kind]}</div>
                  <div className="nv-shape">{fmtShape(l.shape)}</div>
                  <div className="nv-params">{l.params.toLocaleString()} params</div>
                  {open && (
                    <div className="nv-details">
                      {Object.entries(l.config).map(([k2, v]) => <div key={k2}><span>{k2}</span> {Array.isArray(v) ? v.join("×") : String(v)}</div>)}
                      <div><span>inputs</span> {l.inputs.join(", ") || "—"}</div>
                      {n && <><div><span>act. min</span> {sig(n.stats.min)}</div><div><span>act. max</span> {sig(n.stats.max)}</div><div><span>act. mean</span> {sig(n.stats.mean)}</div><div><span>zeros</span> {(n.stats.zero_frac * 100).toFixed(1)}%</div></>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* the network: nodes = most-activated channels/units, lines = real weights */}
          <svg className="nv-svg" width={layers.length * COL + PAD_RIGHT} height={height}>
            <defs>
              <filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="5" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            {edges.map((e) => {
              const width = e.w === null ? 0.8 : 0.5 + 2.8 * (Math.abs(e.w) / e.maxW);
              const color = e.w === null ? "#64748b" : e.w >= 0 ? "#34d399" : "#fb7185";
              const d = e.skip
                ? `M${e.x1},${e.y1} C${e.x1 + 60},${Math.min(e.y1, e.y2) - 70} ${e.x2 - 60},${Math.min(e.y1, e.y2) - 70} ${e.x2},${e.y2}`
                : `M${e.x1},${e.y1} C${(e.x1 + e.x2) / 2},${e.y1} ${(e.x1 + e.x2) / 2},${e.y2} ${e.x2},${e.y2}`;
              return <path key={e.key} d={d} fill="none" stroke={color} strokeWidth={e.on ? width + 1.2 : width} strokeDasharray={e.w === null || e.skip ? "3 3" : undefined}
                opacity={hover ? (e.on ? 0.95 : 0.06) : 0.42}><title>{e.w === null ? "identity (no weight)" : `weight ${sig(e.w)}`}</title></path>;
            })}
            {visible.map((l) => {
              const nodes = shown(l);
              const cx = col[l.name] * COL + COL / 2;
              const maxV = Math.max(1e-9, ...nodes.map((n) => Math.abs(n.v ?? 0)));
              const isOut = l.name === layers[layers.length - 1].name;
              const total = l.shape[l.shape.length - 1];
              return (
                <g key={l.name}>
                  {nodes.map((n, j) => {
                    const cy = TOP + j * ROW + R;
                    const t = n.v === null ? 0 : Math.abs(n.v) / maxV;
                    const top = isOut && predicted === n.i;
                    const lit = hover && nodeEdges?.has(`${l.name}:${n.i}`);
                    return (
                      <g key={n.i} className="nv-node" onMouseEnter={() => setHover({ layer: l.name, i: n.i })} onMouseLeave={() => setHover(null)}>
                        <circle cx={cx} cy={cy} r={top ? R + 3 : R} filter={top ? "url(#glow)" : undefined}
                          fill={n.v === null ? "#07110c" : n.v >= 0 ? `rgba(16,185,129,${0.15 + 0.8 * t})` : `rgba(251,113,133,${0.15 + 0.8 * t})`}
                          stroke={top ? "#a7f3d0" : lit ? "#ffffff" : isOut ? CLASS_COLORS[CLASSES[n.i]] : "rgba(110,231,183,0.45)"} strokeWidth={top ? 2.6 : 1.2} />
                        <text x={cx} y={cy + R + 11} className="nv-val">{n.v === null ? "" : isOut ? `${(n.v * 100).toFixed(1)}%` : sig(n.v)}</text>
                        <text x={cx} y={cy + 4} className="nv-idx">{isOut ? "" : n.i}</text>
                        {isOut && <text x={cx + R + 8} y={cy + 4} className={`nv-class ${top ? "top" : ""}`} style={{ fill: CLASS_COLORS[CLASSES[n.i]] }}>{DISPLAY[CLASSES[n.i]]}{top ? "  ◀ predicted" : ""}</text>}
                      </g>
                    );
                  })}
                  <text x={cx} y={TOP + nodes.length * ROW + 18} className="nv-more">
                    {total > nodes.length ? `+${(total - nodes.length).toLocaleString()} more of ${total.toLocaleString()}` : `all ${total} shown`}
                  </text>
                </g>
              );
            })}
          </svg>

          {hover && hoverNode && (
            <NodeTip layer={hoverLayer} node={hoverNode} x={col[hover.layer] * COL + COL / 2} y={height} isOut={hover.layer === layers[layers.length - 1].name} total={hoverLayer.shape[hoverLayer.shape.length - 1]} />
          )}
        </div>
      </div>
      <div className="nv-legend">
        <span><i className="lg pos" /> positive weight</span><span><i className="lg neg" /> negative weight</span><span><i className="lg id" /> identity / skip connection</span>
        <span>Line thickness = |weight| (conv: 3×3 kernel summed; depthwise: per channel; BN: γ/√(σ²+ε); dense: exact)</span>
        <span>Node value = channel mean after that layer (dense: exact unit value)</span>
      </div>
    </div>
  );
}

function NodeTip({ layer, node, x, isOut, total }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!node.t || !ref.current) return;
    const px = Uint8Array.from(atob(node.t), (c) => c.charCodeAt(0));
    const side = Math.round(Math.sqrt(px.length));
    const ctx = ref.current.getContext("2d"); ref.current.width = side; ref.current.height = side;
    const img = ctx.createImageData(side, side);
    px.forEach((v, i) => { img.data[i * 4] = v * 0.2; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v * 0.6; img.data[i * 4 + 3] = 255; });
    ctx.putImageData(img, 0, 0);
  }, [node]);
  return (
    <div className="nv-tip" style={{ left: x + 24, top: 140 }}>
      <div className="nv-tip-title">{layer.name}</div>
      <div className="muted">{layer.type} · {isOut ? `class ${DISPLAY[CLASSES[node.i]]}` : `node ${node.i} of ${total.toLocaleString()}`}</div>
      <div className="nv-tip-val">{node.v === null ? "analyse an image to see values" : isOut ? `p = ${(node.v * 100).toFixed(2)}%` : `value ${sig(node.v)}`}</div>
      {node.t && <><canvas ref={ref} className="nv-tip-map pixelated" /><div className="muted small">feature map · min {sig(node.lo)} · max {sig(node.hi)}</div></>}
    </div>
  );
}
