// Model server client. Render's free plan sleeps after 15 min idle; the first request wakes it (~30-60 s).
let healthy = false;
const listeners = new Set();
export function onServerStatus(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function emit(s) { listeners.forEach((fn) => fn(s)); }

export async function wakeServer({ maxWaitMs = 120000 } = {}) {
  if (healthy) { emit("ready"); return true; }
  emit("waking");
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 15000);
      const r = await fetch("/api/health", { signal: ctrl.signal, cache: "no-store" });
      clearTimeout(t);
      if (r.ok) { healthy = true; emit("ready"); return true; }
    } catch { /* still waking */ }
    await new Promise((res) => setTimeout(res, 3000));
  }
  emit("offline");
  return false;
}

async function post(path, file) {
  await wakeServer();
  const body = new FormData();
  body.append("file", file, file.name || "image.jpg");
  const r = await fetch(path, { method: "POST", body });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
  return j;
}
export const predict = (file) => post("/api/predict", file);
export const explain = (file, model, k) => post(`/api/explain?model=${model}&k=${k}`, file);

export async function urlToFile(url) {
  const blob = await (await fetch(url)).blob();
  return new File([blob], url.split("/").pop(), { type: blob.type || "image/jpeg" });
}
