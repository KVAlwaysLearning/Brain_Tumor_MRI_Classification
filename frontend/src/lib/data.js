// Static JSON produced offline from the notebook run (tools/export_*.py). Cached per file.
const cache = {};
export function loadData(name) {
  if (!cache[name]) cache[name] = fetch(`/data/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load ${name}.json`);
    return r.json();
  });
  return cache[name];
}
