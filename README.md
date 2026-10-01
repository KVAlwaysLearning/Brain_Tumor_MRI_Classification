# Brain Tumor MRI Classification — web app

Interactive companion to the notebook `Brain_Tumor_MRI_Classification_R02.ipynb`:

- **Dataset & leakage-free cleaning:** perceptual-hash near-duplicate removal, with every cross-split leak visualised.
- **Section 4 EDA:** all 15 charts, interactive, each with the notebook's write-up.
- **Hypothesis tests:** H1 and H2 (Kruskal–Wallis with post-hoc tests) and H3 (McNemar).
- **Model results:** for all 6 trained variants, the training histories, confusion matrices, metric charts, grid search, cross-validation and fine-tuning grids.
- **Comparison dashboard:** every metric for every model, ROC/PR curves, calibration, bootstrap confidence intervals, efficiency, final selection and explainability.
- **Predict:** upload an MRI and get the best model's prediction (MobileNetV2, 90.5% test accuracy), with input warnings.
- **Model explainer:** three model buttons. It shows a layer-by-layer schematic with exact parameter counts and the transformed, scaled input matrix. It runs your image through the real network: each layer shows its most-activated nodes out of its total, with real activation values and lines weighted by the real trained weights. Horizontal scroll, pinned headers and an overview bar make large networks navigable.

> Educational project — **not a medical device**.

## Architecture

```
Browser (React + Vite, reactbits.dev components, Plotly charts)
   │  static JSON (charts, schematics)  ── served with the site, no server work
   │  POST /api/predict, /api/explain   ── image upload
   ▼
FastAPI (one process) ── serves the built site + API
   ├─ preprocessing.py  PIL grayscale → 3 channels → 224×224 antialiased resize (numpy) → ÷255
   ├─ Predictor         MobileNetV2 (TFLite) — loaded at start-up
   └─ Explainer         one-output-per-layer TFLite model of the selected network, loaded on demand
```

- **No TensorFlow at runtime.** The models run on **LiteRT** (`ai-edge-litert`), which keeps memory at about 365 MB peak, inside Render's free 512 MB.
- **Predictions match the notebook.** Through the API, MobileNetV2 scores **143/158 = 0.9051** on the test set, the same as in Colab.
  - The numpy resize differs from TensorFlow's by at most 1 gray level on about 0.01% of pixels, and changed 0 predictions for all three models.
- **Every layer is separate in the explainer.** Each explainer model outputs all layers, so convolution, batch normalization and activation keep their own values. All 36 / 159 / 243 layers were verified against Keras (relative difference below 1e-4).
- **How the line weights are computed:**
  - convolution: the 3×3 kernel summed per input→output channel;
  - depthwise convolution: one value per channel;
  - batch normalization: γ/√(σ²+ε);
  - dense layers: the exact weight.
- **How nodes are chosen.** For each layer, the server picks the top-k channels (4–16) by mean activation. Channel-preserving layers keep the same channels in view so lines line up. The output layer always shows all 4 classes, with the highest probability highlighted.

## Repository layout

```
Dockerfile, render.yaml       one-service deployment (Render free plan, Docker runtime)
backend/
  app/main.py                 FastAPI app (API + static site)
  app/inference.py            LiteRT prediction + explainer network builder
  app/preprocessing.py        TF-free preprocessing + upload checks
  assets/                     *.tflite models, layer graphs, line-weight files, config.json
  requirements.txt
frontend/
  src/sections/               Hero, Dataset, EDA, Hypotheses, Models, Dashboard, Predict
  src/explainer/              Explainer, NetworkView (schematic + network), InputMatrix
  src/reactbits/              React Bits components (see licence note)
  public/data/                chart data exported from the notebook run
  public/img/                 thumbnails, sample test images, notebook figures
tools/                        offline export scripts that produced public/data and backend/assets
```

## Run locally

```bash
# backend (Python 3.10–3.12)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend (Node 20+) — in a second terminal
cd frontend
npm install
npm run dev            # http://localhost:5173 (proxies /api to :8000)
```

For a production-like run, build the site first (`npm run build`). FastAPI then serves `frontend/dist` at http://localhost:8000.

With Docker:

```bash
docker build -t brain-mri .
docker run -p 10000:10000 brain-mri      # http://localhost:10000
```

## Deploy on Render (free plan)

1. Push this folder to a GitHub repository. The largest file is about 17.5 MB, well under GitHub's limit.
2. In Render, choose **New → Blueprint**, select the repository, and confirm the `brain-tumor-mri` service from `render.yaml`. It uses the Docker runtime on the free plan, with health check `/api/health`.
3. Wait for the first build, about 5–10 minutes, then open the service URL.

**Free-plan behaviour:**
- **Sleeping:** the service sleeps after 15 minutes without traffic. The first visit afterwards takes about 30–60 s while it wakes. The site shows a "waking the model server" status, and all charts load immediately from static files without waiting for it.
- **Speed:** the free CPU is slow, so expect about 0.2–1 s per prediction and a few seconds per explainer run. The first EfficientNetB0 explanation after waking is the slowest.
- **Memory:** it's limited to 512 MB. Keep `--workers 1` (already set); explainer requests run one at a time.

## Regenerating the data and model files

`tools/export_eda.py` and `tools/export_models.py` rebuild `frontend/public/data` and `backend/assets` from the notebook run. They need TensorFlow, which is not needed on the server. Set `EXPORT_ROOT` to a folder containing:
- `project_data/`: the raw dataset;
- `clean/`: the de-duplicated dataset with `clean_metadata.csv`;
- the notebook's `reports/`, `metrics/`, `history/`, checkpoints and `deployment_config.json`.

`export_models.py` stops with an error if any variant's recomputed validation or test accuracy differs from the notebook's.

## Licences

- **React Bits components** (`frontend/src/reactbits/`): MIT + Commons Clause (see `LICENSE-reactbits.md`). Fine for personal, portfolio and course use; the components themselves may not be sold.
- **Dataset:** Roboflow export under CC BY 4.0.
