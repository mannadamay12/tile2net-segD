# Segmentation Detective

Segmentation Detective has two parts:

1. **A lightweight D3/Canvas UI** for browsing pre-exported tiles + XAI layers.
2. **Notebooks** used to run Tile2Net inference and generate XAI artifacts (GradCAM, LayerCAM, ScoreCAM, Integrated Gradients, XRAI, confidence maps) on specific examples.

---

## 1) D3 UI (`segdetect-d3/`)

### What it does
The UI loads a tile (RGB) plus optional overlays, then lets you explore XAI layers either:

- **Lens mode**: hover to reveal the selected method inside a circular “flashlight”.
- **Full overlay**: draw the selected method over the whole tile.

It also builds a **histogram from the confidence image** and lets you brush a range to highlight “risky” pixels inside the lens.

### Quick run (local)
You must serve the folder over HTTP (because the UI uses ES modules + `fetch`).

```bash
cd tile2net-segD/segmentation-detective/segdetect-d3
python3 -m http.server 8000
```

Open:

- http://localhost:8000

### Data layout (assets)
The UI is driven by:

- `segdetect-d3/assets/manifest.json`

That manifest declares:

- **classes**: list of semantic classes (id, name, fileKey)
- **methods**: available XAI methods and how to render them
- **tiles**: which tile folders are available and where their images live

#### Tile folder convention
For each tile id `T`, place images under:

```
segdetect-d3/assets/tiles/T/
  base/
    input.png                # required
    prediction.png            # optional (set in manifest)
  xai/
    confidence.png            # optional but used for histogram/risk brushing
    <method>_<classKey>.png   # class-wise methods, e.g. gradcam_sidewalk.png
```

Where:

- `classKey` comes from `manifest.json` → `classes[].fileKey` (fallback: class name).
- `method` comes from `manifest.json` → `methods[].id`.

`confidence.png` is treated as a **grayscale** image mapped to `[0,1]`.

- Default meaning is **confidence** (white = high confidence).
- If your exported map is **uncertainty** (white = risky), set in the manifest per tile:

```json
{ "confidenceMeaning": "uncertainty" }
```

#### Adding a new tile
1. Create `assets/tiles/<tileId>/base/input.png`.
2. Add your XAI images under `assets/tiles/<tileId>/xai/`.
3. Add the tile entry to `assets/manifest.json`:

```json
{
  "id": "city_002",
  "label": "My Tile",
  "base": {
    "input": "assets/tiles/city_002/base/input.png",
    "prediction": null,
    "gt": null,
    "errorMap": null
  },
  "xai": {
    "confidence": "assets/tiles/city_002/xai/confidence.png"
  },
  "confidenceMeaning": "confidence"
}
```

---

## 2) Notebooks (`notebooks/`)

All notebooks live in:

- `tile2net-segD/segmentation-detective/notebooks/`

These notebooks were primarily run in **Kaggle/Colab** (GPU recommended). They include shell cells (`!git clone`, `!pip install`, etc.).

### Common setup used in notebooks
- Clone and install the package:
  - `pip install -e .`
- Extra XAI deps used:
  - `grad-cam` (pytorch-grad-cam)
  - `captum`
  - `saliency` (for XRAI)
- Some flows download or reference model weights (HRNet ImageNet pretrain + Tile2Net segmentation checkpoint). In the original runs this was often handled via Kaggle datasets / explicit paths.

### Notebook index

#### `01_tile2net.ipynb`
**Purpose:** Baseline pipeline on the example tile(s).

Covers:
- Generate example tiles via `examples/example.sh`
- Load Tile2Net segmentation model (`MscaleOCR` / HRNet-W48)
- Run inference and visualize the predicted mask
- Inspect HRNet stages/branches and run **GradCAM** across multiple target layers


#### `01_tile2net_corrected_confmaps.ipynb`
**Purpose:** Extended XAI methods and confidence maps (includes fixes/diagnostics).

Covers (in addition to the above):
- **LayerCAM**
- **ScoreCAM** (typically with resized inputs to reduce memory)
- **Integrated Gradients** (Captum; wrapper to produce scalar class-specific output)
- **Confidence / probability maps** (softmax + predicted-class confidence)
- **XRAI** (Saliency library; wrapper compatible with XRAI call signature)


#### `02-tile2net.ipynb`
**Purpose:** NYC case study + ground truth comparison + error-region GradCAM.

Covers:
- Generate an NYC raster tile (Washington Square Park example) using `tile2net.Raster`
- Download NYC sidewalk ground truth via Socrata:
  - `https://data.cityofnewyork.us/resource/52n9-sdep.geojson`
  - query uses `within_box(the_geom, north, west, south, east)`
- Convert/rasterize geometries to align with the image
- Compute metrics (**IoU, precision, recall, F1**) from polygon intersections/unions
- Derive false-positive regions and run **GradCAM targeted on error masks**


#### `presentation.ipynb`
**Purpose:** Consolidated “story” notebook used for presenting the results.

Covers:
- End-to-end demo structure
- Model loading + inference
- Multiple XAI methods (GradCAM / LayerCAM / ScoreCAM / IG / XRAI)
- Confidence maps and the NYC ground-truth comparison

### Running notebooks locally (minimal)
If you want to run locally instead of Kaggle/Colab, start from the repo root:

```bash
cd tile2net-segD
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install grad-cam captum saliency
```

Then open the notebook files under `segmentation-detective/notebooks/` in Jupyter.

Notes:
- GPU is strongly recommended for GradCAM/ScoreCAM/IG/XRAI on large tiles.
- You will need to provide paths to the model weights if they are not already available in your environment.

---

## Where the UI assets come from
The D3 UI is intended to visualize **precomputed PNG exports** (input tile + confidence + XAI overlays) produced by the notebooks. To publish a new example in the UI:

1. Export images from a notebook into a `segdetect-d3/assets/tiles/<tileId>/...` folder.
2. Register the tile in `segdetect-d3/assets/manifest.json`.
3. Re-run the UI server (`python3 -m http.server`).
