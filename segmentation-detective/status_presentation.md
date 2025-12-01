# Segmentation Detective: XAI Analysis of Tile2Net

## Project Status & Presentation Documentation

**Date:** December 2024
**Authors:** Tile2Net XAI Research Team
**Model:** MscaleOCR with HRNet-W48 backbone
**Task:** Explainable AI for Pedestrian Infrastructure Segmentation

---

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [XAI Methods Implemented](#xai-methods-implemented)
4. [Ground Truth Validation](#ground-truth-validation)
5. [Results Summary](#results-summary)
6. [Future Directions](#future-directions)

---

## Overview

This project applies Explainable AI (XAI) techniques to understand how Tile2Net's semantic segmentation model detects pedestrian infrastructure (sidewalks, roads, crosswalks) from aerial imagery. The goal is to provide transparency into the model's decision-making process for urban planning applications.

### Key Objectives
- Visualize which image regions influence predictions for each class
- Compare model outputs against official NYC ground truth data
- Identify model strengths and limitations through interpretability analysis

---

## Model Architecture

### Tile2Net Segmentation Model

| Component | Details |
|-----------|---------|
| **Architecture** | MscaleOCR (Object-Contextual Representations) |
| **Backbone** | HRNet-W48 (High-Resolution Network) |
| **Classes** | 4 (Sidewalk, Road, Crosswalk, Background) |
| **Input Size** | 512×512 (stitched from 256×256 tiles) |
| **Pretrained** | ImageNet + satellite_2021.pth |

### Class Definitions (from `satellite.py`)

| trainId | Class | Color (RGB) | Description |
|---------|-------|-------------|-------------|
| 0 | Sidewalk | Blue (0, 0, 255) | Pedestrian walkways, paths |
| 1 | Road | Green (0, 128, 0) | Vehicle roadways |
| 2 | Crosswalk | Red (255, 0, 0) | Intersection crossings |
| 3 | Background | Black (0, 0, 0) | Buildings, vegetation, other |

### HRNet-W48 Multi-Scale Structure

```
Stage2: 2 branches (high-res 48ch + low-res 96ch)
  └── Branch 0: 4 BasicBlocks (1/4 resolution)
  └── Branch 1: 4 BasicBlocks (1/8 resolution)

Stage3: 3 branches (multi-scale fusion)
  └── Branch 0: 4 BasicBlocks (1/4 resolution)
  └── Branch 1: 4 BasicBlocks (1/8 resolution)
  └── Branch 2: 4 BasicBlocks (1/16 resolution)

Stage4: 4 branches (final multi-scale)
  └── Branch 0: 4 BasicBlocks (1/4 resolution)
  └── Branch 1: 4 BasicBlocks (1/8 resolution)
  └── Branch 2: 4 BasicBlocks (1/16 resolution)
  └── Branch 3: 4 BasicBlocks (1/32 resolution)
```

**Why HRNet?** HRNet maintains high-resolution representations throughout the network, making it ideal for dense prediction tasks like semantic segmentation where spatial precision matters.

---

## XAI Methods Implemented

### Cell-by-Cell Breakdown

#### Cell 1-3: Environment Setup
```python
!git clone https://github.com/mannadamay12/tile2net-segD
!pip install -e . grad-cam captum saliency
```
**Purpose:** Clone repository, install Tile2Net and XAI libraries (pytorch-grad-cam, Captum, saliency).

#### Cell 4: Model Loading
```python
net = MscaleOCR(num_classes=4, trunk='hrnetv2', criterion=None)
checkpoint = torch.load(SEG_CHECKPOINT)
net.load_state_dict(new_state_dict)
net = net.cuda().eval()
```
**Purpose:** Load pretrained weights from `satellite_2021.pth` and configure model for inference.

#### Cell 5: Model Wrapper for XAI Compatibility
```python
class ModelWrapper(nn.Module):
    def __init__(self, model):
        self.model = model
        self.backbone = model.backbone  # Expose for GradCAM

    def forward(self, x):
        output = self.model({'images': x})
        return output.get('pred', ...)
```
**Why?** pytorch-grad-cam requires direct tensor output, but MscaleOCR returns a dictionary. The wrapper standardizes the interface.

#### Cell 6-7: Tile Loading & Inference
**Purpose:** Load 512×512 stitched tile from Massachusetts example, run inference, visualize segmentation mask with class distribution.

#### Cell 8: Class Order Verification
**Why?** Critical step to confirm the mapping between model output indices and semantic classes. We discovered the correct order from `satellite.py`:
- Index 0 → Sidewalk (not Road!)
- Index 1 → Road
- Index 2 → Crosswalk
- Index 3 → Background

---

### XAI Method 1: GradCAM (Cell 10)

**What it does:** Gradient-weighted Class Activation Mapping computes gradients of the target class score with respect to feature maps, then weights activations by gradient importance.

**Why we chose it:**
- Most widely-used XAI method for CNNs
- Produces interpretable heatmaps showing "where the model looks"
- Works with any differentiable architecture

**Target Layers:**
```python
target_layers = {
    'Stage2-High': backbone.stage2[-1].branches[0][-1],  # High-res features
    'Stage2-Low':  backbone.stage2[-1].branches[1][-1],  # Context features
    'Stage3-High': backbone.stage3[-1].branches[0][-1],
    'Stage3-Mid':  backbone.stage3[-1].branches[1][-1],
    'Stage4-High': backbone.stage4[-1].branches[0][-1],  # Final high-res
    'Stage4-Low':  backbone.stage4[-1].branches[3][-1],  # Final context
}
```

**Why multiple layers?** HRNet's parallel branches capture different scales. High-resolution branches preserve spatial details (edges, boundaries), while low-resolution branches capture semantic context (what type of surface).

**Results:**
- Sidewalk: Strong activations along walkway edges and boundaries
- Road: Activations on road surfaces, especially lane markings
- Crosswalk: Focused activations on striped crossing patterns
- Background: Diffuse activations on buildings/vegetation

---

### XAI Method 2: LayerCAM (Cell 11)

**What it does:** Extends GradCAM by using element-wise product of gradients and activations, then applying ReLU to keep only positive contributions.

**Why we chose it:**
- Better handles multi-layer aggregation
- More fine-grained than standard GradCAM
- Particularly useful for HRNet's multi-branch architecture

**Configuration:**
```python
layer_cam_targets = {
    'Stage2-Multi': [branches[0], branches[1]],      # Both Stage2 branches
    'Stage3-Multi': [branches[0], branches[1], branches[2]],  # All 3
    'Stage4-Multi': [branches[0], branches[3]],      # High + Low
}
```

**Results:** LayerCAM produces sharper boundaries than GradCAM, especially for narrow features like sidewalk edges.

---

### XAI Method 3: ScoreCAM (Cell 12)

**What it does:** Gradient-free method that perturbs input using each activation map as a mask, then measures how each perturbation affects the output score.

**Why we chose it:**
- No gradient computation required (avoids gradient saturation issues)
- More faithful to actual model behavior
- Works as validation against gradient-based methods

**Memory Optimization:**
```python
resize_dim = 256  # Reduced from 512 for memory
resized_input = resize_transform(input_tensor)
```

**Why resize?** ScoreCAM requires forward passes for each activation channel, which is memory-intensive. Resizing to 256×256 reduces memory usage by 4× while preserving meaningful attributions.

**Results:** ScoreCAM confirms GradCAM findings but with smoother, more region-based attributions.

---

### XAI Method 4: Integrated Gradients (Cell 13)

**What it does:** Computes attributions by integrating gradients along a path from a baseline (black image) to the input. Based on Shapley values from game theory.

**Why we chose it:**
- Theoretically grounded (satisfies axioms of attribution)
- Provides pixel-level importance scores
- Different perspective than CAM-based methods

**Implementation:**
```python
class IGWrapper(nn.Module):
    def forward(self, x):
        logits = self.model({'images': x})['pred']
        return logits[:, self.target_class, :, :].sum()  # Scalar output

ig = IntegratedGradients(ig_model)
attributions = ig.attribute(input, baselines=zeros, n_steps=50)
```

**Why 50 steps?** Approximates the integral; more steps = more accurate but slower. 50 provides good balance.

**Results:** IG highlights texture patterns (crosswalk stripes, pavement texture) more than CAM methods, which focus on regions.

---

### XAI Method 5: Confidence Maps (Cell 14)

**What it does:** Visualizes per-class probability distributions from softmax output.

**Why we chose it:**
- Simple, direct interpretation of model uncertainty
- Shows where model is confident vs. uncertain
- No additional computation required

**Implementation:**
```python
probs = F.softmax(pred_logits, dim=1)  # [B, 4, H, W]
confidence_map = probs[predicted_class]  # Per-pixel confidence
```

**Results:**
- High confidence (>0.9) in road centers and building interiors
- Lower confidence (0.5-0.7) at class boundaries (sidewalk-road edges)
- Mean confidence: 0.86 (indicating overall model certainty)

---

### XAI Method 6: XRAI (Cell 15)

**What it does:** Region-based attribution that segments the image and ranks regions by importance. Combines gradient information with superpixel segmentation.

**Why we chose it:**
- Produces human-interpretable region-level explanations
- Better aligned with how humans perceive images
- Google's recommended method for image explanations

**Implementation:**
```python
class XRAIWrapper:
    def forward(self, x):
        probs = F.softmax(model(x)['pred'], dim=1)
        return {"INPUT_OUTPUT_GRADIENTS": probs[:, target_class]}

xrai = XRAI()
attr = xrai.GetMask(image, call_model)
```

**Results:** XRAI identifies coherent regions (entire sidewalk segments, road sections) rather than scattered pixels, making explanations more actionable for urban planners.

---

## Ground Truth Validation

### NYC Open Data Comparison (Cells 16-18)

#### Data Sources
- **Model Input:** NYC Ortho 2024 imagery from ArcGIS REST API
- **Ground Truth:** NYC Sidewalk dataset (52n9-sdep) from NYC Open Data

#### Methodology
1. Download 512×512 tile from Washington Square Park area
2. Run model inference to get sidewalk predictions
3. Fetch official NYC sidewalk polygons via Socrata API
4. Rasterize vector ground truth to match prediction resolution
5. Calculate IoU, Precision, Recall, F1

#### API Query
```python
base_url = "https://data.cityofnewyork.us/resource/52n9-sdep.geojson"
spatial_query = f"$where=within_box(the_geom, {north}, {west}, {south}, {east})"
```

#### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | 99.9% | Model finds ALL official sidewalks |
| **Precision** | 9.6% | Model detects more than GT includes |
| **IoU** | 9.6% | Reflects semantic mismatch |
| **F1** | 17.5% | Harmonic mean |
| **Mean Confidence** | 0.861 | High model certainty |

#### Pixel Counts
- Model Sidewalk: 34,491 pixels (13.2% of image)
- Ground Truth: 3,300 pixels (1.3% of image)
- Intersection: 3,298 pixels (99.9% of GT)

### Key Insight: Semantic Mismatch

**What the model detects as "Sidewalk":**
- Street-adjacent sidewalks ✓
- Park paths and trails
- Building entrances
- Plazas and courtyards
- Driveways
- Any paved pedestrian surface

**What NYC Ground Truth includes:**
- Street-adjacent sidewalks ONLY

**Conclusion:** The low precision/IoU is NOT a model failure—it's a semantic definition mismatch. The model successfully detects ALL official sidewalks (99.9% recall) AND expands coverage to include additional pedestrian-accessible surfaces, which is more useful for comprehensive pedestrian network mapping.

---

## Results Summary

### XAI Findings

| Method | Key Finding |
|--------|-------------|
| **GradCAM** | Model focuses on edges/boundaries for sidewalk, texture for crosswalk |
| **LayerCAM** | Multi-scale features combine high-res boundaries with semantic context |
| **ScoreCAM** | Gradient-free validation confirms CAM findings |
| **Integrated Gradients** | Texture patterns (stripes, pavement) are important features |
| **Confidence Maps** | Model is 86% confident on average; uncertain at class boundaries |
| **XRAI** | Region-level explanations align with human perception |

### Model Strengths
1. **High Recall:** Successfully detects nearly all ground truth infrastructure
2. **Generalization:** Extends beyond narrow definitions to all pedestrian surfaces
3. **Multi-scale Reasoning:** HRNet architecture captures both fine details and context
4. **High Confidence:** Model is certain in its predictions (mean 0.86)

### Model Limitations
1. **Boundary Precision:** Lower confidence at class transitions
2. **Semantic Scope:** Trained definition broader than official datasets
3. **Small Features:** Crosswalks can be missed if partially visible

---

## Future Directions

### Short-term Improvements

1. **Attention Visualization**
   - Implement attention rollout for transformer-based comparison
   - Visualize OCR module's object-context attention weights

2. **Comparative Analysis**
   - Test on multiple cities (LA, Chicago, Boston)
   - Compare US vs. international urban contexts

3. **Uncertainty Quantification**
   - Add Monte Carlo Dropout for epistemic uncertainty
   - Develop confidence calibration metrics

### Medium-term Extensions

4. **Interactive Dashboard**
   - Build Streamlit/Gradio app for real-time XAI exploration
   - Allow users to select regions and view explanations

5. **Counterfactual Explanations**
   - "What would need to change for this to be classified as road?"
   - Useful for understanding misclassifications

6. **Temporal Analysis**
   - Compare predictions across imagery years
   - Detect infrastructure changes over time

### Long-term Research

7. **Human-in-the-Loop Evaluation**
   - User studies with urban planners
   - Measure if XAI improves trust and decision-making

8. **Model Improvement from XAI Insights**
   - Use attention patterns to guide data augmentation
   - Identify failure modes and collect targeted training data

9. **Multi-Modal Explanations**
   - Combine visual explanations with natural language
   - Generate textual descriptions of model decisions

---

## Reproducibility

### Environment
```bash
conda create -n tile2net-xai python=3.11
conda activate tile2net-xai
pip install -e .
pip install grad-cam captum saliency geopandas pyproj
```

### Running the Notebook
1. Upload to Kaggle with GPU (T4 or better)
2. Run all cells sequentially
3. Results will be displayed inline

### Key Files
- `presentation.ipynb` - Main XAI analysis notebook
- `precompute_assets.ipynb` - Asset generation for presentations
- `status_presentation.md` - This documentation

---

## References

1. **Tile2Net Paper:** Hosseini et al. (2023). "Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery." *Computers, Environment and Urban Systems*.

2. **HRNet:** Wang et al. (2020). "Deep High-Resolution Representation Learning for Visual Recognition." *TPAMI*.

3. **GradCAM:** Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*.

4. **Integrated Gradients:** Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks." *ICML*.

5. **XRAI:** Kapishnikov et al. (2019). "XRAI: Better Attributions Through Regions." *ICCV*.

---

*Generated for Tile2Net XAI Research - December 2024*
