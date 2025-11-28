# Segmentation Detective - Progress Summary & Next Steps

**Project:** The Segmentation Detective (CS-GY 9223 Fall 2025)
**Team:** Swapnil Sharma, Uttam Singh, Adamay Mann
**Date:** November 8, 2025

---

## Executive Summary

We have successfully completed the foundational pipeline for analyzing Tile2Net segmentation failures using GradCAM explainability. The system can now identify where and why the model makes errors when detecting pedestrian infrastructure from aerial imagery.

---

## What We've Accomplished

### 1. Model Setup & Integration
- ✅ Loaded pre-trained Tile2Net HRNet-W48 model (MscaleOCR architecture)
- ✅ Fixed configuration issues for inference (MSCALE, TORCH_VERSION, MSCALE_LO_SCALE)
- ✅ Created `ModelWrapper` class to make MscaleOCR compatible with pytorch-grad-cam
- ✅ Integrated pytorch-grad-cam for explainability
- ✅ Successfully ran inference on aerial imagery tiles (Kaggle Tesla T4 GPU)

### 2. XAI Methods Implementation
- ✅ Identified target layers across HRNet stages (Stage 2, 3, 4)
- ✅ Implemented multiple XAI methods for all 4 segmentation classes:
  - Class 0: Road
  - Class 1: Sidewalk
  - Class 2: Crosswalk
  - Class 3: Background

**GradCAM (pytorch-grad-cam):**
- ✅ Visualized multi-scale feature attention across 6 HRNet branches:
  - `Stage2-HighRes`: backbone.stage2[-1].branches[0][-1]
  - `Stage2-LowRes`: backbone.stage2[-1].branches[1][-1]
  - `Stage3-HighRes`: backbone.stage3[-1].branches[0][-1]
  - `Stage3-MidRes`: backbone.stage3[-1].branches[1][-1]
  - `Stage4-HighRes`: backbone.stage4[-1].branches[0][-1]
  - `Stage4-LowRes`: backbone.stage4[-1].branches[3][-1]
- ✅ Confirmed different stages focus on different features:
  - **Stage 2**: Fine edges and boundaries (2 branches)
  - **Stage 3**: Mid-level semantic features (3 branches)
  - **Stage 4**: High-level context and object recognition (4 branches)

**LayerCAM (pytorch-grad-cam):**
- ✅ Multi-layer aggregation from Stage2, Stage3, Stage4
- ✅ Combines gradients from multiple resolution branches
- ✅ More detailed spatial information than single-layer methods

**ScoreCAM (pytorch-grad-cam):**
- ✅ Gradient-free method using model confidence scores
- ✅ Resized input to 256x256 for memory efficiency
- ✅ Applied to Stage2/3/4 HighRes branches

**Integrated Gradients (Captum):**
- ✅ Custom `IGWrapper` class for class-specific scalar output
- ✅ Baseline: black image with same normalization
- ✅ Integration with n_steps=50
- ✅ Visualizations: heatmap, overlay, positive attribution, top 20% important pixels

**Confidence Maps (PyTorch softmax):**
- ✅ Per-class probability maps using F.softmax on logits
- ✅ Overall confidence map (predicted class probability per pixel)
- ✅ Probability range visualization [0, 1]

**XRAI (saliency library):**
- ✅ Custom `XRAINetWrapperClassSpecific` class for model adaptation
- ✅ Region-based attribution with better boundary detection
- ✅ Visualizations: normalized attribution, positive overlay, top 10% important pixels

### 3. Ground Truth Comparison Pipeline
- ✅ Generated NYC tile (Washington Square Park area, bbox: [40.7285, -73.999, 40.731, -73.995])
- ✅ Downloaded NYC Planimetric sidewalk ground truth via Open Data API:
  - URL: `https://data.cityofnewyork.us/resource/52n9-sdep.geojson`
  - Query: `within_box(the_geom, north, west, south, east)`
  - Retrieved 15 MultiPolygon features
- ✅ Converted pixel predictions to georeferenced polygons using rasterio:
  - Used `rasterio.transform.from_bounds()` for affine transformation
  - Pixel size: ~0.00000536° lon/pixel, ~0.00000407° lat/pixel
  - Created GeoDataFrames with EPSG:4326 CRS
- ✅ Calculated quantitative metrics (IoU, Precision, Recall, F1)
- ✅ Identified and visualized error patterns (false positives/negatives)
- ✅ Rasterized false positive geometry for targeted GradCAM analysis

### 4. Error Analysis with GradCAM
- ✅ Applied GradCAM specifically to error regions using `SemanticSegmentationTarget`
- ✅ Rasterized false positive polygons using `rasterio.features.rasterize()`
- ✅ Generated attention heatmaps for 256,195 false positive pixels (24.4% of image)
- ✅ Explained model decision-making on false positives
- ✅ Discovered root cause of failures: texture-based classification without spatial context

### 5. Technical Implementation

**Notebook 01-tile2net.ipynb (Initial Prototype):**
- ✅ Massachusetts example data (Boston area: 42.35°N, -71.07°W)
- ✅ Established Kaggle environment with Tesla T4 GPU
- ✅ Created ModelWrapper class for GradCAM compatibility
- ✅ HRNet backbone structure analysis (stages 2, 3, 4 with multi-resolution branches)
- ✅ Initial GradCAM visualization on Sidewalk class

**Notebook 01_tile2net_corrected_confmaps.ipynb (Extended XAI Methods):**
- ✅ LayerCAM: Multi-layer aggregation visualization
- ✅ ScoreCAM: Gradient-free confidence-based attribution
- ✅ Integrated Gradients: Captum library with IGWrapper class
- ✅ Confidence Maps: Per-class probability and overall confidence visualization
- ✅ XRAI: Region-based attribution with XRAINetWrapperClassSpecific class
- ✅ Memory optimization: Resized inputs to 256x256 for ScoreCAM/IG/XRAI
- ✅ Dependencies: pytorch-grad-cam, captum, saliency

**Notebook 02-tile2net.ipynb (NYC + Ground Truth Pipeline):**
- ✅ Extended to NYC Washington Square Park data
- ✅ Added ground truth comparison pipeline
- ✅ Implemented quantitative metrics (IoU, Precision, Recall, F1)
- ✅ GradCAM on all 4 classes and error regions
- ✅ Dependencies: pytorch-grad-cam, geopandas, rasterio, shapely

**Common Configuration:**
- ✅ Full pipeline running on Kaggle with Tesla T4 GPU
- ✅ Model configuration:
  ```python
  cfg.MODEL.ARCH = 'ocrnet.HRNet_Mscale'
  cfg.DATASET.NUM_CLASSES = 4
  cfg.MODEL.OCR.MID_CHANNELS = 512
  cfg.MODEL.OCR.KEY_CHANNELS = 256
  cfg.MODEL.MSCALE = True
  cfg.MODEL.MSCALE_LO_SCALE = 0.5
  ```
- ✅ Preprocessing: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- ✅ Output: 1024x1024 prediction masks for stitched tiles (4x4 base tiles)

---

## Key Findings

### Quantitative Results

**Segmentation Performance (Washington Square Park):**
- **IoU:** 0.049 (very low overlap between prediction and ground truth)
- **Precision:** 5.7% (94.3% of predictions are false positives)
- **Recall:** 26.2% (73.8% of ground truth sidewalks missed)
- **F1 Score:** 0.093

**Error Distribution:**
- **False Positive Area:** 94.3% of all predictions
- **False Negative Area:** 73.8% of ground truth
- **False Positive Pixels:** 256,195 (24.4% of image)

### Qualitative Insights

#### 1. **Ground Truth Definition Mismatch** (Primary Issue)
The low metrics reveal a fundamental mismatch between what the model learned and what ground truth represents:

- **Model's learned concept:** "Any paved pedestrian surface" = sidewalk
  - Includes park paths, plazas, and walking areas
  - Based purely on visual texture and appearance

- **NYC Ground Truth definition:** Only street-adjacent sidewalks
  - Does NOT include park paths or public plaza walkways
  - Follows urban planning definition, not visual definition

**Impact:** The model detects the entire Washington Square Park paved area as sidewalk (all paths, plazas, walking surfaces), causing massive false positive rates.

#### 2. **Lack of Contextual Reasoning**
GradCAM analysis revealed the model makes decisions based on:

- **Surface texture:** Concrete/asphalt appearance
- **Color patterns:** Gray paved surfaces
- **Local features:** Edges, boundaries, material

**What the model CANNOT do:**
- Distinguish between park paths vs street sidewalks (identical appearance)
- Use spatial context (e.g., "sidewalk = adjacent to road")
- Apply semantic reasoning about urban infrastructure function

**Evidence:** High GradCAM activation (red/yellow heatmap) on park paths shows the model confidently identifies them as sidewalks based on visual similarity alone.

#### 3. **Model Behavior Across Classes**
- **Sidewalk (25.9%):** Over-predicted in parks, under-predicted on streets
- **Road (12.3%):** Generally correct on major streets
- **Crosswalk (1.6%):** Detected at intersections, small sample size
- **Background (60.2%):** Buildings, vegetation, parking lots

---

## Research Questions Answered

### Q1: Where does the segmentation model succeed or fail?

**Successes:**
- Major roads and highways (clear visual distinction)
- Crosswalks at intersections (distinctive striped pattern)
- Large, obvious sidewalks adjacent to roads

**Failures:**
- Park paths and plazas (24.4% false positive rate)
- Narrow or partially occluded sidewalks
- Areas with shadows, vegetation overlap, or poor image quality
- Contextual distinction between functionally different but visually similar surfaces

### Q2: How well do the GradCAM activation regions correspond to areas of high false positives/negatives?

**Perfect correspondence.**

- False positive regions show **high activation** (yellow/red in heatmap)
- Model confidently believes park paths are sidewalks
- Activation patterns focus on paved surface texture
- No evidence of contextual feature usage (e.g., proximity to roads)

### Q3: How can the dashboard incorporate error types or contextual information?

**Identified error categories:**

1. **Type 1: Park Path False Positives**
   - Paved paths in parks/plazas
   - Visually identical to sidewalks
   - Not in ground truth

2. **Type 2: Occlusion False Negatives**
   - Sidewalks under tree shadows
   - Partially covered by vegetation
   - Missed by model

3. **Type 3: Edge Confusion**
   - Building edges vs sidewalk edges
   - Parking lot boundaries
   - Material transitions

**Contextual filters needed:**
- Proximity to roads (buffer analysis)
- Park/public space detection
- Shadow/vegetation masking
- Urban vs suburban context

### Q4: Can interactive visualizations help identify and categorize recurring segmentation issues?

**Yes, demonstrated through:**
- Side-by-side prediction vs ground truth overlays
- Color-coded error maps (true positive/false positive/false negative)
- GradCAM heatmaps showing model attention
- Class distribution histograms

---

## Next Steps

### Phase 1: Expand Analysis (Week of Nov 11)

#### 1.1 Test on Multiple Tiles
**Goal:** Identify recurring error patterns across different contexts.

**Tasks:**
- Generate 10-15 NYC tiles covering diverse areas:
  - Residential neighborhoods (narrow sidewalks)
  - Commercial districts (wide sidewalks, heavy pedestrian traffic)
  - Parks and plazas (false positive test cases)
  - Shadowed areas (tree-lined streets)
  - Intersections (crosswalk detection)

- For each tile:
  - Run inference
  - Download ground truth
  - Calculate metrics
  - Generate GradCAM visualizations

- Create summary table:
  ```
  | Tile ID | Context      | IoU  | Precision | Recall | F1   | Primary Error Type |
  |---------|--------------|------|-----------|--------|------|--------------------|
  | WSP     | Park         | 0.05 | 0.06      | 0.26   | 0.09 | Park path FP       |
  | ...     | ...          | ...  | ...       | ...    | ...  | ...                |
  ```

#### 1.2 Categorize Error Types
**Goal:** Build a taxonomy of segmentation failures.

**Error Categories:**
1. **Contextual Confusion**
   - Park paths (high confidence, wrong context)
   - Parking lots (similar appearance to roads)
   - Building plazas (paved but not sidewalks)

2. **Environmental Factors**
   - Shadow occlusion (tree shadows, building shadows)
   - Vegetation overlap (overhanging trees, bushes)
   - Seasonal variation (winter vs summer imagery)

3. **Visual Ambiguity**
   - Narrow vs wide sidewalks
   - Weathered/damaged pavement
   - Construction zones (temporary surfaces)

4. **Edge Cases**
   - Bridges and overpasses
   - Tunnels and underpasses
   - Elevated walkways

**Implementation:**
- For each error type, collect 5-10 example tiles
- Generate GradCAM for each
- Document visual patterns the model uses
- Identify which HRNet stages contribute most to each error

#### 1.3 Compare GradCAM Across HRNet Stages
**Goal:** Understand feature hierarchy and error propagation.

**Analysis:**
- For the same tile, generate GradCAM for:
  - Stage 2, Branch 0 (high-res, early features)
  - Stage 3, Branch 1 (mid-res, semantic features)
  - Stage 4, Branch 0 (high-res, final features)
  - Stage 4, Branch 3 (low-res, context)

- Questions to answer:
  - Do early stages already show errors, or do they emerge later?
  - Which resolution branches contribute most to false positives?
  - Can we identify "decision points" where the model commits to wrong class?

**Deliverable:**
- Comparative visualization showing activation evolution
- Report on which stages/branches need improvement

---

### Phase 2: Interactive Presentation Notebook (Week of Nov 28 - Dec 5)

> **Note:** Backend/dashboard development deprioritized due to compute access limitations. Focus shifted to pre-computed interactive notebook format for in-class presentation.

#### 2.1 Create Story-Format Presentation Notebook
**Goal:** Build an interactive Jupyter notebook that tells the story of our findings.

**Notebook Structure:**

**Section 1: Introduction & Problem Statement**
- What is Tile2Net and why explainability matters
- Research questions we're answering
- Interactive: Show example tile with prediction overlay

**Section 2: Model Architecture Deep Dive**
- HRNet-W48 multi-resolution structure visualization
- How OCRNet handles semantic segmentation
- Interactive: Diagram of stage2/3/4 branches

**Section 3: XAI Methods Comparison**
- Side-by-side comparison of all 6 methods:
  - GradCAM, LayerCAM, ScoreCAM
  - Integrated Gradients, XRAI, Confidence Maps
- Interactive: Dropdown/tabs to switch between methods
- Pre-computed results for 3-5 example tiles

**Section 4: Ground Truth Analysis**
- NYC sidewalk ground truth comparison
- Error visualization (TP/FP/FN maps)
- Interactive: Slider to toggle overlays
- Quantitative metrics table

**Section 5: Key Findings & Insights**
- Why park paths cause false positives
- GradCAM activation patterns on error regions
- Model limitations: texture vs context

**Section 6: Conclusions & Future Work**
- Summary of contributions
- Recommendations for Tile2Net improvement

**Interactive Features (using ipywidgets):**
- Dropdown selectors for XAI method, class, tile
- Sliders for overlay opacity
- Toggle buttons for showing/hiding layers
- Pre-rendered images loaded on selection

#### 2.2 Pre-compute All Visualizations
**Goal:** Generate all visualizations ahead of time for smooth presentation.

**Assets to Pre-generate:**
- 5-10 example tiles from different contexts:
  - Parks (high FP)
  - Residential streets
  - Commercial areas
  - Intersections with crosswalks
- For each tile:
  - Original image
  - Segmentation prediction
  - Ground truth overlay (where available)
  - GradCAM for all 4 classes
  - LayerCAM, ScoreCAM, IG, XRAI visualizations
  - Confidence maps
  - Error maps (TP/FP/FN)

**Storage:** Save as PNG files in `segmentation-detective/assets/` directory

---

### Phase 3: Finalize Examples & Metrics (Week of Dec 2-5)

#### 3.1 Curate Example Tiles
**Goal:** Select diverse examples that demonstrate key findings.

**Example Categories:**
1. **High False Positive** - Park paths, plazas (Washington Square Park)
2. **High True Positive** - Clear street-adjacent sidewalks
3. **Occlusion Challenges** - Tree shadows, vegetation
4. **Crosswalk Detection** - Intersection examples
5. **Edge Cases** - Parking lots, building edges

#### 3.2 Compile Quantitative Results
**Goal:** Create summary tables and figures for presentation.

**Metrics to Include:**
- Per-tile: IoU, Precision, Recall, F1
- Per-class breakdown
- Error type distribution
- XAI method comparison (qualitative)

---

### Phase 5: Documentation & Final Report (Week of Dec 2-11)

#### 5.1 Code Documentation
- Add docstrings to all functions
- Create API documentation (Swagger/OpenAPI)
- Write README with setup instructions
- Add example usage notebooks

#### 5.2 Final Report Sections

**1. Introduction**
- Problem statement
- Motivation for explainability
- Research questions

**2. Related Work**
- Tile2Net overview
- GradCAM and XAI methods
- Prior work on segmentation debugging

**3. Methodology**
- HRNet-W48 architecture
- GradCAM adaptation for segmentation
- Ground truth comparison pipeline
- Dashboard design

**4. Results**
- Quantitative metrics (tables, figures)
- Error taxonomy with examples
- GradCAM visualizations
- User study findings

**5. Discussion**
- Key insights about model failures
- Limitations of current approach
- Implications for Tile2Net improvements

**6. Conclusion**
- Summary of contributions
- Future work

**7. Appendix**
- Code repository link
- Dashboard demo video
- Additional visualizations

#### 5.3 Demo Video (3-5 minutes)
**Outline:**
1. Problem introduction (30s)
2. Dashboard walkthrough (2min)
   - Load a tile
   - View predictions
   - Inspect errors with GradCAM
   - Filter and compare tiles
3. Key findings (1min)
   - Show park path false positive example
   - Explain GradCAM insight
4. Impact and future work (30s)

---

## Technical Stack

### Core Libraries
- **Model:** PyTorch + MscaleOCR (HRNet-W48)
- **XAI Methods:**
  - pytorch-grad-cam (GradCAM, LayerCAM, ScoreCAM)
  - Captum (Integrated Gradients)
  - saliency (XRAI)
- **Geospatial:** GeoPandas, Rasterio, Shapely
- **Ground Truth:** Requests (NYC Open Data API)
- **Visualization:** Matplotlib, PIL

### Presentation Notebook
- **Interactive Widgets:** ipywidgets (dropdowns, sliders, toggles)
- **Image Display:** IPython.display, PIL
- **Pre-computed Assets:** PNG files in `assets/` directory
- **Platform:** Jupyter Notebook / Google Colab / Kaggle

---

## Key Challenges & Mitigation

### Challenge 1: Coordinate System Alignment
**Issue:** Pixel coordinates → Geographic coordinates conversion errors.

**Solution:**
- Use Rasterio's `from_bounds()` for proper affine transforms
- Validate with known ground truth features
- Add visual overlay checks in notebook

### Challenge 2: Model Memory Usage
**Issue:** HRNet-W48 + GradCAM requires significant GPU memory.

**Solution:**
- Use single GPU, batch size 1
- Clear CUDA cache between XAI method calls
- Resize images to 256x256 for memory-intensive methods (ScoreCAM, IG, XRAI)

### Challenge 3: Ground Truth Limitations
**Issue:** NYC ground truth doesn't match model's learned concept.

**Solution:**
- Acknowledge in report as a finding, not a bug
- Use this mismatch as a key insight for presentation
- Explain difference between "visual sidewalk" vs "urban planning sidewalk"

### Challenge 4: No Compute Access for Presentation
**Issue:** Cannot run live inference during in-class presentation.

**Solution:**
- Pre-compute all visualizations on Kaggle/Colab
- Save as PNG files in `assets/` directory
- Load pre-rendered images in presentation notebook
- Use ipywidgets to switch between pre-computed results

---

## Timeline Recap

| Week       | Milestone                                    | Status      |
|------------|----------------------------------------------|-------------|
| Oct 24     | Submit proposal                              | ✅ Complete |
| Oct 28     | MVP: Tile2Net + GradCAM backend              | ✅ Complete |
| Nov 4      | Multi-layer GradCAM support                  | ✅ Complete |
| Nov 8      | Ground truth comparison pipeline             | ✅ Complete |
| Nov 28     | Extended XAI methods (LayerCAM, ScoreCAM, IG, XRAI, Confidence Maps) | ✅ Complete |
| Nov 28     | Full NYC pipeline + GradCAM on error regions | ✅ Complete |
| Dec 2-5    | **Interactive presentation notebook**        | 🔄 Next     |
| Dec 5-8    | **Pre-compute visualizations for examples**  | 📋 Upcoming |
| Dec 8-11   | **Final report + presentation prep**         | 📋 Upcoming |
| Dec 11     | **In-class presentation**                    | 📋 Upcoming |

> **Pivot:** Backend/dashboard development deprioritized. Focus on interactive Jupyter notebook with pre-computed visualizations for in-class presentation.

---

## Success Criteria

### Minimum Viable Product (Must Have)
- ✅ GradCAM working for all 4 classes (implemented in 02-tile2net.ipynb)
- ✅ Multiple XAI methods implemented (GradCAM, LayerCAM, ScoreCAM, IG, XRAI, Confidence Maps)
- 🔄 Ground truth comparison for at least 3-5 tiles (1/5 complete - Washington Square Park)
- 📋 Interactive presentation notebook with pre-computed visualizations
- ✅ Quantitative metrics (IoU, Precision, Recall, F1) - all implemented and calculated
- 📋 Final report with findings
- 📋 In-class presentation

### Stretch Goals (Nice to Have)
- ⬜ 10+ tile analysis across diverse contexts
- ⬜ Interactive ipywidgets for method/class selection
- ⬜ Animated GIF comparisons of XAI methods
- ✅ Comparison with other XAI methods (LayerCAM, ScoreCAM, Integrated Gradients, XRAI, Confidence Maps)

---

## References

1. **Tile2Net Paper:** Maryam Hosseini et al., "Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery," *Computers, Environment and Urban Systems*, 2023.

2. **HRNet:** Ke Sun et al., "Deep High-Resolution Representation Learning for Visual Recognition," CVPR 2019.

3. **GradCAM:** Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.

4. **NYC Open Data:** NYC Planimetric Database - Sidewalk, https://data.cityofnewyork.us/resource/52n9-sdep.geojson

---
