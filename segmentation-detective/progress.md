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
- ✅ Loaded pre-trained Tile2Net HRNet-W48 model
- ✅ Fixed configuration issues for inference
- ✅ Integrated pytorch-grad-cam for explainability
- ✅ Successfully ran inference on aerial imagery tiles

### 2. GradCAM Implementation
- ✅ Identified target layers across HRNet stages (Stage 2, 3, 4)
- ✅ Implemented GradCAM for all 4 segmentation classes:
  - Class 0: Sidewalk
  - Class 1: Road
  - Class 2: Crosswalk
  - Class 3: Background
- ✅ Visualized multi-scale feature attention across different HRNet branches
- ✅ Confirmed different stages focus on different features:
  - **Stage 2**: Fine edges and boundaries
  - **Stage 3**: Mid-level semantic features
  - **Stage 4**: High-level context and object recognition

### 3. Ground Truth Comparison Pipeline
- ✅ Generated NYC tile (Washington Square Park area)
- ✅ Downloaded NYC Planimetric sidewalk ground truth (15 polygons)
- ✅ Converted pixel predictions to georeferenced polygons
- ✅ Calculated quantitative metrics (IoU, Precision, Recall, F1)
- ✅ Identified and visualized error patterns (false positives/negatives)

### 4. Error Analysis with GradCAM
- ✅ Applied GradCAM specifically to error regions
- ✅ Explained model decision-making on false positives
- ✅ Discovered root cause of failures

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

### Phase 2: Backend Development (Week of Nov 11-18)

#### 2.1 Build FastAPI Backend
**Goal:** Create REST API for on-demand segmentation + GradCAM generation.

**Endpoints:**

**`POST /explain`**
- **Input:**
  - `file`: Uploaded tile image (PNG/JPG)
  - `target_class`: Class to explain (0-3)
  - `layer`: Which HRNet layer to visualize (e.g., "stage4-high")

- **Output:**
  ```json
  {
    "prediction": [[0, 1, 2, ...], ...],  // 2D array of class predictions
    "gradcam": "base64_encoded_heatmap_image",
    "class_distribution": {
      "Sidewalk": 0.259,
      "Road": 0.123,
      "Crosswalk": 0.016,
      "Background": 0.602
    },
    "metrics": {
      "iou": 0.049,
      "precision": 0.057,
      "recall": 0.262,
      "f1": 0.093
    }
  }
  ```

**`GET /tiles`**
- List available pre-generated tiles with metadata
- Return: `[{id, name, location, bbox, metrics}, ...]`

**`GET /layers`**
- List available HRNet layers for GradCAM
- Return: `["stage2-high", "stage2-low", "stage3-high", ...]`

**`POST /ground_truth`**
- **Input:** `bbox` (geographic bounds)
- **Output:** GeoJSON of NYC sidewalk ground truth in that area

**File Structure:**
```
segmentation-detective/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── models.py           # Model loading and caching
│   ├── inference.py        # Segmentation inference logic
│   ├── gradcam_utils.py    # GradCAM generation
│   ├── ground_truth.py     # NYC API integration
│   └── requirements.txt    # Dependencies
└── frontend/               # (Phase 3)
```

**Key Implementation Details:**
- Cache loaded model in memory (avoid reloading on each request)
- Use async/await for I/O operations (ground truth downloads)
- Add request validation (tile size limits, class bounds)
- Return CORS headers for frontend integration

#### 2.2 Optimize Model Loading
**Goal:** Reduce latency for interactive use.

**Optimizations:**
- Load model once at startup, keep in GPU memory
- Pre-load frequently used layers for GradCAM
- Implement request batching if multiple tiles requested
- Add health check endpoint for monitoring

---

### Phase 3: Interactive Dashboard (Week of Nov 18-25)

#### 3.1 Frontend Design
**Goal:** Build web interface for exploring segmentation errors.

**Framework:** React or Streamlit (recommend Streamlit for speed)

**Dashboard Components:**

**Component 1: Tile Explorer**
- Map view showing available tiles (using Leaflet.js)
- Click tile to load visualization
- Filter by error type, context, or metrics

**Component 2: Prediction Viewer**
- 4-panel layout:
  1. Original aerial image
  2. Segmentation prediction (color-coded)
  3. Ground truth overlay
  4. Error map (TP/FP/FN)

**Component 3: GradCAM Inspector**
- Dropdown to select HRNet layer
- Slider to adjust overlay opacity
- Side-by-side: Original | GradCAM | Overlay
- Show activation statistics (mean, max, distribution)

**Component 4: Metrics Dashboard**
- Summary statistics (IoU, Precision, Recall, F1)
- Class-wise breakdown
- Error type distribution (pie chart)
- Comparison across multiple tiles (bar chart)

**Component 5: Error Filter & Search**
- Filter tiles by:
  - Error type (park paths, shadows, occlusion)
  - Context (residential, commercial, park)
  - Performance (IoU range, F1 score)
- Search by location name or coordinates

**Component 6: Explanation Panel**
- Show why model made a decision
- List top activated features
- Compare correct vs incorrect predictions
- Suggest potential fixes

**Wireframe (ASCII):**
```
+--------------------------------------------------+
|  [Map View]  |  [Tile: Washington Square Park]  |
|              |  IoU: 0.049 | F1: 0.093          |
|  [Tile Grid] |  [Original] [Prediction] [Errors] |
|              |  [GradCAM Layer: Stage4-High ▼]  |
|              |  [Heatmap Visualization]          |
|              |  [Metrics] [Explanation Panel]    |
+--------------------------------------------------+
```

#### 3.2 Interactivity Features

**Click-to-Explain:**
- User clicks on any pixel in the image
- Dashboard highlights:
  - Predicted class at that location
  - Ground truth class (if available)
  - GradCAM heatmap centered on that pixel
  - Top-3 features contributing to decision

**Compare Mode:**
- Select 2+ tiles
- Show side-by-side comparison
- Highlight differences in error patterns

**Export Functionality:**
- Download current visualization as PNG
- Export metrics as CSV
- Generate PDF report with findings

---

### Phase 4: Evaluation & User Testing (Week of Nov 25 - Dec 2)

#### 4.1 Quantitative Evaluation

**Metric Collection:**
- Run pipeline on 50+ tiles across NYC
- Calculate aggregate statistics:
  - Mean/median IoU, Precision, Recall, F1
  - Standard deviation (measure consistency)
  - Per-class performance breakdown
  - Error type frequency distribution

**Comparative Analysis:**
- Compare performance across contexts:
  - Urban vs suburban
  - Day vs night imagery (if available)
  - Different seasons/years
  - Different NYC boroughs

**Attention-Error Correlation:**
- Measure correlation between:
  - GradCAM activation strength → False positive rate
  - Low activation → False negative rate
  - Activation uniformity → Prediction confidence

**Deliverable:**
- Table of results for final report
- Figures showing performance distributions
- Statistical tests (t-test, ANOVA) for context differences

#### 4.2 Qualitative User Study

**Participants:**
- 5-10 users from:
  - Urban planning students
  - GIS practitioners
  - Tile2Net developers
  - Domain experts (transportation engineers)

**Tasks:**
1. **Error Identification:** "Find 5 segmentation errors in this tile"
   - Measure: Time to find, accuracy

2. **Error Explanation:** "Why did the model make this mistake?"
   - With GradCAM vs without GradCAM
   - Measure: Explanation quality, confidence

3. **Dashboard Usability:** "Use the dashboard to find all park path false positives"
   - Measure: Task completion rate, time, user satisfaction

4. **Feature Feedback:** "What additional features would help?"
   - Collect qualitative feedback

**Metrics:**
- Task completion time
- Accuracy of error identification
- User satisfaction score (1-5 Likert scale)
- Qualitative feedback (open-ended)

**Deliverable:**
- User study report
- Dashboard improvements based on feedback

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

## Technical Stack Recommendations

### Backend
- **Framework:** FastAPI (async, fast, auto-docs)
- **Model:** PyTorch + pytorch-grad-cam
- **Geospatial:** GeoPandas, Rasterio, Shapely
- **Ground Truth:** Requests (NYC Open Data API)

### Frontend
- **Option 1 (Quick):** Streamlit
  - Pros: Fast development, built-in widgets
  - Cons: Less customizable, slower for large images

- **Option 2 (Robust):** React + Leaflet + Plotly
  - Pros: Full control, better performance, interactive maps
  - Cons: Longer development time

### Deployment
- **Local Demo:** Docker Compose (backend + frontend)
- **Cloud (Optional):**
  - Backend: AWS Lambda or Google Cloud Run
  - Frontend: Vercel or Netlify
  - Storage: S3 for tiles and results

---

## Key Challenges & Mitigation

### Challenge 1: Coordinate System Alignment
**Issue:** Pixel coordinates → Geographic coordinates conversion errors.

**Solution:**
- Use Rasterio's `from_bounds()` for proper affine transforms
- Validate with known ground truth features
- Add visual overlay checks in dashboard

### Challenge 2: Model Memory Usage
**Issue:** HRNet-W48 + GradCAM requires significant GPU memory.

**Solution:**
- Use single GPU, batch size 1
- Clear CUDA cache between requests
- Implement model quantization if needed (FP16)

### Challenge 3: Ground Truth Limitations
**Issue:** NYC ground truth doesn't match model's learned concept.

**Solution:**
- Acknowledge in report as a finding, not a bug
- Consider secondary validation with OpenStreetMap
- Create custom annotations for key error examples

### Challenge 4: Real-time Interactivity
**Issue:** GradCAM generation takes 2-5 seconds per tile.

**Solution:**
- Pre-generate GradCAM for common layers
- Show loading spinner in UI
- Cache results for previously viewed tiles

---

## Timeline Recap

| Week       | Milestone                                    | Status      |
|------------|----------------------------------------------|-------------|
| Oct 24     | Submit proposal                              | ✅ Complete |
| Oct 28     | MVP: Tile2Net + GradCAM backend              | ✅ Complete |
| Nov 4      | Multi-layer GradCAM support                  | ✅ Complete |
| Nov 8      | Ground truth comparison pipeline             | ✅ Complete |
| Nov 11     | **Expand to 10+ tiles, error taxonomy**      | 🔄 Next     |
| Nov 18     | **FastAPI backend + dashboard prototype**    | 📋 Upcoming |
| Nov 25     | **User evaluation, collect feedback**        | 📋 Upcoming |
| Dec 2      | **Prepare final report figures**             | 📋 Upcoming |
| Dec 11     | **Submit final report & demo video**         | 📋 Upcoming |

---

## Success Criteria

### Minimum Viable Product (Must Have)
- ✅ GradCAM working for all 4 classes
- ✅ Ground truth comparison for at least 5 tiles
- ✅ Dashboard showing predictions + errors + explanations
- ✅ Quantitative metrics (IoU, Precision, Recall, F1)
- ✅ Final report with findings

### Stretch Goals (Nice to Have)
- ⬜ 50+ tile analysis with statistical significance
- ⬜ User study with 10+ participants
- ⬜ Deployed web dashboard (public URL)
- ⬜ Interactive error annotation tool
- ⬜ Comparison with other XAI methods (LayerCAM, ScoreCAM)

---

## References

1. **Tile2Net Paper:** Maryam Hosseini et al., "Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery," *Computers, Environment and Urban Systems*, 2023.

2. **HRNet:** Ke Sun et al., "Deep High-Resolution Representation Learning for Visual Recognition," CVPR 2019.

3. **GradCAM:** Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.

4. **NYC Open Data:** NYC Planimetric Database - Sidewalk, https://data.cityofnewyork.us/resource/52n9-sdep.geojson

---
