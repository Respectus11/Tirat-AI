# Research Analysis: An XAI-Enabled 2D-CNN Model for Red Chilli Adulteration

**Original Paper Citation:**  
- **Title:** *An XAI-enabled 2D-CNN model for non-destructive detection of natural adulterants in the wonder hot variety of red chilli powder*  
- **Authors:** Dilpreet Singh Brar, Birmohan Singh, Vikas Nanda (Sant Longowal Institute of Engineering & Technology - SLIET)  
- **Journal:** *Sustainable Food Technology* (Royal Society of Chemistry - RSC), May 2025, Vol. 3, Issue 4, pp. 1099–1113  
- **Full Text PDF:** [`docs/research/An_XAI_Enabled_2D_CNN_Model_Red_Chilli_Adulteration_RSC_2025.pdf`](./An_XAI_Enabled_2D_CNN_Model_Red_Chilli_Adulteration_RSC_2025.pdf) (Rendered directly from complete ScienceDirect publication)

---

## 1. Key Insights & Ground Truth

### 1.1 The Commodity & Adulteration Design
The study investigated the **Wonder Hot (WH)** variety of red chilli (*Capsicum annuum* L.), prized for high capsaicin content and commercial demand, making it a prime target for illegal bulking.

The authors formulated a **16-class dataset** consisting of:
1. **Pure Red Chilli (`WH00`)**: 0% adulteration.
2. **5 Natural Bulking Adulterants** at **3 Concentration Levels (5%, 10%, 15%)**:
   - **Wheat Bran (`WHBM`)**: Coarse outer grain husk; mimics ground pericarp flakes.
   - **Rice Hull / Husk (`WHRH`)**: Fibrous silica-dense milling byproduct.
   - **Wood Sawdust (`WHWS`)**: Fine wood particulate matching ground powder particle sizing.
   - **Brick Powder (`WHBP`)**: Inert mineral powder adding weight and red tint.
   - **Gram Meal (`WHGM`)**: Chickpea/pulse flour diluting pungency and color.

*(5 adulterants × 3 levels = 15 adulterated classes + 1 pure class = **16 classes total**).*

### 1.2 Sample Preparation & Optical Acquisition
- **Physical homogenization**: Blended in a planetary mixer for 10 minutes, sieved through British Standard Sieve (BSS) No. 30 (particle size uniformity), stored in airtight glass containers under refrigeration.
- **Image acquisition**: Planar digital camera capture under standardized diffused illumination. High-resolution captures ($1728 \times 2592 \times 3$) taken from 4 rotated angles to capture multi-perspective grain reflections, resized to $224 \times 224$ with $[0, 1]$ normalization.

---

## 2. Model Architecture & Training Dynamics

### 2.1 DenseNet (121 vs 169)
- **Why DenseNet won**: Direct layer-to-layer feature map concatenation ($\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \dots, \mathbf{x}_{\ell-1}])$) ensures micro-textural signatures (particulate edges, fiber boundaries) persist throughout deep layers without vanishing gradients.
- **DenseNet-169** (13.5M params, 338 layers) achieved greater stability and capacity than DenseNet-121 (7.6M params, 242 layers).

### 2.2 Optimizer: AdamClr (Cyclical Learning Rate)
- Utilized **Adam with Cyclical Learning Rate (AdamClr)**:
  - $\text{LR}_{\min} = 5 \times 10^{-5}$, $\text{LR}_{\max} = 1 \times 10^{-3}$
  - Step size: $25 \times (\text{train\_size} / \text{batch\_size})$
  - Periodic cyclical learning rate sweeps prevented the model from getting trapped in shallow local minima.

### 2.3 Batch Size Dynamics
- **Binary Classification (Pure vs. Adulterated)**:
  - Both DenseNet-121 and DenseNet-169 achieved **99.99% accuracy**, **1.00 ROC-AUC** across all batch sizes (16, 32, 64).
  - Smaller batch sizes (16) produced smoother gradient convergence in binary mode.
- **Multi-Class Classification (16 Concentration Levels)**:
  - **Batch Size 64** was the clear winner, achieving **95.16% accuracy** (vs 89.98% at BS 16).
  - Multi-class decision boundaries require larger batch normalization statistics to stabilize complex feature hierarchies across subtle concentration differences (e.g., 5% vs 10%).

---

## 3. Explainable AI (Grad-CAM & LIME) Validation

Grad-CAM was applied to the final convolutional layer feature maps:
1. **Pure Samples (`WH00`)**: Heatmap activation is uniformly diffused and low across the homogeneous red matrix.
2. **Adulterated Samples (`WHBM`, `WHWS`, etc.)**: Heatmaps exhibit localized high-intensity hotspots (red/yellow) directly pinpointing foreign particulate clusters.
3. **Concentration Scaling**: As adulteration increased from 5% to 15%, the spatial density and intensity of hotspot activations scaled proportionally.
4. Proves the model learns **authentic particulate morphology and granule edges**, not background illumination or lens vignetting.

---

## 4. How Tirat AI Implements & Extends These Findings

| Factor | RSC Research Paper (Lab Setup) | Tirat AI (Mobile Edge Implementation) |
| :--- | :--- | :--- |
| **Hardware** | Dual Intel Xeon + Nvidia A6000 (48GB GPU) | Offline Mobile Smartphones (Android / iOS) |
| **Model Footprint** | DenseNet-169 (~57 MB, 13.5M params) | MobileNetV3-Small Float16 (~1.95 MB, 939K params) |
| **Inference Time** | Server GPU batch execution | < 45 ms real-time on-device via `react-native-fast-tflite` |
| **Feature Extraction** | Deep dense layer concatenation | Multi-scale intermediate + final GAP fusion (`train_redchili.py`) |
| **Unconstrained Environments** | Standardized box LED lighting & planar glass | Real-world camera gates: edge sharpness ($S \ge 2.4$), color prior ($R \ge 0.12$), and texture variance ($\sigma^2 \ge 150$) |
| **Field Guidance** | Lab camera mount | Interactive UI reticle, 90° flatness indicator, and powder fill guidance |
