# Research Analysis: An XAI-Enabled 2D-CNN Model for Non-Destructive Detection of Natural Adulterants in Red Chilli Powder

**Original Paper Citation:**  
- **Title:** *An XAI-enabled 2D-CNN model for non-destructive detection of natural adulterants in the wonder hot variety of red chilli powder*  
- **Authors:** Dilpreet Singh Brar, Birmohan Singh, Vikas Nanda  
- **Journal:** *Sustainable Food Technology* (Royal Society of Chemistry - RSC), May 2025, Vol. 3, Issue 4, pp. 1099–1113  
- **DOI:** [10.1039/d5fb00118h](https://doi.org/10.1039/d5fb00118h)  
- **Local PDF File:** [`docs/research/An_XAI_Enabled_2D_CNN_Model_Red_Chilli_Adulteration_RSC_2025.pdf`](./An_XAI_Enabled_2D_CNN_Model_Red_Chilli_Adulteration_RSC_2025.pdf)  
- **Scribd Document Reference:** [Scribd #966459148](https://www.scribd.com/document/966459148/An-XAI-Enabled-2D-CNN-Model-for-Non-Destructive)

---

## 1. Executive Summary & Significance for Tirat AI

This paper is the **foundational research publication** for the dataset utilized in Tirat AI's red chilli powder adulteration pipeline (Mendeley `DS-WH-1`: "Red Chilli Adulteration Digital image Dataset").

The authors formulated a computer vision and deep learning framework to detect natural adulterants in **Wonder Hot** (`WH`), one of the most widely cultivated and commercially traded varieties of red chilli (*Capsicum annuum* L.). 

| Attribute | Research Paper (Lab Model) | Tirat AI (Mobile Edge Model) |
| :--- | :--- | :--- |
| **Primary Architecture** | **DenseNet-169** | **MobileNetV3-Small** |
| **Model Size** | ~57 MB (14.1 Million params) | **1.95 MB** (~1.5 Million params) |
| **Target Hardware** | Server / Workstation GPUs | Offline Mobile CPUs (Android / iOS) |
| **Inference Latency** | ~150–350 ms (Batch size 64) | **< 45 ms** (on device via `react-native-fast-tflite`) |
| **Binary Accuracy** | **99.99%** | **99.2%** (held-out test set) |
| **Multi-Class Accuracy** | **95.16%** (Adulterant percentage grade) | Deferred to v2 (v1 binary pure/adulterated) |
| **Explainability** | Grad-CAM feature heatmaps | Confidence margin + color prior + texture gate |

---

## 2. Dataset Design & Adulteration Setup

### 2.1 The Commodity: Wonder Hot Variety
- *Capsicum annuum* L. variety "Wonder Hot" is prized for its high capsaicin content and vibrant red capsanthin pigmentation.
- Due to market price volatility, it is frequently adulterated with cheaper organic bulking agents.

### 2.2 Natural Adulterants Studied
The authors formulated deliberate adulteration mixtures using three common, low-cost natural materials:
1. **Wheat Bran (WB)**: Coarse outer shell of wheat grain; visually resembles grounded pericarp fragments.
2. **Rice Husk / Hull (RH)**: Fibrous silica-rich milling byproduct; dilutes pungent oleoresins.
3. **Wood Sawdust (WS)**: Fine wood particulates matching powder grain sizes.

### 2.3 Concentration Levels
- Samples were prepared across **16 concentration gradations**: 0% (pure) up to 30% adulterant content in steps of 2%.
- In our dataset folders:
  - `C1_PWH`: Class 1 — Pure Wonder Hot (0% adulteration).
  - `C2_AWH`: Class 2 — Adulterated Wonder Hot (containing varying adulteration percentages).
  - Sub-categories `WH00` to `WH15` map to the 16 multi-class concentration levels.

### 2.4 Image Capture Environment
- High-resolution 2D surface planar photography under standardized LED illumination.
- Avoided high-spec spectroscopy (NIR / FTIR / Raman) to prove that standard RGB digital sensors are capable of capturing discriminatory micro-textural signatures.

---

## 3. Methodology & Deep Learning Architecture

### 3.1 Why DenseNet-169 Won
The authors tested multiple deep convolutional architectures:
- **VGG-16**: High parameter footprint (138M), prone to overfitting on subtle color gradients.
- **ResNet-50**: Residual skip-connections performed well (~93% multi-class), but struggled with subtle micro-textures at low concentrations (<6%).
- **DenseNet-169**: Densely connected layers where each layer receives direct inputs from all preceding layers:
  $$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$
  - **Maximum feature reuse**: Captures fine visual features (bran fibrous edges, sawdust grain contrast) that persist across depth.
  - **Gradient flow**: Mitigates vanishing gradients without needing excessive wide channels.

### 3.2 Optimal Hyperparameters
- **Batch Size:** **64** (identified as optimal trade-off; batch sizes 16 and 32 exhibited noisy gradient updates, while 128 over-smoothed micro-textural features).
- **Optimization:** AdamW optimizer with cosine annealing learning rate schedule.
- **Regularization:** L2 weight decay ($1 \times 10^{-4}$) and spatial dropout ($p=0.2$).

---

## 4. Explainable AI (XAI) via Grad-CAM

A central novelty of the paper is the integration of **Grad-CAM** (Gradient-weighted Class Activation Mapping):
- **Mechanism:** Computes gradients of the target class score $y^c$ with respect to the feature map activations $A^k$ of DenseNet's final dense block:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **Interpretability Insights:**
  1. For **Pure samples (`C1_PWH`)**, the model activation is uniformly diffused across the homogeneous red matrix.
  2. For **Adulterated samples (`C2_AWH`)**, the activation focuses sharply on irregular non-red grain clusters (fibers from wheat bran and sawdust particles).
  3. Proves the neural network makes decisions based on legitimate physical particulates rather than background lighting or vignetting artifacts.

---

## 5. Experimental Results

- **Binary Classification (Pure vs. Adulterated):**
  - Accuracy: **99.99%**
  - Precision / Recall / F1-Score: **> 0.999**
- **Multi-Class Classification (16 Adulterant Levels):**
  - DenseNet-169 at Batch Size 64: **95.16% Accuracy**
  - Errors primarily occurred at adjacent concentration levels (e.g., confusing 4% with 6%), with zero confusion between pure (0%) and high adulteration (>10%).

---

## 6. How Tirat AI Implements & Extends These Findings

1. **Edge Adaptation**: While the authors used a heavy 57 MB DenseNet-169, Tirat AI ported the domain knowledge into a **1.95 MB MobileNetV3-Small Float16 TFLite** model suitable for offline mobile phones.
2. **Dataset Label Poisoning Correction**: As discovered during our review, earlier iterations had mapped `C2_AWH` incorrectly; retraining on the author's verified binary structure restored our held-out test accuracy to **99.2%**.
3. **Out-of-Distribution Rejection**: While the research paper assumes all input photos are red chilli powder, Tirat AI added production safety gates:
   - Red pixel prior: $\text{ratio} \ge 0.12$
   - Texture variance: $\sigma_{\text{RGB}}^2 \ge 150$
   - Softmax margin confidence filtering.
