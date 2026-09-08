"""
generate_presentation.py — Generates a modern, attractive, professional PowerPoint presentation
documenting the complete Tirat AI project work, architecture, dataset rebalancing, training dynamics,
evaluation benchmarks, edge optimization, and mobile deployment.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# Color Palette & Typography
# -----------------------------------------------------------------------------
BG_DARK = RGBColor(11, 15, 23)        # #0B0F17 Deep Slate Background
CARD_BG = RGBColor(22, 30, 46)        # #161E2E Surface Card
CARD_BORDER = RGBColor(38, 52, 78)    # #26344E Card Border
TEXT_LIGHT = RGBColor(248, 250, 252)  # #F8FAFC White/Light Slate
TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8 Gray/Muted Slate
ACCENT_RED = RGBColor(239, 68, 68)    # #EF4444 Chili Red
ACCENT_GREEN = RGBColor(16, 185, 129) # #10B981 Emerald Green
ACCENT_AMBER = RGBColor(245, 158, 11) # #F59E0B Amber Gold
ACCENT_CYAN = RGBColor(6, 182, 212)   # #06B6D4 Tech Cyan
FONT_HEADING = "Calibri"
FONT_BODY = "Calibri"

ROOT_DIR = Path(__file__).resolve().parent.parent
CHILI_OUTPUTS = ROOT_DIR / "model_training" / "redchili" / "outputs"
CURVES_IMG = CHILI_OUTPUTS / "training_curves.png"
CONF_MAT_IMG = CHILI_OUTPUTS / "confusion_matrix.png"
APP_ICON = ROOT_DIR / "app" / "assets" / "icon.png"

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_slide_base(title: str, category: str, subtitle: str = ""):
        slide = prs.slides.add_slide(blank_layout)

        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()

        # Top Accent Line
        top_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.04))
        top_line.fill.solid()
        top_line.fill.fore_color.rgb = ACCENT_AMBER
        top_line.line.fill.background()

        # Category Badge / Breadcrumb
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(8), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.name = FONT_HEADING
        p_cat.font.color.rgb = ACCENT_CYAN

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.6))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.name = FONT_HEADING
        p_t.font.color.rgb = TEXT_LIGHT

        # Subtitle
        if subtitle:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED
            p_sub.font.name = FONT_BODY

        # Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.3))
        tf_foot = footer_box.text_frame
        p_foot = tf_foot.paragraphs[0]
        p_foot.text = "Tirat AI (ጥራት) — Edge AI Food Adulteration Detection"
        p_foot.font.size = Pt(9)
        p_foot.font.color.rgb = TEXT_MUTED

        return slide

    def add_card(slide, left, top, width, height, title="", title_color=ACCENT_AMBER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.font.name = FONT_HEADING
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Hero)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = BG_DARK
    bg1.line.fill.background()

    # Brand Pill
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.2), Inches(3.2), Inches(0.4))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(24, 33, 52)
    pill.line.color.rgb = ACCENT_AMBER
    pill.line.width = Pt(1)
    tf_pill = pill.text_frame
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = "OFFLINE-FIRST EDGE AI SYSTEM"
    p_pill.font.size = Pt(10)
    p_pill.font.bold = True
    p_pill.font.color.rgb = ACCENT_AMBER
    p_pill.alignment = PP_ALIGN.CENTER

    # Main Hero Title
    hero_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.0), Inches(1.8))
    tf_hero = hero_box.text_frame
    tf_hero.word_wrap = True
    p_hero1 = tf_hero.paragraphs[0]
    p_hero1.text = "Tirat AI (ጥራት)"
    p_hero1.font.size = Pt(44)
    p_hero1.font.bold = True
    p_hero1.font.color.rgb = TEXT_LIGHT

    p_hero2 = tf_hero.add_paragraph()
    p_hero2.text = "Real-Time Food Adulteration Detection on Mobile Edge"
    p_hero2.font.size = Pt(22)
    p_hero2.font.color.rgb = ACCENT_CYAN

    desc_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.6), Inches(10.5), Inches(0.9))
    tf_desc = desc_box.text_frame
    tf_desc.word_wrap = True
    p_desc = tf_desc.paragraphs[0]
    p_desc.text = (
        "An end-to-end computer vision platform designed for offline detection of adulteration in staple foods "
        "(Red Chili Powder and Teff Flour) directly on low-cost smartphones using optimized MobileNetV3 and TFLite."
    )
    p_desc.font.size = Pt(14)
    p_desc.font.color.rgb = TEXT_MUTED

    # 4 Key Metrics Banners
    metrics = [
        ("96.8%", "Pure Recall", "Jumped from 0% after rebalancing", ACCENT_GREEN),
        ("95.2%", "Test Accuracy", "On balanced held-out test split", ACCENT_CYAN),
        ("1.95 MB", "Model Size", "Float16 TFLite for fast edge loading", ACCENT_AMBER),
        ("< 45 ms", "Inference Latency", "Offline native C++ execution", ACCENT_RED),
    ]
    card_w = Inches(2.6)
    card_gap = Inches(0.4)
    for i, (val, title, sub, color) in enumerate(metrics):
        cx = Inches(1.0) + i * (card_w + card_gap)
        cy = Inches(4.8)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, card_w, Inches(1.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        tb = s1.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.15), card_w - Inches(0.3), Inches(1.5))
        tf = tb.text_frame
        p1 = tf.paragraphs[0]
        p1.text = val
        p1.font.size = Pt(28)
        p1.font.bold = True
        p1.font.color.rgb = color

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_LIGHT

        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    s2 = add_slide_base(
        title="Food Adulteration: The Challenge & Motivation",
        category="Problem Context",
        subtitle="Addressing hazardous food fraud in everyday staples without requiring expensive laboratory hardware."
    )

    add_card(s2, Inches(0.8), Inches(1.9), Inches(3.6), Inches(4.8), "The Fraud Reality", ACCENT_RED)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets = [
        ("Hazardous Fillers: ", "Chili powder adulterated with toxic brick powder, sawdust, artificial coal-tar dyes, and low-grade pepper."),
        ("Teff Adulteration: ", "Ethiopia's primary grain (Teff) frequently diluted with cheap starches, clay, and wheat husks."),
        ("Severe Health Risks: ", "Chronic ingestion leads to gastrointestinal damage, organ toxicity, and carcinogenic exposure."),
        ("Economic Impact: ", "Smallholder farmers and honest merchants lose market value to fraudulent distributors.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    add_card(s2, Inches(4.8), Inches(1.9), Inches(3.6), Inches(4.8), "Why Conventional Tests Fail", ACCENT_AMBER)
    tb = s2.shapes.add_textbox(Inches(5.0), Inches(2.5), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets2 = [
        ("Prohibitive Equipment: ", "HPLC, mass spectrometry, and chemical chromatography cost $20k-$80k per testing setup."),
        ("Multi-Day Turnaround: ", "Samples must be shipped to centralized laboratories, taking 3-7 days for report issuance."),
        ("No Market Accessibility: ", "Zero instant feedback for consumers and food inspectors in local open-air markets."),
        ("High Reagent Costs: ", "Single-use chemical reagents are recurring expenses out of reach for rural communities.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets2):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    add_card(s2, Inches(8.8), Inches(1.9), Inches(3.7), Inches(4.8), "The Tirat Edge AI Solution", ACCENT_GREEN)
    tb = s2.shapes.add_textbox(Inches(9.0), Inches(2.5), Inches(3.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets3 = [
        ("100% Offline Inference: ", "Runs locally on Android hardware via TensorFlow Lite; zero cellular data or cloud dependency."),
        ("Instant Results (<50ms): ", "Provides immediate binary purity verdict and confidence breakdown upon photo capture."),
        ("Multi-Layer Guardrails: ", "Built-in texture variance and color ratio gates reject walls, hands, or invalid surfaces."),
        ("Accessible & Bilingual: ", "Full Amharic (አማርኛ) and English localization tailored for Ethiopian and regional markets.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets3):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: System Architecture
    # =========================================================================
    s3 = add_slide_base(
        title="Full-Stack Edge Architecture",
        category="System Architecture",
        subtitle="End-to-end integration across React Native frontend, native C++ TFLite engine, and local SQLite persistence."
    )

    steps = [
        ("1. Camera & Capture", "Expo Camera / Native frame grabber. Image captured at native resolution with focus guidance overlay.", ACCENT_CYAN),
        ("2. OOD & Texture Gate", "Fast on-device statistical validation. Checks color ratio (Red/Brown) and pixel variance to reject invalid non-food objects.", ACCENT_AMBER),
        ("3. Preprocessing Engine", "Preserves aspect ratio, center crops to 224x224 RGB. Keeps raw [0..255] float tensors (model embeds normalization).", ACCENT_CYAN),
        ("4. Native TFLite Engine", "C++ Nitro runtime (react-native-fast-tflite). Executes Float16 MobileNetV3 model in ~35ms with zero JNI overhead.", ACCENT_GREEN),
        ("5. UX & Local DB", "Instant verdict meter, confidence gauge, and auto-logging to local SQLite database with async batch sync queue.", ACCENT_AMBER)
    ]
    step_w = Inches(2.15)
    step_gap = Inches(0.24)
    for i, (title, desc, color) in enumerate(steps):
        sx = Inches(0.8) + i * (step_w + step_gap)
        sy = Inches(1.9)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, sx, sy, step_w, Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        tb = s3.shapes.add_textbox(sx + Inches(0.15), sy + Inches(0.2), step_w - Inches(0.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = color

        p2 = tf.add_paragraph()
        p2.space_before = Pt(12)
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 4: Red Chili Diagnosis & Root Causes
    # =========================================================================
    s4 = add_slide_base(
        title="Red Chili Adulteration Model: Diagnosis & Root Causes",
        category="Deep-Dive Investigation",
        subtitle="Investigating why the initial model classified 100% of samples (including pure training data) as 'Adulterated'."
    )

    add_card(s4, Inches(0.8), Inches(1.9), Inches(3.6), Inches(4.8), "Cause 1: 10:1 Class Imbalance", ACCENT_RED)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    c1_points = [
        "Dataset Anatomy: Mendeley DS-WH-1 consists of DS-I (binary) and DS-II (16-class quantification).",
        "The Imbalance: DS-I + DS-II yielded 4,735 Adulterated images against only 492 Pure images.",
        "Model Prior Bias: The cross-entropy loss heavily penalized missing adulterated cases, converging to an overwhelming prior probability of ~91% Adulterated."
    ]
    for i, pt in enumerate(c1_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    add_card(s4, Inches(4.8), Inches(1.9), Inches(3.6), Inches(4.8), "Cause 2: Double Normalization", ACCENT_AMBER)
    tb = s4.shapes.add_textbox(Inches(5.0), Inches(2.5), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    c2_points = [
        "Graph Layer: train_redchili.py explicitly embedded tf.keras.layers.Rescaling(1/127.5, offset=-1.0).",
        "The Discrepancy: External test scripts and testing harnesses pre-divided incoming images by 255.0 (img / 255.0 => [0..1.0]).",
        "Tensor Collapse: Re-scaling [0..1.0] by 1/127.5 collapsed inputs to ~[-1.0, -0.99] (black image), forcing the network into default majority-class fallback."
    ]
    for i, pt in enumerate(c2_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    add_card(s4, Inches(8.8), Inches(1.9), Inches(3.7), Inches(4.8), "Cause 3: Omitted Pure Data", ACCENT_CYAN)
    tb = s4.shapes.add_textbox(Inches(9.0), Inches(2.5), Inches(3.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    c3_points = [
        "Data Audit: Discovered 124 pristine pure chili images tucked away inside C1_PWH/Test and WH00/Test directories.",
        "Unlocking Balance: Recovering these test splits expanded total pure data from 492 to 616 images.",
        "The Solution: Reconstructed an exact 1:1 balanced dataset (616 Pure vs 616 Adulterated) stratified across all 16 adulterant types."
    ]
    for i, pt in enumerate(c3_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: Rebalancing & Dataset Engineering
    # =========================================================================
    s5 = add_slide_base(
        title="Dataset Engineering & Stratified Rebalancing",
        category="Data Pipeline",
        subtitle="Building a rigorous 1:1 balanced distribution across 16 adulterants to eliminate bias and ensure generalization."
    )

    add_card(s5, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.8), "Stratified Sampling Strategy", ACCENT_AMBER)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    strat_points = [
        ("Total Pure Samples (616): ", "308 from C1_PWH (binary pure set) + 308 from WH00 (0% adulteration set)."),
        ("Total Adulterated Samples (616): ", "Uniformly sampled across 16 sub-categories (38-39 images each) representing Sawdust, Brick Powder, Wheat Bran, Rice Husk, and Spent Chili at 5%, 10%, and 15% concentrations."),
        ("Strict 80 / 10 / 10 Split: ", "Training: 985 images (492 pure, 493 adulterated)\nValidation: 123 images (62 pure, 61 adulterated)\nTest Split: 124 images (62 pure, 62 adulterated)"),
        ("Phone-Camera Augmentation: ", "Simulates handheld mobile use with random rotation (±15°), zoom (0.9-1.1x), brightness jitter (±10%), and horizontal flips.")
    ]
    for i, (b_title, b_desc) in enumerate(strat_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    add_card(s5, Inches(6.8), Inches(1.9), Inches(5.7), Inches(4.8), "Dataset Distribution Breakdown", ACCENT_CYAN)
    
    # Table inside card
    rows, cols = 5, 4
    table_shape = s5.shapes.add_table(rows, cols, Inches(7.1), Inches(2.6), Inches(5.1), Inches(2.2))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.8)
    tbl.columns[1].width = Inches(1.1)
    tbl.columns[2].width = Inches(1.1)
    tbl.columns[3].width = Inches(1.1)

    headers = ["Dataset Split", "Pure", "Adulterated", "Total"]
    data = [
        ["Train (80%)", "492", "493", "985"],
        ["Validation (10%)", "62", "61", "123"],
        ["Test (Held-Out 10%)", "62", "62", "124"],
        ["Total Dataset", "616", "616", "1,232"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = ACCENT_AMBER

    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            cell = tbl.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 26, 40)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_LIGHT if col_idx == 0 else ACCENT_CYAN

    # Highlight note below table
    note_box = s5.shapes.add_textbox(Inches(7.1), Inches(5.0), Inches(5.1), Inches(1.5))
    tf_note = note_box.text_frame
    tf_note.word_wrap = True
    p = tf_note.paragraphs[0]
    p.text = "Key Takeaway: Balanced 1.0/1.0 Class Weights"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_GREEN

    p2 = tf_note.add_paragraph()
    p2.space_before = Pt(6)
    p2.text = (
        "Rebalancing the dataset completely eliminated the prior bias, enabling the neural network "
        "to discover actual morphological features of pure chili rather than taking a statistical shortcut."
    )
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 6: Model Architecture & Training Methodology
    # =========================================================================
    s6 = add_slide_base(
        title="Model Architecture & Two-Phase Transfer Learning",
        category="Training Strategy",
        subtitle="Deploying MobileNetV3-Small with progressive unfreezing and embedded graph normalization."
    )

    m_cards = [
        ("MobileNetV3-Small Backbone", "Selected for lowest FLOP count, minimal RAM usage (<15 MB during inference), and high latency efficiency on ARM Cortex cores. Pretrained on ImageNet-1k.", ACCENT_CYAN),
        ("In-Graph Normalization", "The Rescaling(1/127.5, offset=-1.0) layer is frozen directly in the computation graph. The mobile app passes raw 0..255 floats, completely immune to train-serve normalization skew.", ACCENT_AMBER),
        ("Phase 1: Feature Extraction", "10 Epochs with backbone frozen. Trains custom dense head (Dropout 0.3 + Dense 128 + Softmax 2) using Adam optimizer at lr=1e-3. Reached 94.3% validation accuracy.", ACCENT_GREEN),
        ("Phase 2: Fine-Tuning", "15 Epochs unfreezing the top 30 convolutional layers. Uses fine learning rate (1e-4) with Cosine Decay and Warmup. Pushed validation accuracy to 96.75%.", ACCENT_CYAN)
    ]
    c_w = Inches(2.7)
    c_gap = Inches(0.3)
    for i, (head, body, col) in enumerate(m_cards):
        cx = Inches(0.8) + i * (c_w + c_gap)
        cy = Inches(1.9)
        card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, c_w, Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)

        tb = s6.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.2), c_w - Inches(0.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.space_before = Pt(12)
        p2.text = body
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 7: Training Dynamics & Curves
    # =========================================================================
    s7 = add_slide_base(
        title="Empirical Training Dynamics & Convergence",
        category="Model Training Results",
        subtitle="Tracking loss minimization and accuracy progression across Phase 1 (frozen) and Phase 2 (fine-tuning)."
    )

    # Embed training_curves.png if available
    if CURVES_IMG.exists():
        s7.shapes.add_picture(str(CURVES_IMG), Inches(0.8), Inches(1.9), width=Inches(7.2))
    else:
        add_card(s7, Inches(0.8), Inches(1.9), Inches(7.2), Inches(4.8), "Training Dynamics Plot", ACCENT_AMBER)

    add_card(s7, Inches(8.3), Inches(1.9), Inches(4.2), Inches(4.8), "Training Insights", ACCENT_GREEN)
    tb = s7.shapes.add_textbox(Inches(8.5), Inches(2.5), Inches(3.8), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    t_points = [
        ("Phase 1 Convergence: ", "Head adaptation occurred rapidly within 4 epochs, reaching 94.3% validation accuracy and val_loss of 0.22."),
        ("Phase 2 Refinement: ", "Unfreezing top layers allowed feature extractors to capture subtle grain texture differences between chili seeds and sawdust/husk particles."),
        ("Peak Val Accuracy: ", "Achieved 96.75% validation accuracy with validation loss dropping to 0.0999."),
        ("Overfitting Control: ", "Dropout (0.3) combined with phone-camera data augmentations prevented memorization on small sample subsets.")
    ]
    for i, (b_title, b_desc) in enumerate(t_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 8: Evaluation & Confusion Matrix
    # =========================================================================
    s8 = add_slide_base(
        title="Held-Out Test Benchmarks & Confusion Matrix",
        category="Evaluation & Validation",
        subtitle="Rigorous evaluation on 124 held-out test images demonstrating 96.8% recall on pure chili samples."
    )

    # Embed confusion_matrix.png if available
    if CONF_MAT_IMG.exists():
        s8.shapes.add_picture(str(CONF_MAT_IMG), Inches(0.8), Inches(1.9), width=Inches(5.6))
    else:
        add_card(s8, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.8), "Confusion Matrix", ACCENT_CYAN)

    add_card(s8, Inches(6.7), Inches(1.9), Inches(5.8), Inches(4.8), "Classification Report (Test Set)", ACCENT_CYAN)
    
    # Table of metrics
    rows, cols = 4, 5
    table_shape = s8.shapes.add_table(rows, cols, Inches(6.9), Inches(2.5), Inches(5.4), Inches(2.0))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.6)
    tbl.columns[1].width = Inches(0.95)
    tbl.columns[2].width = Inches(0.95)
    tbl.columns[3].width = Inches(0.95)
    tbl.columns[4].width = Inches(0.95)

    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    c_data = [
        ["Pure Chili", "93.8%", "96.8%", "95.2%", "62"],
        ["Adulterated", "96.7%", "93.5%", "95.1%", "62"],
        ["Macro Avg", "95.2%", "95.2%", "95.2%", "124"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = ACCENT_AMBER

    for row_idx, row in enumerate(c_data, start=1):
        for col_idx, val in enumerate(row):
            cell = tbl.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 26, 40)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_LIGHT if col_idx == 0 else (ACCENT_GREEN if val == "96.8%" else ACCENT_CYAN)

    tb = s8.shapes.add_textbox(Inches(6.9), Inches(4.7), Inches(5.4), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    eval_bullets = [
        ("Massive Recall Leap: ", "Pure recall surged from ~0% (prior majority collapse) to 96.8% (60 of 62 pure test samples correctly identified)."),
        ("Minimal False Adulteration: ", "Only 2 pure samples were misclassified, safeguarding consumers and traders from false alarms."),
        ("Balanced Accuracy: ", "95.2% overall test accuracy on totally unseen images confirms genuine feature discrimination.")
    ]
    for i, (b_title, b_desc) in enumerate(eval_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(10.5)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(10.5)
        run2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 9: Edge Optimization & Guardrails
    # =========================================================================
    s9 = add_slide_base(
        title="Edge Optimization, Quantization & App Guardrails",
        category="Edge AI Deployment",
        subtitle="Optimizing the model for zero-cloud latency and bulletproof on-device execution."
    )

    opt_cards = [
        ("Float16 TFLite Quantization", "Converted via TFLiteConverter with FP16 weights. Reduced binary size to 1.95 MB while preserving 99.98% floating point accuracy.", ACCENT_CYAN),
        ("Nitro C++ Native Runtime", "Powered by react-native-fast-tflite directly over Android NDK. Eliminates React Native bridge overhead for <45ms execution.", ACCENT_GREEN),
        ("Universal Texture Gate", "Pixel variance analysis rejects flat non-food surfaces (walls, dark pockets, phone screens) before invoking neural inference.", ACCENT_AMBER),
        ("Color Ratio Pre-Filter", "Food-specific color gating (Red ratio for chili, Brown ratio for teff) rejects off-target objects and non-food contaminants.", ACCENT_RED)
    ]
    for i, (head, body, col) in enumerate(opt_cards):
        cx = Inches(0.8) + i * (c_w + c_gap)
        cy = Inches(1.9)
        card = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, c_w, Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)

        tb = s9.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.2), c_w - Inches(0.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.space_before = Pt(12)
        p2.text = body
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 10: Production Engineering & Repository Cleanup
    # =========================================================================
    s10 = add_slide_base(
        title="Production Engineering & Repository Optimization",
        category="Software Engineering",
        subtitle="Maintaining strict code standards, passing automated unit tests, and optimizing build artifacts."
    )

    add_card(s10, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.8), "Repository Purge (-317 MB)", ACCENT_AMBER)
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    c_bullets = [
        ("Removed Heavy Binaries: ", "Purged 181 MB app-today.apk and 86 MB app-latest.apk from git tracking."),
        ("Cleaned Checkpoint Backups: ", "Removed ~50 MB of intermediate weights (backup_78pct, backup_binary)."),
        ("Deleted Scratch Scripts: ", "Removed ad-hoc test scripts (inspect_tflite.py, test_image.py, dataset_structure.txt)."),
        ("Updated .gitignore Rules: ", "Added *.apk, *.aab, and **/checkpoints/ to guarantee no accidental bloat.")
    ]
    for i, (b_title, b_desc) in enumerate(c_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    add_card(s10, Inches(6.8), Inches(1.9), Inches(5.7), Inches(4.8), "Quality Assurance & CI Readiness", ACCENT_GREEN)
    tb = s10.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    qa_bullets = [
        ("Automated Unit Testing: ", "All Vitest test suites (vitest run) pass with 100% success covering ML inference and SQLite buffering."),
        ("Static Type Safety: ", "Zero TypeScript compile errors across the entire app codebase (tsc --noEmit clean)."),
        ("Race Condition Fix: ", "Added cancellation ref checks across all async steps in camera capture to prevent unmounted state updates."),
        ("Production Logging Gated: ", "Console logs in inference, db init, and result screens gated behind __DEV__ flags.")
    ]
    for i, (b_title, b_desc) in enumerate(qa_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = TEXT_LIGHT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 11: Roadmap & Future Expansion
    # =========================================================================
    s11 = add_slide_base(
        title="Project Roadmap & Next Horizons",
        category="Future Vision",
        subtitle="Scaling Tirat AI across additional staple foods, hardware extensions, and field pilot deployments."
    )

    road_cards = [
        ("1. Brick Powder Fine-Graining", "Integrate Mendeley Dataset cpm7y44746/1 to enable multi-level quantification of brick powder concentration (0% to 100%).", ACCENT_AMBER),
        ("2. Real Teff Harvard Dataset", "Train production multi-head network on Harvard Dataverse Teff dataset (5,000 images, DOI 10.7910/DVN/XKCEX3) to replace current dummy weights.", ACCENT_CYAN),
        ("3. Optical Clip-On Lens Support", "Support 30x-100x smartphone macro clip-on lenses ($5-$10 accessories) to inspect microscopic grain morphology in field trials.", ACCENT_GREEN),
        ("4. Market Pilot & Policy Integration", "Deploy pilot app with agricultural inspectors in Addis Ababa and regional grain hubs with centralized batch telemetry sync.", ACCENT_CYAN)
    ]
    for i, (head, body, col) in enumerate(road_cards):
        cx = Inches(0.8) + i * (c_w + c_gap)
        cy = Inches(1.9)
        card = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, c_w, Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)

        tb = s11.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.2), c_w - Inches(0.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.space_before = Pt(12)
        p2.text = body
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 12: Summary & Conclusion
    # =========================================================================
    s12 = add_slide_base(
        title="Summary & Project Deliverables",
        category="Conclusion",
        subtitle="Transforming mobile phones into reliable, offline food safety instruments."
    )

    add_card(s12, Inches(0.8), Inches(1.9), Inches(11.733), Inches(4.8), "Key Milestones Achieved", ACCENT_GREEN)
    tb = s12.shapes.add_textbox(Inches(1.2), Inches(2.6), Inches(11.0), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True
    concl = [
        ("Diagnosed & Solved Critical Model Skew: ", "Identified 10:1 class imbalance and double-normalization bugs, shifting pure recall from 0% to 96.8%."),
        ("Engineered Robust Dataset: ", "Built balanced 1,232-image dataset with uniform coverage across 16 adulterants and handheld camera augmentations."),
        ("High-Performance Edge AI: ", "Trained MobileNetV3-Small to 96.75% validation accuracy and deployed 1.95 MB Float16 TFLite model."),
        ("Production Mobile Application: ", "Built offline-first React Native + Expo app with C++ native TFLite execution, SQLite buffering, and Amharic/English i18n."),
        ("Clean, CI-Ready Repository: ", "Purged 317 MB of debug artifacts, passed all unit tests and TypeScript typechecks, with automated release APK signing.")
    ]
    for i, (b_title, b_desc) in enumerate(concl):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        run1 = p.add_run()
        run1.text = "✔  " + b_title
        run1.font.bold = True
        run1.font.size = Pt(12)
        run1.font.color.rgb = ACCENT_AMBER
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(12)
        run2.font.color.rgb = TEXT_LIGHT

    out_file = ROOT_DIR / "Tirat_AI_Project_Presentation.pptx"
    prs.save(str(out_file))
    print(f"Presentation saved successfully to: {out_file} ({out_file.stat().st_size:,} bytes)")

if __name__ == "__main__":
    create_presentation()
