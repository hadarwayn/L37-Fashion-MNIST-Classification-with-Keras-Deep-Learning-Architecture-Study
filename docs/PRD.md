# L37 – Fashion-MNIST Classification Using Keras
## Deep Learning Architecture Exploration Study

**Version:** 3.0  
**Last Updated:** February 2026  
**Course:** AI Developer Expert (Dr. Yoram Segal)  
**Previous Lesson:** L36 – MNIST Digits Classification with Keras  

---

## 1. Project Overview

### Project Name
L37 – Fashion-MNIST Deep Learning Architecture Study

### One-Line Description
A hands-on educational project that classifies clothing images into 10 categories using **10 different neural network architectures**, compares their performance scientifically, and teaches *why* certain designs work better than others — explained so even a 12-year-old can follow along.

### Problem Statement
Imagine you work at an online clothing store. Every day, thousands of new items arrive, and someone must sort them: "This is a shoe," "That's a bag," "This is a coat." Doing it by hand would take forever!

**Our goal:** Teach a computer to look at a tiny black-and-white photo of a clothing item (28×28 pixels — about the size of your thumbnail) and correctly say which of 10 categories it belongs to.

But we won't just build *one* model. We'll build **10 different architectures** and run a scientific experiment to answer these real questions:

| Question | Why It Matters |
|----------|---------------|
| Why do CNNs beat Fully Connected networks for images? | Understanding the right tool for the right job |
| Does making a network deeper always help? | Knowing when "more" stops being "better" |
| Does making a network wider always help? | Understanding the depth vs. width tradeoff |
| How do different loss functions change learning? | Choosing the right "grading system" for your model |
| Can we add smart constraints to the loss function? | Advanced optimization techniques |
| How much does hardware (CPU vs GPU) matter? | Real-world deployment decisions |

### Why This Matters (Real-World Impact)

This isn't just a homework exercise. The exact same techniques are used in:

- **E-commerce:** Amazon, Zalando, and ASOS automatically tag millions of product images
- **Quality Control:** Factories check clothing items for defects on assembly lines
- **Fashion AI:** Apps like "Google Lens" identify clothing items from photos
- **Accessibility:** Helping visually impaired people identify what they're wearing
- **Inventory Management:** Warehouses automatically sort incoming clothing shipments

### Relationship to L36 (MNIST Digits)
This project builds directly on L36 (digit classification). The key differences:

| Aspect | L36 (Digits) | L37 (Fashion) |
|--------|-------------|---------------|
| Dataset | Handwritten digits 0–9 | Clothing items |
| Difficulty | Easier (clear shapes) | Harder (similar-looking items) |
| Architectures | Basic models | 10 diverse architectures |
| Loss Functions | Standard only | Standard + Custom regularized |
| Analysis Depth | Basic comparison | Full scientific study |

---

## 2. Dataset Description

### Source
Fashion-MNIST dataset — **built directly into Keras** (no download needed!).

Created by Zalando Research as a more challenging drop-in replacement for MNIST digits.

### Dataset Specifications

| Property | Value |
|----------|-------|
| Total Images | 70,000 |
| Image Size | 28 × 28 pixels (784 total pixels) |
| Color | Grayscale (1 channel, values 0–255) |
| Number of Classes | 10 |
| Training Set | 60,000 images |
| Test Set | 10,000 images |

### The 10 Classes

| Label | Class Name | Real-World Example | Difficulty Note |
|-------|-----------|-------------------|-----------------|
| 0 | T-shirt/top | A basic round-neck t-shirt | Often confused with Shirt (6) |
| 1 | Trouser | Jeans or dress pants | Usually easy — unique shape |
| 2 | Pullover | A sweater you pull over your head | Often confused with Coat (4) |
| 3 | Dress | Any kind of dress | Medium difficulty |
| 4 | Coat | A jacket or coat | Often confused with Pullover (2) |
| 5 | Sandal | Open-toed shoes | Usually easy — unique shape |
| 6 | Shirt | A button-down shirt | Hardest! Looks like T-shirt (0) |
| 7 | Sneaker | Athletic shoes | Sometimes confused with Ankle boot (9) |
| 8 | Bag | A handbag or purse | Usually easy — unique shape |
| 9 | Ankle boot | Short boots | Sometimes confused with Sneaker (7) |

### Why Shirts Are the Hardest (The "Shirt Problem")
The most common confusion in Fashion-MNIST is between T-shirt (0), Pullover (2), and Shirt (6). In a tiny 28×28 grayscale image, these three items look almost identical! This is what makes Fashion-MNIST more challenging than digit MNIST, where a "3" looks very different from a "7".

### Data Split Strategy

```
Total: 70,000 images
├── Training:    50,000 images (used to teach the model)
├── Validation:  10,000 images (used to check learning during training)
└── Test:        10,000 images (final exam — NEVER seen during training)
```

**How we split:** Keras provides 60,000 train + 10,000 test. We further split the 60,000 into 50,000 training + 10,000 validation.

### MANDATORY Pre-Training Check: Class Distribution
Before ANY training begins, we MUST verify that classes are balanced (roughly equal number of items per class). Why? If 90% of our data is "T-shirts," the model might just guess "T-shirt" every time and get 90% accuracy — without actually learning anything!

**Expected:** ~5,000–6,000 images per class in training, ~1,000 per class in validation and test.

**Visualization Required:** A bar chart showing the count of each class.

---

## 3. Target Users & Applications

### Primary Users

| User | How They Benefit |
|------|-----------------|
| AI/ML Students | Learn deep learning architecture design through hands-on experiments |
| Course Instructor | Evaluate student understanding of CNN concepts |
| Self-Learners | Follow along as a tutorial to understand neural network design choices |
| Portfolio Reviewers | See evidence of systematic ML experimentation skills |

### Real-World Use Cases

**Use Case 1: Online Fashion Retailer**
- **Who:** Product listing teams at Zalando, ASOS, or Amazon
- **Scenario:** A seller uploads 500 new product photos. The system automatically classifies each as "Dress," "Sneaker," "Bag," etc., and places them in the correct category
- **Benefit:** Saves hours of manual sorting; reduces mis-categorized products

**Use Case 2: Clothing Donation Center**
- **Who:** Volunteers sorting donated clothing
- **Scenario:** A camera above the sorting table photographs each item. The system suggests "This looks like a Coat" to help speed up sorting
- **Benefit:** Faster processing; helpful for volunteers who can't identify certain items

**Use Case 3: Fashion Search Engine**
- **Who:** Shoppers looking for specific clothing types
- **Scenario:** User uploads a photo of shoes they like. The system classifies the image and searches for similar items in the catalog
- **Benefit:** Visual search instead of text-based search

---

## 4. Functional Requirements

### 4.1 Core Pipeline

| # | Requirement | Description |
|---|------------|-------------|
| FR-1 | Load Dataset | Load Fashion-MNIST from `keras.datasets` |
| FR-2 | Explore Data | Display sample images from each class + class distribution bar chart |
| FR-3 | Preprocess | Normalize pixel values from 0–255 to 0.0–1.0 |
| FR-4 | Split Data | 50K train / 10K validation / 10K test |
| FR-5 | Train Models | Train all 10 architectures (see Section 5) |
| FR-6 | Compare Loss Functions | Train selected architectures with both Sparse Categorical Crossentropy AND Categorical Crossentropy |
| FR-7 | Custom Loss | Implement at least one regularized loss function (e.g., L2-weighted loss) |
| FR-8 | Measure Time | Record wall-clock training time for every model |
| FR-9 | Report Hardware | Print device info (CPU/GPU type, RAM allocated by Colab) |
| FR-10 | Evaluate | Accuracy, loss, and confusion matrix on test set for every model |

### 4.2 Visualization Requirements

| # | Visualization | Purpose |
|---|--------------|---------|
| VIZ-1 | Sample Images Grid | Show 2–3 example images per class (20–30 images total) |
| VIZ-2 | Class Distribution Bar Chart | Verify balanced classes before training |
| VIZ-3 | Accuracy vs. Epoch (per model) | Show learning curves during training |
| VIZ-4 | Loss vs. Epoch (per model) | Show convergence behavior |
| VIZ-5 | Confusion Matrix (per model) | Show which classes get confused with each other |
| VIZ-6 | Architecture Comparison Bar Chart | Side-by-side accuracy comparison of all 10 models |
| VIZ-7 | Training Time Comparison | Bar chart of training duration per model |
| VIZ-8 | FC vs. CNN Comparison | Dedicated plot showing why CNNs win |
| VIZ-9 | Loss Function Comparison | Same architecture with different loss functions |
| VIZ-10 | Misclassified Examples | Show images the model got wrong + what it predicted |

### 4.3 Output Requirements

| Output | Format | Location |
|--------|--------|----------|
| Training logs | Text in notebook | Inline cells |
| Accuracy/Loss plots | PNG images | `results/graphs/` |
| Confusion matrices | PNG images | `results/graphs/` |
| Comparison tables | Markdown tables | README.md |
| Model summaries | Text | Notebook + README |
| Hardware report | Text | Notebook + README |
| Training times | Table | Notebook + README |

---

## 5. Model Architectures (10 Total)

### Architecture Design Philosophy

We organize our 10 models into **4 groups**, each answering a specific research question:

### Group A: Fully Connected Baselines (Models 1–3)
*Question: How well can we do WITHOUT convolutions?*

#### Model 1: FC Baseline
- **Architecture:** Flatten → Dense(128, ReLU) → Dense(10, Softmax)
- **Purpose:** Our simplest starting point. Treats the image as a flat list of 784 numbers.
- **Real-World Analogy:** Like reading a book by looking at all the letters scattered randomly on a table — you lose all the spatial arrangement.
- **Expected Accuracy:** ~85–87%

#### Model 2: Narrow Deep FC
- **Architecture:** Flatten → Dense(64, ReLU) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
- **Purpose:** Test if adding more layers (depth) helps FC networks. Narrow = fewer neurons per layer.
- **Real-World Analogy:** Like a long, thin pipe — information passes through many stages but each stage can only handle a little at a time.
- **Expected Accuracy:** ~84–87%

#### Model 3: Wide Shallow FC
- **Architecture:** Flatten → Dense(512, ReLU) → Dense(10, Softmax)
- **Purpose:** Test if making a single layer very wide (many neurons) is better than going deep.
- **Real-World Analogy:** Like a wide highway with lots of lanes but only one exit — lots of capacity but only one processing step.
- **Expected Accuracy:** ~86–88%

### Group B: CNN Exploration (Models 4–7)
*Question: How do depth and width affect CNN performance?*

#### Model 4: Baseline CNN
- **Architecture:** Conv2D(32, 3×3) → MaxPool → Conv2D(64, 3×3) → MaxPool → Flatten → Dense(128) → Dense(10)
- **Purpose:** Our CNN starting point. The "standard recipe" that most tutorials use.
- **Real-World Analogy:** Like reading a book properly — first you recognize individual letters (edges), then words (patterns), then sentences (objects).
- **Expected Accuracy:** ~89–91%

#### Model 5: Deep CNN
- **Architecture:** 4 Conv blocks (Conv2D → Conv2D → MaxPool), each with increasing filters (32→64→128→256), then Dense layers
- **Purpose:** Test if stacking more convolutional layers improves accuracy.
- **Real-World Analogy:** Like a detective examining evidence at increasingly higher levels — first physical clues, then fingerprints, then DNA, then behavioral patterns.
- **Expected Accuracy:** ~90–92%

#### Model 6: Very Deep CNN (5+ Conv Blocks)
- **Architecture:** 5+ Conv blocks with filters (32→64→128→256→512), using same-padding to preserve dimensions
- **Purpose:** Push depth to the extreme. At what point does "deeper" stop helping?
- **Real-World Analogy:** Like a bureaucracy with too many departments — at some point, adding more levels of review slows things down without improving the decision.
- **Expected Accuracy:** ~90–93% (may suffer from vanishing gradients)

#### Model 7: Wide CNN
- **Architecture:** Conv2D(128, 3×3) → MaxPool → Conv2D(256, 3×3) → MaxPool → Flatten → Dense(256) → Dense(10)
- **Purpose:** Instead of going deep, go wide. Fewer layers but each has many more filters.
- **Real-World Analogy:** Like having a huge team of specialists all working on the same level — lots of eyes but only one round of review.
- **Expected Accuracy:** ~89–91%

### Group C: Regularization Techniques (Models 8–9)
*Question: How can we prevent the model from "memorizing" instead of "learning"?*

#### Model 8: CNN + Dropout
- **Architecture:** Same as Model 4 (Baseline CNN) but with Dropout(0.25) after each Conv block and Dropout(0.5) before the final Dense layer
- **Purpose:** Test if randomly "turning off" neurons during training prevents overfitting.
- **Real-World Analogy:** Like a sports team that randomly benches star players during practice — forces every player to be good, so the team doesn't depend on just one person.
- **Expected Accuracy:** ~89–92% (better generalization than Model 4)

#### Model 9: CNN + Batch Normalization
- **Architecture:** Same as Model 4 but with BatchNorm after every Conv2D and Dense layer (before activation)
- **Purpose:** Test if normalizing the data flowing between layers speeds up training and improves accuracy.
- **Real-World Analogy:** Like a factory quality check between each production step — make sure the half-finished product is in good shape before sending it to the next machine.
- **Expected Accuracy:** ~90–92%

### Group D: Advanced Architecture (Model 10)
*Question: Can we use advanced techniques to push accuracy even higher?*

#### Model 10: CNN + Residual-Style Skip Connections
- **Architecture:** Based on Model 5 (Deep CNN) but with shortcut connections that skip over Conv blocks (like a mini-ResNet)
- **Purpose:** Test if skip connections allow us to train deeper networks without vanishing gradients.
- **Real-World Analogy:** Like having both an elevator AND stairs in a building — information can take the shortcut (elevator/skip) if the regular path (stairs/convolutions) is too slow or lossy.
- **Expected Accuracy:** ~91–93%

### Architecture Summary Table

| # | Model Name | Type | Layers | Parameters (est.) | Key Feature |
|---|-----------|------|--------|-------------------|-------------|
| 1 | FC Baseline | FC | 2 Dense | ~101K | Simplest model |
| 2 | Narrow Deep FC | FC | 4 Dense | ~57K | Deep but narrow |
| 3 | Wide Shallow FC | FC | 2 Dense | ~406K | Wide single layer |
| 4 | Baseline CNN | CNN | 2 Conv + 2 Dense | ~200K | Standard CNN |
| 5 | Deep CNN | CNN | 8 Conv + 2 Dense | ~800K | Many layers |
| 6 | Very Deep CNN | CNN | 10+ Conv + 2 Dense | ~1.5M | Extreme depth |
| 7 | Wide CNN | CNN | 2 Conv + 2 Dense | ~1M | Fat layers |
| 8 | CNN + Dropout | CNN | 2 Conv + Dropout + 2 Dense | ~200K | Anti-overfitting |
| 9 | CNN + BatchNorm | CNN | 2 Conv + BN + 2 Dense | ~201K | Faster training |
| 10 | CNN + Skip | CNN | 8 Conv + Skip + 2 Dense | ~850K | Residual connections |

---

## 6. Loss Functions

### 6.1 Standard Loss Functions (Both MUST be tested)

#### Sparse Categorical Crossentropy
- **Labels format:** Integer labels (e.g., `y = 3` means "Dress")
- **When to use:** When your labels are single numbers (0, 1, 2, …, 9)
- **Real-World Analogy:** Like a teacher grading with a simple answer key — "The correct answer is C"

#### Categorical Crossentropy
- **Labels format:** One-hot encoded (e.g., `y = [0,0,0,1,0,0,0,0,0,0]` means "Dress")
- **When to use:** When your labels are converted to binary vectors
- **Real-World Analogy:** Like a teacher grading with a checklist — "Not A, Not B, Not C, YES D, Not E, …"

**Comparison Experiment:** Train Model 4 (Baseline CNN) with BOTH loss functions and compare:
- Final accuracy difference
- Convergence speed (which reaches good accuracy faster?)
- Loss curve shape

### 6.2 Custom Regularized Loss Function

**Purpose:** Show that we can add "penalties" to the loss function to control what the model learns.

#### L2-Regularized Crossentropy
```
Total Loss = Crossentropy Loss + λ × Σ(weights²)
```

**What this does:** Adds a penalty for large weights. The model is "punished" for making any single connection too strong, which prevents over-reliance on specific features.

**Real-World Analogy:** Imagine a basketball team where the coach says: "Your score depends on points scored MINUS a penalty for every player who scores more than 20 points." This forces the team to spread the scoring around instead of relying on one star player.

**λ (lambda) values to test:** 0.001, 0.01, 0.1

#### Optional Advanced: Custom Weighted Loss
For classes that are harder to classify (like Shirt), we can weight the loss function to "care more" about getting those right:
```
Weighted Loss = Σ(class_weight[i] × loss[i])
```

---

## 7. Technical Requirements

### 7.1 Dual Execution Modes

This project MUST support **two ways to run**, so anyone can use it regardless of their hardware:

#### Mode A: Google Colab Notebook (PRIMARY — Recommended for Most Users)

| Component | Requirement |
|-----------|-------------|
| Platform | Google Colab |
| Runtime | GPU (request T4 or better via Runtime → Change runtime type) |
| Python | 3.10+ (pre-installed) |
| Framework | TensorFlow/Keras (pre-installed) |
| File | `notebooks/L37_Fashion_MNIST_Architecture_Study.ipynb` |

**The Colab notebook is the SHOWCASE version.** It must be:
- **Richly descriptive** — every code cell preceded by a Markdown cell explaining what's about to happen and why
- **Visually dense** — diagrams, sample image grids, inline plots after every training run
- **Table-heavy** — summary tables after each model group comparing results
- **Annotated** — code comments explain every non-obvious line
- **Self-contained** — runs top-to-bottom with zero setup (all dependencies pre-installed in Colab)
- **Interactive** — shows architecture diagrams via `model.summary()` and `tf.keras.utils.plot_model()`

**Colab-Specific Features:**
- Hardware detection cell at the top (GPU type, RAM, disk)
- Google Drive mount option for saving results persistently
- Progress bars during training
- Inline image display for sample images, confusion matrices, and comparison charts
- Markdown sections with teaching content, analogies, and "What did we learn?" summaries

#### Mode B: Local Execution (For Users with a Good Device)

| Component | Requirement |
|-----------|-------------|
| Platform | Local machine (Windows WSL, Linux, or macOS) |
| Python | 3.10+ |
| Virtual Environment | UV (MANDATORY per course guidelines) |
| Framework | TensorFlow/Keras |
| GPU (optional) | NVIDIA GPU with CUDA support (dramatically faster, but not required) |
| File | `main.py` + modules in `src/` |

**The local version is the PRODUCTION version.** It must:
- Follow the standard project directory structure (main.py → src/ modules)
- Use UV virtual environment with `requirements.txt`
- Save all results to `results/` directory automatically
- Generate all visualizations as PNG files in `results/graphs/`
- Print a summary report to console after all models finish
- Work on CPU (slower) or GPU (faster) — auto-detect and report

**Local-Specific Features:**
- `main.py` as single entry point
- Command-line arguments to select which models to train (e.g., `python main.py --models 1,4,8` or `python main.py --all`)
- Ring buffer logging system (per course guidelines)
- Modular code in `src/` with all files under 150 lines
- Results saved as reusable CSV files + PNG plots

#### Shared Code Strategy

Both modes use **the same Keras model-building and training logic.** The difference is only in:
- How results are displayed (inline in Colab vs. saved files locally)
- How the environment is set up (auto in Colab vs. UV locally)
- How progress is shown (notebook widgets vs. console output)

```
Shared Core (identical):          Mode-Specific (different):
├── Model architecture code       ├── Colab: Rich Markdown cells + inline plots
├── Training logic                ├── Local: CLI arguments + file saving
├── Evaluation metrics            ├── Colab: Interactive widgets
└── Loss function implementations └── Local: Ring buffer logging
```

### 7.2 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.15+ | Deep learning framework |
| numpy | 1.24+ | Array operations |
| matplotlib | 3.7+ | Plotting |
| seaborn | 0.12+ | Confusion matrix heatmaps |
| scikit-learn | 1.3+ | Metrics (confusion_matrix, classification_report) |

### 7.3 Hardware Reporting (MANDATORY)

Every notebook run MUST print:
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Device name: {tf.test.gpu_device_name()}")
# Also print RAM allocated by Colab
```

This tells us exactly what hardware Google Colab gave us, because:
- Training on GPU is 5–50× faster than CPU
- Different GPUs (T4, V100, A100) have different speeds
- This affects training time measurements

### 7.4 Training Configuration

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Epochs | 20 (default), up to 50 for deep models | Enough to show convergence |
| Batch Size | 64 or 128 | Good balance of speed and stability |
| Optimizer | Adam (default) | Adaptive learning rate, works well out of the box |
| Learning Rate | 0.001 (Adam default) | Standard starting point |
| Early Stopping | patience=5, monitor='val_loss' | Stop if validation loss stops improving |
| Callbacks | EarlyStopping + ModelCheckpoint | Save best model, prevent overtraining |

### 7.5 Performance Requirements

| Metric | Target |
|--------|--------|
| FC models accuracy | 80–88% on test set |
| CNN models accuracy | 88–93% on test set |
| Total training time (all 10 models) | < 30 minutes on GPU |
| Notebook execution | Must run top-to-bottom without errors |

### 7.6 Security Requirements

| Requirement | Implementation |
|------------|---------------|
| No API keys needed | Dataset is built into Keras |
| No external data downloads | Everything is self-contained |
| No hardcoded file paths | Use relative paths or Colab defaults |
| Reproducibility | Set random seeds (42) for consistent results |

---

## 8. Project Structure

### 8.1 Google Colab Notebook Structure

The notebook MUST follow this exact section structure:

```
📓 L37_Fashion_MNIST_Architecture_Study.ipynb
│
├── Section 0: Setup & Hardware Check
│   ├── Import libraries
│   ├── Set random seeds
│   ├── Print GPU/CPU info
│   └── Print TensorFlow version
│
├── Section 1: Data Loading & Exploration
│   ├── Load Fashion-MNIST from Keras
│   ├── Print dataset shapes
│   ├── Display sample images (grid of all 10 classes)
│   ├── Plot class distribution (bar chart)
│   └── Verify balanced classes
│
├── Section 2: Data Preprocessing
│   ├── Normalize to 0–1
│   ├── Split into train/validation/test (50K/10K/10K)
│   ├── Reshape for CNN (add channel dimension)
│   └── One-hot encode labels (for categorical crossentropy)
│
├── Section 3: Helper Functions
│   ├── build_and_train() — unified training function
│   ├── plot_history() — accuracy and loss curves
│   ├── plot_confusion_matrix() — heatmap
│   ├── show_misclassified() — wrong predictions with images
│   └── compare_models() — bar charts and tables
│
├── Section 4: Group A — Fully Connected Models
│   ├── Model 1: FC Baseline
│   ├── Model 2: Narrow Deep FC
│   ├── Model 3: Wide Shallow FC
│   └── Group A Summary & Analysis
│
├── Section 5: Group B — CNN Architectures
│   ├── Model 4: Baseline CNN
│   ├── Model 5: Deep CNN
│   ├── Model 6: Very Deep CNN
│   ├── Model 7: Wide CNN
│   └── Group B Summary & Analysis
│
├── Section 6: Group C — Regularization
│   ├── Model 8: CNN + Dropout
│   ├── Model 9: CNN + Batch Normalization
│   └── Group C Summary & Analysis
│
├── Section 7: Group D — Advanced
│   ├── Model 10: CNN + Skip Connections
│   └── Group D Summary
│
├── Section 8: Loss Function Comparison
│   ├── Sparse Categorical Crossentropy experiments
│   ├── Categorical Crossentropy experiments
│   ├── Custom L2-Regularized Loss
│   └── Loss Function Analysis
│
├── Section 9: Grand Comparison
│   ├── All 10 models accuracy bar chart
│   ├── All 10 models training time chart
│   ├── Best vs Worst analysis
│   ├── FC vs CNN definitive comparison
│   └── Final results table
│
└── Section 10: Conclusions & Learnings
    ├── Key findings summary
    ├── Best architecture recommendation
    ├── What surprised us
    └── Next steps / future experiments
```

### 8.2 Local Execution Directory Structure

```
L37-Fashion-MNIST-Architecture-Study/
├── README.md                          # THE primary deliverable (see Section 9A)
├── main.py                            # Entry point for local execution
├── requirements.txt                   # Exact dependency versions
├── .gitignore                         # Secrets, cache, .venv
├── .env.example                       # Template (if needed)
│
├── venv/                              # Virtual environment indicator
│   └── .gitkeep
│
├── notebooks/                         # Colab notebook
│   └── L37_Fashion_MNIST_Architecture_Study.ipynb
│
├── src/                               # All source code (150 lines max per file)
│   ├── __init__.py
│   ├── data_loader.py                 # Load & preprocess Fashion-MNIST
│   ├── models/                        # One file per model group
│   │   ├── __init__.py
│   │   ├── fc_models.py               # Models 1–3 (Fully Connected)
│   │   ├── cnn_models.py              # Models 4–7 (CNN variants)
│   │   ├── regularized_models.py      # Models 8–9 (Dropout, BatchNorm)
│   │   └── advanced_models.py         # Model 10 (Skip Connections)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Unified train & evaluate pipeline
│   │   └── loss_functions.py          # Custom L2-regularized loss
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── training_plots.py          # Accuracy/Loss curves
│   │   ├── confusion_matrix.py        # Confusion matrix heatmaps
│   │   └── comparison_charts.py       # Cross-model comparison plots
│   └── utils/
│       ├── __init__.py
│       ├── hardware.py                # GPU/CPU detection & reporting
│       ├── paths.py                   # Relative path utilities
│       └── logger.py                  # Ring buffer logging
│
├── docs/                              # Planning documents
│   ├── PRD.md
│   └── tasks.json
│
├── results/                           # All outputs
│   ├── graphs/                        # All PNG visualizations
│   │   ├── class_distribution.png
│   │   ├── sample_images_grid.png
│   │   ├── model_01_fc_baseline_curves.png
│   │   ├── model_01_fc_baseline_confusion.png
│   │   ├── ...                        # (curves + confusion for all 10 models)
│   │   ├── grand_comparison_accuracy.png
│   │   ├── grand_comparison_time.png
│   │   ├── fc_vs_cnn_comparison.png
│   │   └── loss_function_comparison.png
│   ├── tables/
│   │   └── results_summary.csv        # All metrics in one CSV
│   └── examples/
│       ├── misclassified_best_model.png
│       └── misclassified_worst_model.png
│
├── logs/                              # Ring buffer logs
│   ├── config/
│   │   └── log_config.json
│   └── .gitkeep
│
└── config/                            # Training configuration
    └── training_config.yaml           # Hyperparameters for all models
```

---

## 9. README.md — THE Primary Deliverable

### ⚠️ THIS IS THE MOST IMPORTANT PART OF THE ENTIRE PROJECT

The README.md is not just documentation — it IS the project. A person who only reads the README (without running any code) should:
1. **Understand** what neural networks are and how they classify images
2. **See** every result — all graphs, tables, confusion matrices, and sample images embedded directly
3. **Learn** why CNNs beat Fully Connected networks, what depth/width tradeoffs look like, and how regularization works
4. **Feel** like they just completed a mini-course on deep learning architecture design

### README Quality Standard: "The 12-Year-Old Museum Tour"

Imagine a 12-year-old visiting a science museum exhibit about AI. The README is their guided tour. Every section should have:
- A **visual** (image, graph, table, or diagram) — never more than 2 paragraphs without something visual
- A **real-world analogy** — connect every technical concept to something they already know
- An **"Aha!" moment** — explicitly highlight what's surprising or counterintuitive about each result

### README Required Sections (In Exact Order)

#### Section 1: Title, Banner & Abstract
- Project title with emoji
- One-line tagline that hooks the reader
- 3-sentence abstract: what, how, and the most surprising finding
- Badges (Python version, TensorFlow version, License)

#### Section 2: "What Is This Project?" (For Complete Beginners)
- **The Big Picture:** 2-paragraph explanation assuming ZERO technical knowledge
- **The Clothing Sorting Analogy:** "Imagine you're sorting laundry but you're blindfolded and can only feel 784 tiny dots…"
- **Visual:** Grid of sample images from all 10 classes with labels
- **Visual:** A simple diagram showing "Image → Neural Network → Prediction"

#### Section 3: The 10 Fashion Categories
- **Visual:** 2×5 or 3×4 grid showing example images from each class
- **Table:** Class names, descriptions, and difficulty notes
- **The "Shirt Problem":** Explain and SHOW why T-shirt/Pullover/Shirt are confusing (with side-by-side image comparison)

#### Section 4: Dataset Deep Dive
- **Visual:** Class distribution bar chart
- How the data was split (train/validation/test) with visual diagram
- Preprocessing steps explained with before/after pixel value examples

#### Section 5: "What Is a Neural Network?" (Educational Foundation)
- **For 12-year-olds:** Step-by-step visual explanation
- **Visual diagram:** What a neuron does (inputs → weights → sum → activation → output)
- **Visual diagram:** What a layer looks like (many neurons working together)
- **The Bakery Analogy:** Each layer is like a step in a recipe — raw ingredients → mix → bake → decorate → finished cake
- **FC vs. CNN visual comparison:** Show HOW they process the same image differently

#### Section 6: The 10 Architectures — Our Experiments
For EACH of the 10 models, include:

- **Model Card:** Name, group, architecture summary, parameter count
- **Why This Model?** 2-sentence explanation of the unique question it answers
- **Architecture Diagram:** Text diagram or Keras `model.summary()` output
- **The Analogy:** Real-world comparison a kid would understand
- **Training Curves:** Accuracy vs. Epoch AND Loss vs. Epoch (two plots)
- **Confusion Matrix:** Heatmap showing per-class performance
- **Results Table:** Final accuracy, loss, training time, best epoch
- **The Verdict:** Did it meet expectations? What did we learn?

After each GROUP (A, B, C, D), include a **Group Summary** with:
- Side-by-side comparison chart of all models in the group
- Key takeaway in bold
- "If you remember ONE thing from this section…" summary

#### Section 7: Loss Function Comparison
- **Visual:** Same model trained with different loss functions — side-by-side curves
- **Table:** Convergence speed and final accuracy for each loss function
- **Custom Loss explanation:** What L2 regularization does, with the basketball team analogy
- **Visual:** Effect of different lambda values on accuracy

#### Section 8: The Grand Comparison
- **Visual:** Bar chart comparing ALL 10 models' test accuracy (the "hero chart")
- **Visual:** Bar chart comparing ALL 10 models' training time
- **Visual:** FC vs. CNN definitive comparison (Models 1–3 vs. Models 4–10)
- **Master Results Table:** All 10 models in one table with accuracy, loss, time, parameters, unique finding
- **Best & Worst Analysis:** Which model won? Which lost? WHY?
- **The Definitive Answer:** "If you had to choose ONE architecture for Fashion-MNIST, choose ______ because ______"

#### Section 9: Misclassified Examples — "Where the Model Fails"
- **Visual:** Grid of images the BEST model got wrong, showing (True Label → Predicted Label)
- **Visual:** Grid of images the WORST model got wrong
- **Analysis:** Which classes are hardest? Why? (Connect to "The Shirt Problem")

#### Section 10: Hardware & Performance Report
- What GPU/CPU Colab assigned
- Training time breakdown per model
- How much faster GPU is vs. CPU (if tested)

#### Section 11: "What I Learned" — Key Takeaways
- Numbered list of 5–10 insights, each with a supporting visual or number
- "What surprised me most was…"
- "If I were to improve this project, I would…"

#### Section 12: How to Run This Project
- **Option A: Google Colab** (one-click link, step-by-step with screenshots)
- **Option B: Local with UV** (full installation guide per course guidelines)
- Expected output description
- Troubleshooting common issues

#### Section 13: Project Structure
- Directory tree with purpose of each file/folder
- Code Files Summary table (file name, description, line count)

#### Section 14: Technologies & Tools
- Table of all packages with versions and purposes
- Hardware used

#### Section 15: Author, License & Acknowledgments

### README Visual Density Requirement

The README must contain **at minimum** these visuals (embedded as images, not links):

| # | Visual | Type | Location in README |
|---|--------|------|-------------------|
| 1 | Sample images grid (all 10 classes) | PNG | Section 3 |
| 2 | Class distribution bar chart | PNG | Section 4 |
| 3 | Neural network explanation diagram | PNG or text art | Section 5 |
| 4–23 | Training curves (accuracy + loss) per model | PNG | Section 6 (2 per model × 10 models) |
| 24–25 | Confusion matrices (best + worst model minimum) | PNG | Section 6 or 9 |
| 26 | Grand accuracy comparison bar chart | PNG | Section 8 |
| 27 | Grand training time comparison | PNG | Section 8 |
| 28 | FC vs CNN comparison chart | PNG | Section 8 |
| 29 | Loss function comparison | PNG | Section 7 |
| 30 | Misclassified examples grid | PNG | Section 9 |

**Total minimum: ~30 embedded images/charts in the README.**

### README Length Expectation
Given the educational scope and 10 model architectures, the README is expected to be **extensive** — potentially 500–1000+ lines of Markdown. This is intentional. Quality and completeness over brevity. Every image, every table, every analogy earns its place.

---

## 9A. Educational Requirements (The "12-Year-Old Test")

### Every Model Section MUST Include:

1. **Before Training:** A plain-English explanation of what this architecture does and WHY we're testing it
2. **The Analogy:** A real-world analogy that a 12-year-old would understand
3. **Architecture Diagram:** Either a text diagram or `model.summary()` output
4. **After Training:** What happened? Did it meet our expectations? Why or why not?
5. **The Lesson:** One clear takeaway from this experiment

### Example Teaching Moment (for Model 8: Dropout)

```markdown
### 🎲 Model 8: CNN with Dropout — "Training with a Handicap"

**The Problem:** Sometimes a model gets TOO good at the training data. 
It memorizes specific images instead of learning general patterns. 
This is called "overfitting" — like a student who memorizes test answers 
but can't solve new problems.

**The Solution:** During training, we randomly "turn off" some neurons 
(set them to zero). This is called Dropout.

**Analogy:** Imagine a soccer team where the coach randomly benches 
25% of the players during each practice match. This means:
- No single player can carry the team alone
- Every player MUST learn to be useful
- The team becomes stronger overall because everyone contributes

**In our model:** Dropout(0.25) means 25% of neurons are randomly 
disabled during each training step. At test time, ALL neurons are active 
(but scaled down), giving us the full team's strength.
```

---

## 10. Success Criteria

### 10.1 Functional Success

- [ ] All 10 architectures train successfully without errors
- [ ] All 10 architectures evaluated on test set
- [ ] Both loss functions (Sparse + Categorical) compared
- [ ] Custom regularized loss function implemented
- [ ] Hardware info reported (GPU type, RAM)
- [ ] Training time recorded for every model

### 10.2 Per-Model Accuracy Benchmarks & Uniqueness

Each architecture is specifically chosen to teach a **unique lesson**. No two models answer the same question.

| # | Model Name | Expected Accuracy | What Makes It UNIQUE | Why It's Interesting to Test |
|---|-----------|-------------------|---------------------|----------------------------|
| 1 | **FC Baseline** | 85–87% | The simplest possible model — our "control group" | Sets the floor. Every other model must beat this or we ask "why bother with complexity?" |
| 2 | **Narrow Deep FC** | 84–87% | 3 hidden layers with only 64 neurons each — the "skinny tower" | Tests whether stacking more FC layers helps. Spoiler: for images, depth without convolutions barely helps — this teaches why FC alone isn't enough. |
| 3 | **Wide Shallow FC** | 86–88% | One huge layer with 512 neurons — the "wide net" | Tests whether brute-force width compensates for lack of convolutions. It usually gets slightly better than Model 1 but hits a ceiling — proving that raw neuron count can't replace spatial understanding. |
| 4 | **Baseline CNN** | 89–91% | First model with Conv2D layers — the breakthrough moment | The "aha!" moment. Students see a ~4% accuracy jump just by switching from FC to CNN. This is the single most important comparison in the entire project. |
| 5 | **Deep CNN** | 90–92% | 4 Conv blocks (8 Conv layers) — double the depth of Model 4 | Tests whether "more layers = more accurate." Usually yes, but with diminishing returns. The accuracy gap between Model 4→5 is smaller than Model 1→4, teaching that architecture type matters more than raw depth. |
| 6 | **Very Deep CNN** | 89–93% ⚠️ | 5+ Conv blocks (10+ layers) — pushing depth to the extreme | The "too much of a good thing" experiment. May FAIL to converge or show vanishing gradients. If accuracy drops compared to Model 5, that's actually the most valuable lesson: there's an optimal depth, and going past it hurts. |
| 7 | **Wide CNN** | 89–91% | Fewer layers but 128/256 filters each — the "wide highway" | Direct comparison with Model 5 (deep). Same parameter budget spent differently. Teaches width vs. depth tradeoff: wide models train faster but may not capture hierarchical features as well. |
| 8 | **CNN + Dropout** | 89–92% | Identical to Model 4 but with random neuron deactivation | The ONLY model specifically designed to fight overfitting. Compare its training-vs-validation gap to Model 4. If Model 4 overfits (training acc >> validation acc) but Model 8 doesn't, Dropout proved its value. |
| 9 | **CNN + BatchNorm** | 90–92% | Identical to Model 4 but with normalization between layers | The ONLY model testing training speed optimization. Compare how many epochs it takes to converge vs. Model 4. BatchNorm typically reaches peak accuracy 30–50% faster — a huge deal when training costs money. |
| 10 | **CNN + Skip Connections** | 91–93% | Deep like Model 5 but with residual shortcuts — mini-ResNet | The ONLY model using the technique behind ResNet (which won ImageNet 2015). If it beats Model 6 (Very Deep), it proves skip connections solve the vanishing gradient problem. This is the bridge to modern architectures like GPT and Vision Transformers. |

#### Key Comparison Pairs (Each Answers ONE Question)

| Comparison | Models | Question Answered |
|-----------|--------|-------------------|
| FC vs. CNN | Model 1 vs. Model 4 | Do convolutions matter for images? |
| Shallow vs. Deep FC | Model 1 vs. Model 2 | Does depth help FC networks? |
| Narrow vs. Wide FC | Model 2 vs. Model 3 | Is width or depth better for FC? |
| Shallow vs. Deep CNN | Model 4 vs. Model 5 | Does depth help CNNs? |
| Too Deep CNN | Model 5 vs. Model 6 | Can you go TOO deep? |
| Deep vs. Wide CNN | Model 5 vs. Model 7 | Width or depth for CNNs? |
| With vs. Without Dropout | Model 4 vs. Model 8 | Does Dropout reduce overfitting? |
| With vs. Without BatchNorm | Model 4 vs. Model 9 | Does BatchNorm speed up training? |
| Deep without vs. with Skip | Model 5 vs. Model 10 | Do skip connections help deep nets? |
| Very Deep vs. Skip | Model 6 vs. Model 10 | Can skip connections rescue failing deep nets? |

### 10.3 Visualization Completeness

- [ ] Sample images grid displayed
- [ ] Class distribution bar chart
- [ ] Training curves (accuracy + loss) for every model
- [ ] Confusion matrix for at least the best and worst models
- [ ] Grand comparison bar chart (all 10 models)
- [ ] Training time comparison chart
- [ ] Misclassified examples shown for at least 2 models

### 10.4 Educational Quality

- [ ] Every model section has a plain-English explanation
- [ ] Every model section has a real-world analogy
- [ ] Key concepts explained for beginners (CNN, Dropout, BatchNorm, Skip Connections)
- [ ] "What I learned" conclusions section
- [ ] A 12-year-old could follow the notebook and understand the main ideas

### 10.5 Documentation Completeness

- [ ] **README.md is a complete educational experience** (see Section 9 for full requirements)
- [ ] README contains ~30 embedded images/charts minimum
- [ ] README includes real-world analogies for every architecture
- [ ] README passes the "12-Year-Old Museum Tour" test
- [ ] README includes all 15 required sections in order
- [ ] Google Colab notebook has rich Markdown explanations between every code cell
- [ ] Local execution works via `main.py` with UV virtual environment
- [ ] PRD.md finalized
- [ ] tasks.json created
- [ ] Code Files Summary table in README

---

## 11. Learning Objectives

After completing this project, the student will be able to:

| # | Learning Objective | Demonstrated By |
|---|-------------------|-----------------|
| 1 | Understand why CNNs are superior for image data | FC vs CNN accuracy comparison |
| 2 | Explain the depth vs. width tradeoff | Comparing Models 2, 3, 5, 6, 7 |
| 3 | Identify and prevent overfitting | Dropout and BatchNorm experiments |
| 4 | Choose appropriate loss functions | Sparse vs Categorical comparison |
| 5 | Design custom loss functions with regularization | L2-weighted loss implementation |
| 6 | Read and interpret confusion matrices | Identifying the "Shirt Problem" |
| 7 | Measure and report training performance | Time + hardware reporting |
| 8 | Compare architectures scientifically | Controlled experiments with one variable changed |
| 9 | Understand skip connections (ResNet concept) | Model 10 implementation |
| 10 | Make architecture recommendations based on evidence | Final analysis section |

### Parameter Variations to Explore

| Parameter | Values to Try | What We Learn |
|-----------|--------------|---------------|
| Number of Conv layers | 2, 4, 6, 10+ | Effect of depth |
| Filters per layer | 32, 64, 128, 256 | Effect of width |
| Dropout rate | 0.0, 0.25, 0.5 | Effect of regularization strength |
| Loss function | Sparse CE, Categorical CE, L2-Regularized | Effect on convergence |
| L2 lambda | 0.001, 0.01, 0.1 | Regularization strength |
| Batch size | 32, 64, 128 | Training stability vs speed |

---

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Colab session disconnects during training | Medium | High | Save checkpoints; train models one-by-one |
| GPU not available in Colab | Low | High | Code must also work on CPU (just slower); local mode as backup |
| Very Deep CNN fails to converge | Medium | Low | Expected outcome — teaches about vanishing gradients |
| Training takes too long | Low | Medium | Use EarlyStopping; reduce epochs if needed |
| Overfitting on simple models | High | Low | Expected — this is the point of the experiment |
| Confusion between similar classes | High | Low | Expected — teaches about model limitations |
| Local TensorFlow installation issues | Medium | Medium | Provide detailed troubleshooting in README; Colab as fallback |
| Local GPU not detected | Medium | Low | Auto-fallback to CPU with warning message |

---

## 13. Constraints & Assumptions

### Constraints
- Google Colab free tier may limit GPU availability
- Notebook must run end-to-end in under 1 hour (including all 10 models)
- Local execution must work on CPU (slower) even without GPU
- Images are grayscale only (no color information)
- Dataset is fixed — no data augmentation in this project (that's for a future lesson)

### Assumptions
- Student has completed L36 (MNIST digit classification) and understands basic Keras workflow
- Student has a Google account for Colab access
- For local execution: Python 3.10+ and UV installed (per course guidelines)
- Internet connection available for Colab (local mode works offline after initial pip install)
- No prior knowledge of CNNs required (will be taught in the notebook AND README)

---

## 14. Deliverables Checklist

| # | Deliverable | Format | Location | Priority |
|---|------------|--------|----------|----------|
| 1 | **README.md** | Markdown with ~30 embedded images | Root directory | 🔴 #1 — THE primary deliverable |
| 2 | Google Colab Notebook (richly descriptive) | `.ipynb` | `notebooks/` | 🔴 Critical |
| 3 | Local execution code | `.py` modules | `main.py` + `src/` | 🟡 High |
| 4 | PRD.md | Markdown | `docs/PRD.md` | 🟢 Planning |
| 5 | tasks.json | JSON | `docs/tasks.json` | 🟢 Planning |
| 6 | All result images (graphs, confusion matrices) | PNG | `results/graphs/` | 🔴 Critical |
| 7 | Results summary data | CSV | `results/tables/` | 🟡 High |
| 8 | requirements.txt | Text | Root directory | 🔴 Critical |
| 9 | .gitignore | Text | Root directory | 🔴 Critical |
| 10 | Training config | YAML | `config/training_config.yaml` | 🟡 High |

---

## 15. Glossary (Terms a 12-Year-Old Should Know)

| Term | Simple Explanation |
|------|-------------------|
| **Neural Network** | A computer program inspired by the brain. It learns by adjusting tiny "dials" (weights) until it gets good at a task. |
| **CNN (Convolutional Neural Network)** | A special neural network that's great at understanding images. It scans the image in small patches, like reading with a magnifying glass. |
| **Fully Connected (FC)** | A type of layer where every input is connected to every output. Simple but doesn't understand spatial patterns. |
| **Epoch** | One complete pass through ALL training images. Like re-reading an entire textbook once. |
| **Batch Size** | How many images the model looks at before updating its "dials." Smaller = more careful, Larger = faster but less precise. |
| **Loss Function** | A formula that measures "how wrong" the model is. Lower loss = better model. Like a golf score — lower is better! |
| **Overfitting** | When the model memorizes the training data but can't handle new images. Like studying only past exam papers and failing with new questions. |
| **Dropout** | Randomly turning off some neurons during training to prevent over-reliance on specific pathways. |
| **Batch Normalization** | Standardizing the data between layers so each layer gets "clean" input. Like washing ingredients between cooking steps. |
| **Skip Connection** | A shortcut that lets information skip over layers. Helps deep networks train by preventing information from getting "lost." |
| **Confusion Matrix** | A table showing what the model predicted vs. what was actually correct. Reveals which classes get mixed up. |
| **Accuracy** | Percentage of images the model classified correctly. 90% accuracy = 9 out of 10 correct. |
| **Validation Set** | Images the model NEVER trains on but uses to check its progress. Like practice tests before the real exam. |
| **GPU** | Graphics Processing Unit — a chip originally designed for video games, but amazing at the math needed for AI. Much faster than CPU for neural networks. |
| **Regularization** | Techniques that prevent the model from becoming too complex. Keeps the model "honest" so it generalizes well. |

---

*This PRD was prepared as part of the AI Developer Expert course by Dr. Yoram Segal.*  
*Document version 3.0 — Enhanced with per-model uniqueness, dual execution modes, and README-first philosophy.*
