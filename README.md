# Machine Learning Fundamentals: A Study of Predictive Modelling Techniques using R

**Industrial Training Demonstration & Technical Laboratory**  
**Infosys Springboard – "Explore Machine Learning using R"**

[![GitHub Pages](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2563eb?style=for-the-badge&logo=github)](https://shivanshgautamsg.github.io/infosys-springboard/)
[![R Validation](https://img.shields.io/badge/Validation-26%2F26%20Passed%20(100%25)-059669?style=for-the-badge&logo=r)](https://github.com/shivanshgautamsg/infosys-springboard)
[![Tests](https://img.shields.io/badge/Tests-39%2F39%20Passed-0ea5e9?style=for-the-badge)](https://github.com/shivanshgautamsg/infosys-springboard)

**Live Interactive Dashboard:** [https://shivanshgautamsg.github.io/infosys-springboard/](https://shivanshgautamsg.github.io/infosys-springboard/)  
**Repository:** [https://github.com/shivanshgautamsg/infosys-springboard](https://github.com/shivanshgautamsg/infosys-springboard)  

**Author:** Shivansh Gautam  
**Degree:** B.Tech Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Manipal Institute of Technology, Bengaluru  

---

## 1. Project Overview & Authoritative Grounding

This project provides an exact, technically rigorous, reproducible reconstruction of the practical machine learning experiments, statistical models, diagnostic evaluations, and conclusions documented in Shivansh Gautam's Industrial Training Report for **Infosys Springboard**.

Every number, equation, curve, and diagnostic displayed in the dashboard is computed dynamically by R from Fisher's built-in `iris` dataset. **No metrics are hard-coded.**

### Core Experimental Domains Reconstructed
1. **Dataset Exploration:** Verification of the 150 balanced observations (50 per species) across 4 continuous dimensions in centimetres.
2. **Simple Linear Regression:** `lm(Petal.Length ~ Petal.Width)` yielding $R^2 = 0.9271$, $\text{RMSE} = 0.4750\text{ cm}$, and slope $\beta_1 = 2.2299$ ($p < 0.001$).
3. **Multiple Linear Regression:** `lm(Petal.Length ~ Petal.Width + Sepal.Length + Sepal.Width)` yielding $R^2 = 0.9680$ and $\text{RMSE} = 0.3147\text{ cm}$ (a $33.7\%$ reduction in residual variance).
4. **Multicollinearity & VIF Analysis:** Examination of pairwise correlations (e.g. $r = 0.818$ between Petal.Width and Sepal.Length) and Variance Inflation Factors (all $< 4.0$, safely below the threshold of 5.0). Tracing the slope shift of Petal.Width from $2.2299 \rightarrow 1.4468$ due to partial correlation.
5. **Binary Logistic Regression:** `glm(Species ~ Petal.Length + Petal.Width, family = binomial)` on Versicolor vs Virginica ($N = 100$) achieving $94.00\%$ accuracy and $0.9400$ F-score at threshold $0.5$ ($\text{TN}=47, \text{TP}=47, \text{FP}=3, \text{FN}=3$).
6. **Logistic Decision-Threshold Dynamics:** Demonstrating empirical sensitivity across thresholds:
   - $\text{Threshold } 0.3 \rightarrow 96.00\%\text{ Accuracy}$
   - $\text{Threshold } 0.5 \rightarrow 94.00\%\text{ Accuracy}$
   - $\text{Threshold } 0.7 \rightarrow 95.00\%\text{ Accuracy}$
   *Proving that 0.5 is merely an operational convention rather than a mathematical optimum.*
7. **Decision Tree Classification:** `rpart` model at max depth = 3 producing 5 leaves, $97.78\%$ test accuracy (44/45 correct), and Macro F1 = $0.9778$.
8. **Decision Tree Overfitting Experiment:** Contrasting tree depths 2, 3, 4, and Unrestricted. Demonstrating that the unrestricted tree reaches $100.00\%$ training accuracy but drops to $93.33\%$ test accuracy, clearly exposing the overfitting divergence.
9. **K-Nearest Neighbours ($k$-NN):** `class::knn` across odd $k \in \{1, 3, 5, 7, 9, 11\}$. At $k = 5$ normalized, accuracy is $93.33\%$ with Macro F1 = $0.9327$.
10. **Raw vs Min-Max Normalised Features in $k$-NN:** A counterintuitive empirical finding where Raw features achieve $97.78\%$ accuracy while Min-Max Normalisation drops to $93.33\%$. Proves that normalising features of the same unit (cm) compresses high-signal petal features into the same range as low-signal sepal width, diluting the petal signal.
11. **Final Model Synthesis:** Highlighting the depth-3 decision tree as the top performer ($97.78\%$) and explaining why orthogonal axis-aligned boundaries naturally model the Iris morphology.

---

## 2. Directory Structure

```
itr-ml-r-dashboard/
├── app.R                          # Production-grade R Shiny dashboard
├── reproduce_results.R            # Standalone CLI execution & export pipeline
├── validate_results.R             # Automated tolerance-based validation script
├── README.md                      # Complete documentation & viva presentation guide
├── R/
│   ├── data.R                     # Dataset loading, stratified split, scaling
│   ├── metrics.R                  # Statistical & ML evaluation metrics
│   ├── regression.R               # Simple & Multiple linear regression
│   ├── multicollinearity.R        # Correlation matrix, VIF, slope shift analysis
│   ├── logistic.R                 # Binary logistic regression & threshold sensitivity
│   ├── decision_tree.R            # Decision trees (depth-limited & overfitting)
│   ├── knn.R                      # KNN classifier (k sensitivity, raw vs norm)
│   └── validation.R               # 26 automated benchmark verification checks
├── results/
│   ├── results.csv                # Tabular validation audit log
│   ├── results.json               # Full machine-readable experimental metrics
│   ├── model_comparison.csv       # Comparative synthesis of all model families
│   └── validation_report.txt      # 26/26 verification text report
├── tests/
│   └── testthat/
│       └── test_models.R          # 39/39 passing unit tests
└── www/
    └── styles.css                 # Custom technical/academic laboratory stylesheet
```

---

## 3. Installation & Prerequisites

### Prerequisites
- **R Runtime:** Version 4.0 or higher (Tested on R 4.6.1 ucrt on Windows).

### Package Installation
Run the following command in R or your terminal:

```R
install.packages(c(
  "shiny", "bslib", "ggplot2", "dplyr", "tidyr", 
  "rpart", "rpart.plot", "class", "caret", 
  "DT", "plotly", "jsonlite", "testthat"
), repos = "https://cloud.r-project.org")
```

---

## 4. How to Run the Project

### A. Run Automated Reproduction Pipeline
To execute all models from scratch, compute all metrics, and export results to CSV and JSON:

```bash
Rscript reproduce_results.R
```

### B. Run Automated Benchmark Validation
To test all live computed values against the report targets:

```bash
Rscript validate_results.R
```
*Expected Result:* `26 / 26 Validation Checks Passed (100.0%)`.

### C. Run Unit Test Suite
To verify model contracts, scaling isolation, and metric computations:

```bash
Rscript -e "testthat::test_file('tests/testthat/test_models.R')"
```
*Expected Result:* `39 / 39 tests passing with 0 failures, 0 warnings`.

### D. Launch the Interactive Shiny Dashboard
To start the laboratory dashboard locally:

```bash
Rscript -e "shiny::runApp('.', port = 8787, launch.browser = TRUE)"
```
Then open your browser to `http://127.0.0.1:8787`.

---

## 5. Official Verification Target Audit (26 / 26 Passed)

| Model Family | Metric | Report Target | Computed Live | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Simple Linear Regression** | $R^2$ | 0.9271 | 0.9271 | **PASS** |
| **Simple Linear Regression** | Adjusted $R^2$ | 0.9266 | 0.9266 | **PASS** |
| **Simple Linear Regression** | RMSE (cm) | 0.4750 | 0.4750 | **PASS** |
| **Simple Linear Regression** | Slope (Petal.Width) | 2.2299 | 2.2299 | **PASS** |
| **Multiple Linear Regression** | Multiple $R^2$ | 0.9680 | 0.9680 | **PASS** |
| **Multiple Linear Regression** | Adjusted $R^2$ | 0.9674 | 0.9674 | **PASS** |
| **Multiple Linear Regression** | RMSE (cm) | 0.3147 | 0.3147 | **PASS** |
| **Multiple Linear Regression** | Coef (Petal.Width) | 1.4468 | 1.4468 | **PASS** |
| **Multiple Linear Regression** | Coef (Sepal.Length) | 0.7291 | 0.7291 | **PASS** |
| **Multiple Linear Regression** | Coef (Sepal.Width) | -0.6460 | -0.6460 | **PASS** |
| **Multicollinearity Diagnostics**| VIF: Petal.Width | 3.8900 | 3.8900 | **PASS** |
| **Multicollinearity Diagnostics**| VIF: Sepal.Length | 3.4160 | 3.4157 | **PASS** |
| **Multicollinearity Diagnostics**| VIF: Sepal.Width | 1.3060 | 1.3055 | **PASS** |
| **Binary Logistic Regression** | Accuracy ($\theta = 0.5$) | 0.9400 | 0.9400 | **PASS** |
| **Binary Logistic Regression** | F1-Score ($\theta = 0.5$) | 0.9400 | 0.9400 | **PASS** |
| **Binary Logistic Regression** | True Negatives (TN) | 47 | 47 | **PASS** |
| **Binary Logistic Regression** | True Positives (TP) | 47 | 47 | **PASS** |
| **Threshold Experiment** | Accuracy ($\theta = 0.3$) | 0.9600 | 0.9600 | **PASS** |
| **Threshold Experiment** | Accuracy ($\theta = 0.7$) | 0.9500 | 0.9500 | **PASS** |
| **Decision Tree** | Test Accuracy (Depth 3) | 0.9778 | 0.9778 | **PASS** |
| **Decision Tree** | Test Macro F1 (Depth 3) | 0.9778 | 0.9778 | **PASS** |
| **Overfitting Divergence** | Train Accuracy (Unrestricted) | 1.0000 | 1.0000 | **PASS** |
| **Overfitting Divergence** | Test Accuracy (Unrestricted) | 0.9333 | 0.9333 | **PASS** |
| **K-Nearest Neighbours** | Raw Features ($k = 5$) | 0.9778 | 0.9778 | **PASS** |
| **K-Nearest Neighbours** | Normalised ($k = 5$) | 0.9333 | 0.9333 | **PASS** |
| **K-Nearest Neighbours** | Normalised Macro F1 ($k = 5$)| 0.9327 | 0.9333 | **PASS** |

---

## 6. Recommended 5-to-10 Minute Demonstration Script for Evaluators

When presenting to faculty or industrial training evaluators, navigate to the **Demo Mode** tab in the dashboard and follow this exact 12-step sequence:

- **Step 1: Dataset Grounding (1 min)**
  - *Action:* Point to the 150 observations, 3 species (50 each), 4 features in cm.
  - *Talking Point:* "We begin with Fisher's 1936 Iris dataset. Notice that all measurements share the same physical unit (cm). This physical consistency will become crucial when we examine feature scaling."

- **Step 2: Simple Linear Regression (30 sec)**
  - *Action:* Show the fitted line: $\text{Petal.Length} = 1.0836 + 2.2299 \times \text{Petal.Width}$.
  - *Talking Point:* "Petal Width alone explains $92.71\%$ of the variance in Petal Length with an RMSE of $0.4750\text{ cm}$."

- **Step 3: Multiple Linear Regression (30 sec)**
  - *Action:* Show the multiple equation: $\text{Petal.Length} = -0.2627 + 1.4468 \text{PW} + 0.7291 \text{SL} - 0.6460 \text{SW}$.
  - *Talking Point:* "Adding sepal measurements improves $R^2$ to $0.9680$ and reduces residual error by $33.7\%$ to $0.3147\text{ cm}$."

- **Step 4: Multicollinearity & VIF Analysis (1 min)**
  - *Action:* Point out the Petal.Width slope change from $2.2299 \rightarrow 1.4468$ and VIF bars.
  - *Talking Point:* "Notice the slope of Petal Width dropped by $35\%$. This is due to shared correlation with Sepal Length ($r = 0.818$). However, because all VIF values remain below 4.0, multicollinearity is well within acceptable limits."

- **Step 5: Binary Logistic Regression (30 sec)**
  - *Action:* Show the 94% accuracy, TN=47, TP=47, FP=3, FN=3.
  - *Talking Point:* "On the 100 observations of Versicolor and Virginica, logistic regression achieves $94.00\%$ accuracy at the standard 0.5 cutoff."

- **Step 6: Threshold Sensitivity Experiment (1 min)**
  - *Action:* Move the threshold slider to 0.3 (shows 96%) and 0.7 (shows 95%).
  - *Talking Point:* "By shifting the threshold to 0.3, accuracy actually improves to $96.00\%$. This proves that 0.5 is merely an operational convention, not an inherent mathematical optimum."

- **Step 7: Depth-3 Decision Tree (1 min)**
  - *Action:* Show the rendered tree (5 leaves, 97.78% test accuracy).
  - *Talking Point:* "A shallow, depth-3 tree misclassifies only 1 test observation. Because Iris morphology is separated by orthogonal cuts on petal measurements, decision trees represent the problem naturally."

- **Step 8: Overfitting Experiment (1 min)**
  - *Action:* Switch to 'Unrestricted' tree. Show the training curve at 100% and test curve dropping to 93.33%.
  - *Talking Point:* "Here is the textbook demonstration of overfitting: unconstrained trees memorize training noise, reaching 100% training accuracy while out-of-sample generalization drops."

- **Step 9 & 10: $k$-NN & Feature Scaling Counter-Intuition (1.5 min)**
  - *Action:* Toggle from Raw ($97.78\%$) to Normalised ($93.33\%$).
  - *Talking Point:* "This was the report's most unexpected finding. Normalization actually degraded $k$-NN accuracy from $97.78\%$ down to $93.33\%$. Why? Because all features were already in centimetres. Min-max normalization artificially compressed the high-signal petal features into the same range as the low-signal sepal width, diluting the signal."

- **Step 11 & 12: Final Synthesis & Defense Conclusion (30 sec)**
  - *Action:* Display the master comparison table.
  - *Talking Point:* "The overall lesson from this industrial training work is clear: *Model selection in practice is governed less by algorithmic novelty than by disciplined, out-of-sample evaluation.*"
