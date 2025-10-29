# ML Algorithms — One-Page Cheatsheet

**Goal:** Pick the right tool fast, know what knobs to turn, and avoid common traps.

---

## K-Means (Clustering)
**Use when:** You want to group unlabeled points by similarity.  
**Pros:** Simple, fast, scalable.  
**Cons:** Must choose *k*; sensitive to scale & outliers; assumes spherical clusters.  
**Key params:** `k` (clusters), `max_iter`, `init seeds`.  
**Pitfalls:** Not robust to different densities; features must be scaled.

---

## Linear Regression (Prediction of a number)
**Use when:** Relationship looks roughly linear; you need interpretability.  
**Pros:** Fast, interpretable coefficients, baseline for many tasks.  
**Cons:** Sensitive to outliers, multicollinearity.  
**Key params:** None (closed form); consider regularization (Ridge/Lasso) in practice.  
**Pitfalls:** Non-linearity → consider feature engineering or tree/NN models.

---

## Logistic Regression (Binary classification)
**Use when:** You need probabilities and a linear decision boundary.  
**Pros:** Fast, probabilistic output, well-calibrated with proper regularization.  
**Cons:** Struggles with non-linear boundaries.  
**Key params:** Learning rate, epochs; (practically: regularization strength).  
**Pitfalls:** Unscaled features slow convergence; class imbalance → use class weights.

---

## Decision Trees (CART)
**Use when:** Non-linear relationships and interpretability matter.  
**Pros:** Captures interactions, handles mixed feature types, explainable paths.  
**Cons:** Overfits if deep; unstable to small data changes.  
**Key params:** `max_depth`, `min_samples_split`.  
**Pitfalls:** Prune/limit depth; consider ensembles for better generalization.

---

## Random Forest (Ensemble of trees)
**Use when:** Strong baseline for tabular data with non-linearities.  
**Pros:** Robust, less overfitting than single trees, good accuracy.  
**Cons:** Less interpretable; slower for very large data.  
**Key params:** `n_trees`, `max_depth`, `max_features`.  
**Pitfalls:** Too shallow → underfit; too few trees → unstable.

---

## Linear SVM (Support Vector Machine)
**Use when:** High-dimensional sparse features (e.g., text); need a strong linear classifier.  
**Pros:** Maximizes margin; often strong performance.  
**Cons:** Not probabilistic by default; scaling matters.  
**Key params:** `C` (margin vs. violations), learning rate/epochs (for SGD).  
**Pitfalls:** Wrong label encoding (must be -1/+1 in this repo); unscaled features.

---

## Tiny Neural Network (2-layer MLP)
**Use when:** You expect non-linear decision boundaries (e.g., XOR).  
**Pros:** Flexible function approximator.  
**Cons:** Needs tuning, data, and careful training; less interpretable.  
**Key params:** Hidden units, learning rate, epochs, activation.  
**Pitfalls:** Too few epochs → underfit; too high LR → divergence; no regularization → overfit.

---

## Quick Rules of Thumb
- **Scale features** for K-means, logistic regression, SVM, and NNs.  
- Start with **Logistic/Linear Regression** as baselines; then try **Random Forest** for tabular non-linearities.  
- If you need **probabilities**, use Logistic Regression or calibrated models.  
- For **small data + interpretability**, use Decision Trees (shallow) or Logistic Regression.  
- For **non-linear patterns** (e.g., XOR), use a **Tiny MLP** or tree ensembles.  
- Use **train/validation split** (or CV), check **overfitting** (train vs test).  
- Always set a **seed** for reproducibility.
