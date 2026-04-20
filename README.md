# Fake Review Detection: ABC-Optimized Ensemble Fusion 🐝📊

An advanced machine learning pipeline designed to distinguish between Original (OR/Genuine) and Computer-Generated (CG/Fake) reviews. This project utilizes a **Late Fusion Architecture**, combining the deep contextual understanding of Transformer models (RoBERTa) with the statistical power of traditional Machine Learning (TF-IDF + Logistic Regression) and custom metadata features. 

The final predictions are orchestrated by an **XGBoost Meta-Classifier**, hyperparameter-tuned from scratch using a custom **Artificial Bee Colony (ABC) Swarm Optimization** algorithm.

---

## 🏗️ Architecture Overview

The pipeline is split into two distinct feature-extraction branches that converge into a highly optimized fusion layer:

### 1. Branch A: Deep Contextual NLP (RoBERTa)
* Analyzes the raw text to grasp deep semantics, sentiment, and context.
* Outputs a confidence probability (`roberta_prob`) of the review being genuine.

### 2. Branch B: Statistical & Metadata ML
* **TF-IDF Vectorization** (Max 5,000 features) captures term frequencies and n-gram statistical patterns.
* **Numerical Metadata Extraction**:
    * `char_count`: Total characters in the review.
    * `word_count`: Total words.
    * `punct_density`: Ratio of punctuation marks to total length.
    * `cap_ratio`: Ratio of capital letters to total length.
    * `rating`: The star rating attached to the review.
* These sparse text vectors and dense numerical features are combined and passed through a **Logistic Regression** model.
* Outputs a secondary confidence probability (`lr_prob`).

### 3. The Fusion Layer (XGBoost + ABC Optimization)
Instead of simply averaging the probabilities, the model learns *when* to trust RoBERTa and *when* to trust the Logistic Regression model based on the metadata.
* **Inputs:** `[roberta_prob, lr_prob, char_count, word_count, punct_density, cap_ratio, rating]`
* **Optimizer:** A custom Artificial Bee Colony (ABC) algorithm uses swarm intelligence to find the mathematically perfect `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` for XGBoost via 3-Fold Cross-Validation.
* **Output:** The final `[Genuine / Fake]` classification.

---

## 🛠️ Tech Stack & Libraries
* **Python 3.x**
* **Data Processing:** `pandas`, `numpy`, `scipy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Deep Learning (Branch A):** `transformers` (HuggingFace), `PyTorch`
* **Visualization:** `matplotlib`, `seaborn`
* **Optimization:** Custom ABC Meta-Heuristic Algorithm

---

## 📂 Project Structure & Workflow

1.  **Data Preprocessing (`Notebook 1`)**
    * Loads raw data and engineers the 5 numerical features.
    * Performs a strict mathematically aligned **80/20 Train/Test Split** to prevent data leakage across branches.
    * Saves foundational data: `train_text_data.csv`, `train_num_data.csv`, etc.
2.  **Branch B Training (`Notebook 1`)**
    * Fits the TF-IDF Vectorizer and scales the numerical features.
    * Trains Logistic Regression and extracts `train_lr_probabilities.csv`.
3.  **Branch A Inference (RoBERTa)**
    * *Note: RoBERTa is trained/run separately due to GPU/Compute requirements.*
    * Takes `train_text_data.csv` and outputs `train_roberta_probabilities.csv`.
4.  **ABC Swarm Optimization & Final Fusion (`Notebook 2`)**
    * Loads all probabilities and scaled metadata (7-feature matrix).
    * Runs the ABC Swarm to tune XGBoost.
    * Evaluates final accuracy, generates confusion matrices, and plots feature importance.
    * Saves the final models for deployment.

---

## 🚀 Saved Deployment Artifacts
Running the complete pipeline generates the following assets required for future inference:
* `abc_xgboost_fusion_model.json` - The finalized swarm-optimized Meta-Classifier.
* `numerical_scaler.pkl` - The StandardScaler object required to normalize metadata for unseen reviews.
* *(And the respective text vectorizers / RoBERTa weights depending on your deployment environment).*

---

## 📊 Results Summary
* **Branch B (Standalone LR + TFIDF):** ~87.15% Accuracy
* **Branch A (Standalone RoBERTa):** ~94.00% Accuracy 
* **Final XGBoost Fusion:** ~97.00% Accuracy *(Achieves superior accuracy by dynamically weighting model strengths based on metadata features).*
