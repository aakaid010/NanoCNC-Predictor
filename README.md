# NanoCNC Predictor — ML Training Code
### Kaggle Notebook Documentation
 
---
 
## Overview
 
This notebook trains machine learning models to predict nanocellulose properties from acid hydrolysis parameters. It was developed as part of a Materials Science / Chemical Engineering thesis using 341 real experimental literature records.
 
**Inputs (Features):**
- Cellulose Group (categorical)
- Acid Concentration (wt%)
- Temperature (°C)
- Time (min)
**Outputs (Targets):**
- CNC Length (nm)
- Crystallinity (%)
---
 
## Dataset
 
- **File:** `updated_thesis_dataset_v3.csv`
- **Records:** 341 experimental literature records
- **Path on Kaggle:** `/kaggle/input/datasets/kaziakaid010/dataset/updated_thesis_dataset_v3 (1).csv`
---
 
## What the Code Does (Step by Step)
 
| Step | Description |
|------|-------------|
| 1 | Load dataset and print shape/columns |
| 2 | Define features, targets, and model configurations |
| 3 | EDA plots — missing values, distributions, boxplots, heatmap, scatter plots |
| 4 | Helper functions — data prep, pipeline builder, evaluator, feature importance |
| 5 | Data preparation — one-hot encode Cellulose_Group, extract feature_cols |
| 6 | Train/test split (75% train, 25% test, random_state=42) |
| 7 | Train and evaluate 7 models on both targets |
| 8 | 5-fold cross-validation on top 4 models |
| 9 | Hyperparameter tuning — GridSearchCV on Random Forest |
| 10 | Feature importance for tree-based models |
| 11 | Summary table — R², RMSE, MAE for all models |
| 12 | Auto-select best model per target |
| 13 | Plots — predicted vs actual, model comparison, feature importance, residuals |
| 14 | Save best model bundle as `.pkl` |
 
---
 
## Models Trained
 
| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Ridge | L2 regularization |
| Lasso | L1 regularization |
| Random Forest | 200 trees, max_depth=8 |
| Gradient Boosting | 200 estimators, lr=0.05, max_depth=4 |
| Extra Trees | 200 trees, max_depth=8 |
| SVR | RBF kernel, C=10 |
| RF Tuned | GridSearchCV on Random Forest |
 
---
 
## Figures Saved
 
| File | Description |
|------|-------------|
| `fig_dataset_missing_values.png` | Missing value counts per column |
| `fig_dataset_target_distribution.png` | Histogram of CNC Length and Crystallinity |
| `fig_dataset_group_boxplot.png` | Target distributions by Cellulose Group |
| `fig_dataset_correlation_heatmap.png` | Correlation matrix of numeric variables |
| `fig_dataset_scatter_inputs_targets.png` | Scatter plots of each input vs each target |
| `fig1_predicted_vs_actual.png` | Predicted vs actual for 4 main models |
| `fig2_model_comparison.png` | R² and RMSE bar chart across all models |
| `fig3_feature_importance.png` | Feature importance from best tree model |
| `fig4_residuals.png` | Residual plots for best model |
| `fig5_summary_table.png` | Performance summary table as figure |
| `fig_pred_vs_actual.png` | Predicted vs actual for best model only |
 
---
 
## Model Bundle — What Gets Saved
 
At the end of the notebook, the best model for each target is saved as a single `.pkl` bundle:
 
```python
model_bundle = {
    "length_model":               <sklearn Pipeline>,
    "crystallinity_model":        <sklearn Pipeline>,
    "feature_cols":               [...],   # exact column names used in training
    "cellulose_groups":           [...],   # all known group names
    "model_name_length":          "RF Tuned",
    "model_name_crystallinity":   "RF Tuned",
    "r2_length":                  0.87,
    "r2_crystallinity":           0.79
}
```
 
Each pipeline contains:
```
SimpleImputer(strategy="mean") → StandardScaler → <best model>
```
 
---
 
## Save Code (Add at End of Notebook)
 
```python
import joblib
import sklearn
 
best_length_name = max(results["CNC_Length_nm"],
                       key=lambda m: results["CNC_Length_nm"][m]["R2"])
best_cryst_name  = max(results["Crystallinity_percent"],
                       key=lambda m: results["Crystallinity_percent"][m]["R2"])
 
model_bundle = {
    "length_model":               results["CNC_Length_nm"][best_length_name]["pipe"],
    "crystallinity_model":        results["Crystallinity_percent"][best_cryst_name]["pipe"],
    "feature_cols":               feature_cols,
    "cellulose_groups":           sorted(df["Cellulose_Group"].dropna().unique().tolist()),
    "model_name_length":          best_length_name,
    "model_name_crystallinity":   best_cryst_name,
    "r2_length":                  round(results["CNC_Length_nm"][best_length_name]["R2"], 3),
    "r2_crystallinity":           round(results["Crystallinity_percent"][best_cryst_name]["R2"], 3)
}
 
joblib.dump(model_bundle, "nanocnc_model_bundle.pkl")
 
print(f"sklearn version     : {sklearn.__version__}")
print(f"Length model        : {best_length_name}")
print(f"Crystallinity model : {best_cryst_name}")
print(f"R² Length           : {model_bundle['r2_length']}")
print(f"R² Crystallinity    : {model_bundle['r2_crystallinity']}")
print(f"Feature cols        : {feature_cols}")
print(f"Cellulose groups    : {model_bundle['cellulose_groups']}")
print("Done! Download nanocnc_model_bundle.pkl from the Kaggle Output tab.")
```
 
> **Note the sklearn version printed.** You must install the same version locally to load the model without errors.
 
---
 
## Important Notes
 
**One-hot encoding must match exactly.**
The `feature_cols` list is the exact column order the model was trained on. When loading the model locally, you must build a DataFrame with those exact column names in that exact order.
 
**Best model is selected automatically.**
The code ranks all models by R² (higher is better) and RMSE (lower is better) and picks the top performer per target.
 
**Separate models per target.**
`length_model` and `crystallinity_model` are trained independently — each on their own available rows (rows with missing target values are dropped per target).
 
**Random state is fixed.**
`random_state=42` is used throughout for reproducibility.
