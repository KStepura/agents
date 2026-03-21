# Tabular regression and Kaggle MSE (short guide)

## EDA priorities
- Check missing values per column; impute or drop with justification.
- Encode high-cardinality categoricals carefully (target encoding or grouping) to avoid overfitting.
- Parse dates into numeric features (days since epoch, month, weekday).
- Inspect correlation with the target; remove or combine collinear numeric features if needed.

## Feature engineering
- Scale numeric inputs for linear models (Ridge); tree models are less sensitive.
- One-hot or ordinal encoding for categories; use `handle_unknown` for test-only levels.
- Log-transform skewed positive targets or features when distributions are heavy-tailed (if competition allows).

## Modeling
- Strong baselines: Ridge, Random Forest, Gradient Boosting (XGBoost, LightGBM).
- Tune `n_estimators`, `max_depth`, and `learning_rate` on validation MSE; early stopping helps boosting.
- For final submission, train on full training data after choosing hyperparameters on a holdout or CV.

## Submission
- Output CSV with columns `index` and `prediction` matching the sample format.
- Clip predictions to plausible ranges if domain knowledge suggests bounds (e.g. non-negative prices).
