# Credit Risk Scoring Engine

This is my first complete machine learning project after spending a long time studying the math, algorithms, and Python tooling behind predictive modeling. Most of my earlier practice stayed inside isolated notebooks, so I built this repository to connect the full workflow: pulling raw credit records, engineering domain features, tuning a gradient boosted tree model, tracking experiments, and serving predictions through a REST API.

The pipeline estimates probability of default using the UCI Default of Credit Card Clients dataset (ID 350, published by Yeh and Lien, 2009). It contains 30,000 real client records with 23 baseline variables covering credit limits, demographics, and six months of bill statements, repayment amounts, and delinquency statuses.

## Data pipeline and feature engineering

`src/data/loader.py` downloads the original archive from the UCI repository, checks column integrity, and writes the raw dataset to `data/raw/` as a Parquet file. During preprocessing (`src/data/preprocessor.py`), undocumented category codes present in the raw `education` (`0`, `5`, `6`) and `marriage` (`0`) columns are consolidated into their respective "other" categories (`4` and `3`), matching the dataset documentation.

`src/features/feature_engineering.py` uses `Polars` to compute 12 derived financial features across the six-month observation window:

- Revolving credit utilization in the most recent month (`utilization_ratio_m1`), six-month mean utilization (`avg_utilization_6m`), and six-month peak utilization (`max_utilization_6m`).
- Delinquency frequency and severity metrics, including maximum months past due (`max_delinquency_months`), total months in delay (`months_past_due_count`), months with severe delinquency of 60+ days (`severe_delinquency_count`), and recent repayment trend (`recent_delinquency_trend`).
- Cashflow and repayment ratios covering cumulative six-month bills (`total_bill_6m`), cumulative payments (`total_pay_6m`), payment-to-bill coverage (`payment_to_bill_ratio_6m`), the unpaid balance gap from the prior statement (`recent_payment_gap_m1`), and six-month balance growth (`bill_growth_6m`).

## Model training and evaluation results

The model is trained with `LightGBM` on an 80/20 stratified train-test split (24,000 training rows and 6,000 hold-out test rows). Hyperparameter tuning in `src/models/optimizer.py` runs 5-fold stratified cross-validation using Optuna's Tree-structured Parzen Estimator (TPE) sampler. Experiment parameters, evaluation metrics, and global feature attributions are logged to `MLflow`.

Because undetected defaults carry a higher financial loss than false alarms in credit underwriting, the classification cutoff is selected by minimizing expected cost across the confusion matrix (using a 5.0 weight for false negatives and a 1.0 weight for false positives).

| Metric | Hold-out test set (n = 6,000) |
| :--- | :--- |
| ROC-AUC | 0.78209 |
| Gini coefficient | 0.56418 |
| Kolmogorov-Smirnov (KS) statistic | 0.43292 |
| PR-AUC (Average Precision) | 0.56385 |
| Brier score | 0.13408 |
| Cost-optimal decision threshold | 0.1700 |
| Recall at optimal threshold | 0.74228 |
| Precision at optimal threshold | 0.38492 |
| F1-score at optimal threshold | 0.50695 |

Prediction explainability is handled by `shap.TreeExplainer` in `src/models/explainer.py`. Ranked by mean absolute SHAP value on the test set, the five strongest predictors are `max_delinquency_months` (0.40709), `pay_0` (0.24439), `limit_bal` (0.10298), `bill_amt1` (0.09181), and `bill_growth_6m` (0.08551). The full evaluation output and tuned hyperparameters are stored in `reports/metrics_summary.json`.

## License

Released under the MIT License. See `LICENSE` for details.
