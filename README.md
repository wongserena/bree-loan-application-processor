# Bree Loan Default Prediction

## How to Run

1. Clone the repo and create a virtual environment:
\```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
\```

2. The dataset is already included as `loan_applications.csv` (generated once 
   from the provided script).

3. Open and run `loan_default_prediction.ipynb` top to bottom.

## Approach & Key Decisions

**Model choice: XGBoost**
Chosen over logistic regression (too linear for these feature interactions) 
and neural networks (overkill for 1836 rows, harder to explain). XGBoost 
handles missing values natively, works well with class imbalance via 
scale_pos_weight, and produces gain-based feature importances for explainability.

**Data handling**
- Ongoing applications excluded from training (no outcome to learn from). 
  This introduces mild survivorship bias, noted in the notebook.
- Missing documented_monthly_income filled contextually: income_ratio set 
  to 0 for undocumented applicants (missing docs = risk signal, not neutral).
- No SMOTE or oversampling, used scale_pos_weight=2.37 instead. Cleaner 
  and less likely to introduce synthetic artifacts on a small dataset.

**Key finding**
The rule-based system has a 21.5 percentage point approval gap between 
employed and self-employed applicants despite a 0.6 percentage point 
difference in actual default rates. Our model reduces this gap to 2.6 
percentage points while maintaining comparable default detection.

## What I'd Do With More Time
- Fix SHAP compatibility (pin xgboost==2.x) to enable per-applicant 
  explanations for regulatory audits
- Hyperparameter tuning with cross-validation instead of a single train/test split
- Calibrate predicted probabilities (Platt scaling) so scores mean something 
  concrete to reviewers: "this applicant has a 73% default probability"
- Build a proper threshold analysis: the 0.5 decision boundary is arbitrary, 
  the optimal threshold depends on the business cost of false positives vs 
  false negatives
- Retrain periodically as ongoing applications resolve, the model has never 
  seen those outcomes

**Note on SHAP:** Due to a known incompatibility between XGBoost 3.x and 
SHAP 0.49, this project uses XGBoost's built-in gain-based feature 
importances instead of SHAP values. To enable full SHAP support, 
downgrade to xgboost==2.1.1.