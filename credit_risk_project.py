
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import pickle
import json

# 1) Generate synthetic dataset
def generate_synthetic_loan_data(n=5000, random_state=42):
    np.random.seed(random_state)
    ages = np.random.randint(21, 70, size=n)
    incomes = np.random.normal(50000, 20000, size=n).clip(8000, 250000).astype(int)
    loan_amounts = np.random.normal(15000, 10000, size=n).clip(1000, 150000).astype(int)
    terms = np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.05,0.1,0.4,0.25,0.2])
    credit_scores = np.random.normal(650, 70, size=n).clip(300, 850).astype(int)
    num_defaults = np.random.poisson(0.2, size=n).clip(0,5)
    employment_years = np.random.randint(0, 30, size=n)
    home_ownership = np.random.choice(['RENT','OWN','MORTGAGE','OTHER'], size=n, p=[0.45,0.25,0.25,0.05])
    purpose = np.random.choice(['debt_consolidation','credit_card','home_improvement','major_purchase','small_business','other'],
                               size=n, p=[0.4,0.2,0.15,0.05,0.05,0.15])
    monthly_payment = (loan_amounts / terms) + np.random.normal(0, 50, size=n)
    dti = (monthly_payment * 12) / incomes

    risk_raw = (
        0.00002 * loan_amounts +
        0.6 * (num_defaults) - 
        0.004 * credit_scores + 
        8.0 * dti -
        0.02 * employment_years +
        np.where(home_ownership=='RENT', 0.2, 0.0) +
        np.where(purpose=='small_business', 0.3, 0.0)
    )
    prob_default = 1 / (1 + np.exp(-risk_raw))
    prob_default = np.clip(prob_default * np.random.uniform(0.85,1.15,size=n), 0, 1)
    default = np.random.binomial(1, prob_default)

    df = pd.DataFrame({
        'age': ages,
        'income': incomes,
        'loan_amount': loan_amounts,
        'term_months': terms,
        'credit_score': credit_scores,
        'num_of_defaults': num_defaults,
        'employment_years': employment_years,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'monthly_payment': monthly_payment.round(2),
        'dti': dti.round(4),
        'prob_default': prob_default.round(4),
        'default': default
    })
    return df

# Generate dataset
df = generate_synthetic_loan_data(6000)
df.to_csv("synthetic_credit_data.csv", index=False)

# Preprocessing
numeric_features = ['age','income','loan_amount','term_months','credit_score','num_of_defaults','employment_years','monthly_payment','dti']
categorical_features = ['home_ownership','purpose']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = df[numeric_features + categorical_features]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models
logistic_pipeline = Pipeline(steps=[('preproc', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
rf_pipeline = Pipeline(steps=[('preproc', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])

logistic_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

def evaluate_model(pipeline, X_test, y_test):
    y_prob = pipeline.predict_proba(X_test)[:,1]
    y_pred = pipeline.predict(X_test)
    auc_score = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return auc_score, rep, cm

auc_log, rep_log, cm_log = evaluate_model(logistic_pipeline, X_test, y_test)
auc_rf, rep_rf, cm_rf = evaluate_model(rf_pipeline, X_test, y_test)

print("Logistic Regression AUC:", auc_log)
print(rep_log)
print("Random Forest AUC:", auc_rf)
print(rep_rf)

# Save models
with open("logistic_pipeline.pkl","wb") as f:
    pickle.dump(logistic_pipeline, f)
with open("rf_pipeline.pkl","wb") as f:
    pickle.dump(rf_pipeline, f)

# Save metrics
metrics = {
    "logistic": {"auc": auc_log, "report": rep_log},
    "random_forest": {"auc": auc_rf, "report": rep_rf}
}
with open("metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
