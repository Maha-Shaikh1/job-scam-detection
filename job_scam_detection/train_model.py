import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, f1_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

print("=" * 60)
print("   🛡️  JobShield AI — Model Training")
print("=" * 60)

# ================================================================
# STEP 1 — LOAD & CLEAN DATASET
# ================================================================

df = pd.read_csv("fake_job_postings.csv", encoding="latin-1")
df = df.fillna("")

print(f"\n📂 Dataset loaded: {len(df)} rows")
print(f"📊 Class distribution:\n{df['fraudulent'].value_counts()}")
print(f"   Fraud rate: {df['fraudulent'].mean()*100:.2f}%\n")

# ================================================================
# STEP 2 — FEATURE ENGINEERING
# ================================================================

# ---- 2A: Text Features ----
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
for col in ["location", "employment_type", "required_experience", "industry", "function"]:
    if col in df.columns:
        text_cols.append(col)

df["text"] = df[text_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# ---- 2B: Engineered Numeric Scam Signals ----

# Has company logo — scam jobs often skip this
df["has_logo"] = df["has_company_logo"].astype(float) if "has_company_logo" in df.columns else 0.0

# Telecommuting flag
df["telecommuting"] = df["telecommuting"].astype(float) if "telecommuting" in df.columns else 0.0

# Scam keyword count in full text
SCAM_KEYWORDS = [
    "urgent", "immediately", "guaranteed", "no experience",
    "work from home", "easy money", "daily payment", "weekly payment",
    "send your", "bank account", "aadhaar", "whatsapp", "call now",
    "limited seats", "earn", "per day", "per week", "no qualification",
    "registration fee", "deposit", "investment", "part time", "part-time"
]

df["scam_kw_count"] = df["text"].apply(
    lambda t: sum(1 for kw in SCAM_KEYWORDS if kw in t.lower())
)

# Phone number present in text
df["has_phone"] = df["text"].str.contains(
    r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', regex=True
).astype(float)

# Very short description = suspicious
df["is_short_desc"] = (df["description"].apply(len) < 200).astype(float)

# No company profile = suspicious
df["no_company_profile"] = (df["company_profile"].str.strip() == "").astype(float)

# No requirements = suspicious
df["no_requirements"] = (df["requirements"].str.strip() == "").astype(float)

# Salary range present
df["has_salary_info"] = (
    (df["salary_range"].str.strip() != "") if "salary_range" in df.columns
    else pd.Series([False] * len(df))
).astype(float)

NUMERIC_FEATURES = [
    "has_logo", "telecommuting", "scam_kw_count",
    "has_phone", "is_short_desc", "no_company_profile",
    "no_requirements", "has_salary_info"
]

X_text   = df["text"]
X_numeric = df[NUMERIC_FEATURES].values
y        = df["fraudulent"]

# ================================================================
# STEP 3 — TRAIN / TEST SPLIT
# ================================================================

(X_text_train, X_text_test,
 X_num_train,  X_num_test,
 y_train,      y_test) = train_test_split(
    X_text, X_numeric, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train: {len(y_train)} rows | Test: {len(y_test)} rows\n")

# ================================================================
# STEP 4 — TEXT MODEL  (TF-IDF + Logistic Regression)
#           Primary model — handles job description text
# ================================================================

print("🔄 Training Text Model (TF-IDF + Logistic Regression)...")

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )),
    ("clf", CalibratedClassifierCV(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.5,
            solver="lbfgs"
        ),
        cv=5, method="sigmoid"
    ))
])

text_pipeline.fit(X_text_train, y_train)

classes    = list(text_pipeline.classes_)
fraud_idx  = classes.index(1) if 1 in classes else 1

text_probs_test  = text_pipeline.predict_proba(X_text_test)[:, fraud_idx]
text_probs_train = text_pipeline.predict_proba(X_text_train)[:, fraud_idx]

print(f"   Text Model  ROC-AUC (test): {roc_auc_score(y_test,  text_probs_test):.4f}")

# ================================================================
# STEP 5 — NUMERIC MODEL (Gradient Boosting on engineered features)
#           Catches patterns text alone might miss:
#           missing logo, phone numbers, scam keywords count, etc.
# ================================================================

print("🔄 Training Numeric Feature Model (Gradient Boosting)...")

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        cv=5, method="isotonic"
    ))
])

numeric_pipeline.fit(X_num_train, y_train)

num_classes   = list(numeric_pipeline.classes_)
num_fraud_idx = num_classes.index(1) if 1 in num_classes else 1

num_probs_test  = numeric_pipeline.predict_proba(X_num_test)[:,  num_fraud_idx]
num_probs_train = numeric_pipeline.predict_proba(X_num_train)[:, num_fraud_idx]

print(f"   Numeric Model ROC-AUC (test): {roc_auc_score(y_test, num_probs_test):.4f}")

# ================================================================
# STEP 6 — WEIGHTED ENSEMBLE  (Text 70% + Numeric 30%)
# ================================================================

ENSEMBLE_WEIGHTS = {"text": 0.70, "numeric": 0.30}

final_probs = (
    text_probs_test * ENSEMBLE_WEIGHTS["text"] +
    num_probs_test  * ENSEMBLE_WEIGHTS["numeric"]
)

print(f"\n   Ensemble    ROC-AUC (test): {roc_auc_score(y_test, final_probs):.4f}")

# ================================================================
# STEP 7 — FIND OPTIMAL THRESHOLD  (maximise Scam F1)
# ================================================================

print("\n🔍 Finding Optimal Decision Threshold...")

best_threshold = 0.5
best_f1        = 0.0

for thresh in np.arange(0.10, 0.90, 0.01):
    preds = (final_probs >= thresh).astype(int)
    f1    = f1_score(y_test, preds, pos_label=1, zero_division=0)
    if f1 > best_f1:
        best_f1        = f1
        best_threshold = round(thresh, 2)

print(f"   🎯 Best Threshold : {best_threshold}  |  Scam F1 = {best_f1:.4f}")

# ================================================================
# STEP 8 — FINAL EVALUATION REPORT
# ================================================================

final_preds = (final_probs >= best_threshold).astype(int)
cm          = confusion_matrix(y_test, final_preds)

print("\n" + "=" * 60)
print("📋  FINAL MODEL PERFORMANCE")
print("=" * 60)
print(f"\n  ✅ Accuracy  : {accuracy_score(y_test, final_preds)*100:.2f}%")
print(f"  ✅ ROC-AUC   : {roc_auc_score(y_test, final_probs):.4f}")
print(f"  ✅ Scam F1   : {best_f1:.4f}")
print(f"  ✅ Threshold : {best_threshold}\n")

print(classification_report(y_test, final_preds, target_names=["Real Job", "Scam Job"]))

print("📊 Confusion Matrix:")
print(f"                   Predicted REAL   Predicted SCAM")
print(f"  Actual REAL           {cm[0][0]:5d}            {cm[0][1]:5d}")
print(f"  Actual SCAM           {cm[1][0]:5d}            {cm[1][1]:5d}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives  (Correct Real) : {tn}")
print(f"  False Positives (Real→Scam)    : {fp}  ← real jobs wrongly flagged")
print(f"  False Negatives (Scam→Real)    : {fn}  ← scam jobs missed")
print(f"  True Positives  (Correct Scam) : {tp}")

# ================================================================
# STEP 9 — SAVE MODEL BUNDLE
# ================================================================

model_bundle = {
    # Both trained pipelines
    "text_pipeline":     text_pipeline,
    "numeric_pipeline":  numeric_pipeline,

    # Meta info needed by app.py
    "numeric_features":  NUMERIC_FEATURES,
    "scam_keywords":     SCAM_KEYWORDS,
    "threshold":         best_threshold,
    "ensemble_weights":  ENSEMBLE_WEIGHTS,
    "classes":           classes,
    "fraud_class_index": fraud_idx,
    "num_fraud_index":   num_fraud_idx,
}

with open("job_scam_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

with open("scam_threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("\n" + "=" * 60)
print("✅  job_scam_model.pkl  — Full model bundle saved!")
print(f"✅  scam_threshold.pkl  — Threshold {best_threshold} saved!")
print("=" * 60)
print("\n👉  Next step:  python app.py\n")
