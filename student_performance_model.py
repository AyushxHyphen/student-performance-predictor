"""
Student Academic Performance Predictor
Internship Project — ML Model Training Script
Dataset: Kaggle - Students Performance in Exams
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATA
# ════════════════════════════════════════════════
print("=" * 60)
print("STUDENT ACADEMIC PERFORMANCE PREDICTOR")
print("=" * 60)

df = pd.read_csv('StudentsPerformance.csv')
print(f"\n📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df[['math score','reading score','writing score']].describe().round(2))

# ════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['score_range'] = (df[['math score','reading score','writing score']].max(axis=1)
                     - df[['math score','reading score','writing score']].min(axis=1))
df['math_reading_gap'] = df['math score'] - df['reading score']
df['strong_subject'] = df[['math score','reading score','writing score']].idxmax(axis=1)

# Target: PASS if average >= 65
PASS_THRESHOLD = 65
df['result'] = (df['average_score'] >= PASS_THRESHOLD).astype(int)
print(f"\nPass threshold: {PASS_THRESHOLD} (average of all 3 subjects)")
print(f"Pass: {df['result'].sum()} ({df['result'].mean()*100:.1f}%)")
print(f"Fail: {(df['result']==0).sum()} ({(1-df['result'].mean())*100:.1f}%)")

# Encode categoricals
le_dict = {}
cat_cols = ['gender', 'race/ethnicity', 'parental level of education',
            'lunch', 'test preparation course', 'strong_subject']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  Encoded '{col}': {le_dict[col]}")

# ════════════════════════════════════════════════
# 3. PREPARE FEATURES
# ════════════════════════════════════════════════
feature_cols_full = [
    'gender_enc', 'race/ethnicity_enc', 'parental level of education_enc',
    'lunch_enc', 'test preparation course_enc',
    'math score', 'reading score', 'writing score',
    'score_range', 'math_reading_gap', 'strong_subject_enc'
]
feature_cols_demo = [
    'gender_enc', 'race/ethnicity_enc', 'parental level of education_enc',
    'lunch_enc', 'test preparation course_enc'
]

X_full = df[feature_cols_full]
X_demo = df[feature_cols_demo]
y = df['result']

X_train_f, X_test_f, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y)
X_train_d, X_test_d, _, _ = train_test_split(
    X_demo, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train_f)} | Test: {len(X_test_f)}")

# ════════════════════════════════════════════════
# 4. TRAIN MODELS
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Model 1: Random Forest (Full Features)
print("\n[1] Random Forest — Full Features")
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             random_state=42, class_weight='balanced')
rf.fit(X_train_f, y_train)
y_pred_rf = rf.predict(X_test_f)
y_prob_rf = rf.predict_proba(X_test_f)[:, 1]
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy: {acc_rf:.4f} ({acc_rf*100:.1f}%)")
print(classification_report(y_test, y_pred_rf, target_names=['Fail','Pass']))

# Model 2: Gradient Boosting (Demo / Demographics)
print("[2] Gradient Boosting — Demographics Only")
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                 max_depth=4, random_state=42)
gb.fit(X_train_d, y_train)
y_pred_gb = gb.predict(X_test_d)
y_prob_gb = gb.predict_proba(X_test_d)[:, 1]
acc_gb = accuracy_score(y_test, y_pred_gb)
cv = cross_val_score(gb, X_demo, y, cv=5)
print(f"  Accuracy: {acc_gb:.4f} ({acc_gb*100:.1f}%)")
print(f"  5-fold CV: {cv.round(3)} | Mean: {cv.mean():.3f}")
print(classification_report(y_test, y_pred_gb, target_names=['Fail','Pass']))

# Model 3: Logistic Regression (baseline)
print("[3] Logistic Regression — Baseline")
lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
lr.fit(X_train_f, y_train)
y_pred_lr = lr.predict(X_test_f)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy: {acc_lr:.4f} ({acc_lr*100:.1f}%)")

# ════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Student Performance Prediction — Model Analysis', fontsize=16, fontweight='bold', y=0.98)

# (a) Pass/Fail distribution
ax = axes[0, 0]
counts = df['result'].value_counts()
colors = ['#d94f4f', '#1a9e6e']
bars = ax.bar(['Fail', 'Pass'], [counts[0], counts[1]], color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Pass / Fail Distribution', fontweight='bold')
ax.set_ylabel('Number of Students')
ax.set_ylim(0, max(counts) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) Score distributions
ax = axes[0, 1]
for col, color, label in [('math score','#2a6ef5','Math'),
                           ('reading score','#1a9e6e','Reading'),
                           ('writing score','#e07a20','Writing')]:
    ax.hist(df[col], bins=25, alpha=0.55, color=color, label=label, edgecolor='none')
ax.axvline(PASS_THRESHOLD, color='#d94f4f', linestyle='--', linewidth=2, label=f'Pass ({PASS_THRESHOLD})')
ax.set_title('Score Distributions', fontweight='bold')
ax.set_xlabel('Score')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) Confusion Matrix — RF Full
ax = axes[0, 2]
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm_rf, display_labels=['Fail','Pass'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — Random Forest\n(Accuracy: {acc_rf*100:.1f}%)', fontweight='bold')

# (d) Feature Importance
ax = axes[1, 0]
feat_labels_display = ['Gender','Ethnicity','Parent Edu','Lunch','Test Prep',
                        'Math','Reading','Writing','Score Range','Math-Read Gap','Strong Subj']
fi = rf.feature_importances_
sorted_idx = np.argsort(fi)
ax.barh([feat_labels_display[i] for i in sorted_idx], fi[sorted_idx],
        color='#2a6ef5', edgecolor='none', height=0.7)
ax.set_title('Feature Importances (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (e) ROC Curves
ax = axes[1, 1]
for y_prob, name, color in [(y_prob_rf, 'Random Forest (Full)', '#2a6ef5'),
                              (y_prob_gb, 'Gradient Boosting (Demo)', '#1a9e6e'),
                              (lr.predict_proba(X_test_f)[:,1], 'Logistic Regression', '#e07a20')]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (f) Model Comparison
ax = axes[1, 2]
models = ['Random Forest\n(Full)', 'Gradient Boosting\n(Demo)', 'Logistic Reg\n(Full)']
accs = [acc_rf, acc_gb, acc_lr]
colors_bar = ['#2a6ef5', '#1a9e6e', '#e07a20']
bars = ax.bar(models, [a*100 for a in accs], color=colors_bar, width=0.5, edgecolor='white', linewidth=1.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Model Accuracy Comparison', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 110)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/claude/model_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: model_analysis.png")

# ════════════════════════════════════════════════
# 6. SAVE MODELS
# ════════════════════════════════════════════════
joblib.dump(rf, 'model_full.pkl')
joblib.dump(gb, 'model_demo.pkl')
joblib.dump(lr, 'model_lr.pkl')

metadata = {
    'le_dict': {k: {str(c): int(i) for c,i in v.items()} for k,v in le_dict.items()},
    'feature_cols_full': feature_cols_full,
    'feature_cols_demo': feature_cols_demo,
    'accuracy_full': round(acc_rf, 4),
    'accuracy_demo': round(acc_gb, 4),
    'accuracy_lr': round(acc_lr, 4),
    'cv_mean_demo': round(float(cv.mean()), 4),
    'pass_threshold': PASS_THRESHOLD,
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ All models saved: model_full.pkl, model_demo.pkl, model_lr.pkl")
print("✅ Metadata saved: model_metadata.json")

# ════════════════════════════════════════════════
# 7. PREDICTION FUNCTION
# ════════════════════════════════════════════════
def predict_student(gender, race_ethnicity, parent_education, lunch, test_prep,
                    math_score=None, reading_score=None, writing_score=None,
                    model='full'):
    """
    Predict if a student will pass.
    Returns: {'prediction': 'Pass'/'Fail', 'probability': float, 'confidence': float}
    """
    gender_enc = le_dict['gender'][gender]
    race_enc = le_dict['race/ethnicity'][race_ethnicity]
    edu_enc = le_dict['parental level of education'][parent_education]
    lunch_enc = le_dict['lunch'][lunch]
    prep_enc = le_dict['test preparation course'][test_prep]

    if model == 'full' and all(v is not None for v in [math_score, reading_score, writing_score]):
        avg = (math_score + reading_score + writing_score) / 3
        score_range = max(math_score, reading_score, writing_score) - min(math_score, reading_score, writing_score)
        math_reading_gap = math_score - reading_score
        scores = {'math score': math_score, 'reading score': reading_score, 'writing score': writing_score}
        strong_enc = le_dict['strong_subject'][max(scores, key=scores.get)]
        features = [[gender_enc, race_enc, edu_enc, lunch_enc, prep_enc,
                     math_score, reading_score, writing_score,
                     score_range, math_reading_gap, strong_enc]]
        clf = rf
    else:
        features = [[gender_enc, race_enc, edu_enc, lunch_enc, prep_enc]]
        clf = gb

    prob = clf.predict_proba(features)[0]
    pred = 'Pass' if prob[1] >= 0.5 else 'Fail'
    confidence = max(prob)
    return {'prediction': pred, 'probability_pass': round(float(prob[1]), 4), 'confidence': round(float(confidence), 4)}

# Example predictions
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)
examples = [
    ("female", "group C", "bachelor's degree", "standard", "completed", 78, 82, 80),
    ("male", "group A", "some high school", "free/reduced", "none", 42, 38, 40),
    ("female", "group D", "some college", "standard", "none", 65, 70, 68),
]
for ex in examples:
    result = predict_student(*ex)
    print(f"  {ex[0].title()}, {ex[1]}, {ex[2]}, {ex[3]}, prep={ex[4]}, "
          f"scores=({ex[5]},{ex[6]},{ex[7]}) → {result['prediction']} "
          f"(confidence: {result['confidence']*100:.0f}%)")

print("\n🎉 All done!")
