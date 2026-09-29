"""
6001CMD - UCI Bank Marketing
Updated data-quality investigation and leakage-safe preprocessing workflow.

Key corrections compared with the previous version:
1) Explicitly handles pdays == 999 as a sentinel meaning "not previously contacted".
2) Reports duplicates, implicit missingness, skewness and pdays consistency evidence.
3) Removes duration before modelling to prevent temporal leakage.
4) Performs stratified train/test split BEFORE fitting data-dependent preprocessing.
5) Fits imputation/scaling/encoding on training data only.
6) Uses SMOTENC (mixed numerical + categorical data) instead of ordinary SMOTE after OHE.
7) Applies log1p to campaign as an optional skewness treatment.
8) Fits PCA only on the training representation; the untouched test set is only transformed.
9) Generates an updated Figure 4.4 workflow diagram.
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 42

# ------------------------------------------------------------
# 0. Plot settings
# ------------------------------------------------------------
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Download and load UCI Bank Marketing dataset
# ============================================================
print("1. Loading UCI Bank Marketing dataset...")

url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
zip_path = "bank_marketing.zip"
extract_dir = "bank_extracted"
final_dir = "bank_final"

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

inner_zip = os.path.join(extract_dir, "bank-additional.zip")
if not os.path.exists(final_dir):
    with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
        zip_ref.extractall(final_dir)

csv_path = os.path.join(
    final_dir, "bank-additional", "bank-additional-full.csv"
)
df = pd.read_csv(csv_path, sep=';')
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# 2. Task 3 - Data quality investigation and figures
# ============================================================
print("\n2. Task 3 - Data quality investigation")

# 2.1 Target imbalance -------------------------------------------------
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True).mul(100)
print("\n[Target distribution]")
for cls in target_counts.index:
    print(f"{cls}: {target_counts[cls]:,} ({target_pct[cls]:.2f}%)")

plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='y', hue='y', legend=False)
plt.title('Target Variable Distribution (Class Imbalance)', fontsize=12)
plt.xlabel('Subscribed Term Deposit (y)')
plt.ylabel('Count')
for p in ax.patches:
    h = p.get_height()
    ax.annotate(
        f'{int(h):,} ({h / len(df) * 100:.1f}%)',
        (p.get_x() + p.get_width() / 2, h),
        ha='center', va='bottom', xytext=(0, 5),
        textcoords='offset points'
    )
plt.tight_layout()
plt.savefig('Figure_3_1_target_imbalance.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Outliers / skewness ----------------------------------------------
raw_num_cols = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

plt.figure(figsize=(15, 8))
for i, col in enumerate(['age', 'duration', 'campaign', 'euribor3m'], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='y', y=col, hue='y', legend=False)
    plt.title(f'Boxplot of {col} by Target (y)')
plt.tight_layout()
plt.savefig('Figure_3_2_numerical_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[Skewness]")
print(df[raw_num_cols].skew().sort_values(ascending=False).round(3))

# 2.3 Correlation -------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[raw_num_cols].corr(), annot=True, fmt='.2f',
    cmap='coolwarm', vmin=-1, vmax=1
)
plt.title('Correlation Heatmap of Numerical Features', fontsize=12)
plt.tight_layout()
plt.savefig('Figure_3_3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.4 Implicit missing values -----------------------------------------
print("\n[Implicit missing values: 'unknown']")
unknown_counts = (df == 'unknown').sum()
for col, count in unknown_counts[unknown_counts > 0].items():
    print(f"{col}: {count:,} ({count / len(df) * 100:.2f}%)")

# 2.5 Duplicate and consistency checks --------------------------------
duplicate_count = int(df.duplicated().sum())
print("\n[Duplicate records]")
print(
    f"Duplicates: {duplicate_count:,} "
    f"({duplicate_count / len(df) * 100:.4f}%)"
)

# IMPORTANT: in bank-additional-full.csv, pdays == 999 means
# the client was NOT previously contacted; it is NOT an ordinary 999-day value.
pdays_999_count = int((df['pdays'] == 999).sum())
print("\n[pdays sentinel check]")
print(
    f"pdays = 999: {pdays_999_count:,} "
    f"({pdays_999_count / len(df) * 100:.2f}%)"
)

# Cross-variable consistency evidence.
# If previous == 0, poutcome is normally 'nonexistent'.
inconsistent_previous = int(
    ((df['pdays'] == 999) & (df['previous'] > 0)).sum()
)
inconsistent_poutcome = int(
    ((df['previous'] == 0) & (df['poutcome'] != 'nonexistent')).sum()
)
print("\n[Cross-variable consistency checks]")
print(f"pdays=999 but previous>0: {inconsistent_previous}")
print(
    "previous=0 but poutcome!='nonexistent': "
    f"{inconsistent_poutcome}"
)

# Basic range checks: these help distinguish genuine outliers from invalid data.
print("\n[Basic range checks]")
print(f"age <= 0: {(df['age'] <= 0).sum()}")
print(f"duration < 0: {(df['duration'] < 0).sum()}")
print(f"campaign < 1: {(df['campaign'] < 1).sum()}")
print(f"previous < 0: {(df['previous'] < 0).sum()}")


# ============================================================
# 3. Task 4 - Leakage-safe preprocessing workflow
# ============================================================
print("\n3. Task 4 - Leakage-safe preprocessing")

# 3.1 Deterministic cleaning ------------------------------------------
# Duplicate removal does not estimate parameters from the target/test data.
df_clean = df.drop_duplicates().copy()

# Convert the UCI string 'unknown' to explicit missing values.
df_clean.replace('unknown', np.nan, inplace=True)

# Explicit semantic handling of pdays == 999.
# 999 means "not previously contacted", not 999 elapsed days.
df_clean['previously_contacted'] = (df_clean['pdays'] != 999).astype(int)

# Keep elapsed days meaningful only for clients who were contacted before.
# For never-contacted clients, 0 is a neutral placeholder used together with
# the binary indicator above; it is NOT interpreted as an actual 0-day interval.
df_clean['pdays_since_previous_contact'] = np.where(
    df_clean['pdays'] == 999,
    0,
    df_clean['pdays']
)
df_clean.drop(columns=['pdays'], inplace=True)

# Remove duration BEFORE modelling because it is unavailable pre-call and
# would create temporal leakage in a real customer-targeting system.
X = df_clean.drop(columns=['y', 'duration'])
y = df_clean['y'].map({'no': 0, 'yes': 1})

print(f"Rows after duplicate removal: {len(df_clean):,}")
print("Removed from predictors: duration")
print("Derived: previously_contacted, pdays_since_previous_contact")

# 3.2 Split BEFORE fitting data-dependent preprocessing ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# campaign is strongly right-skewed; log1p is evaluated as a model-specific
# transformation. duration is NOT log-transformed because it has been removed.
log_cols = ['campaign']
other_num_cols = [c for c in num_cols if c not in log_cols]

# 3.3 Fit imputation + scaling on TRAIN ONLY ---------------------------
# We first create a mixed-type training DataFrame for SMOTENC.
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train_stage = X_train.copy()
X_test_stage = X_test.copy()

X_train_stage[other_num_cols] = num_imputer.fit_transform(
    X_train[other_num_cols]
)
X_test_stage[other_num_cols] = num_imputer.transform(
    X_test[other_num_cols]
)

# Separate imputer for campaign before log1p.
campaign_imputer = SimpleImputer(strategy='median')
X_train_stage[log_cols] = campaign_imputer.fit_transform(X_train[log_cols])
X_test_stage[log_cols] = campaign_imputer.transform(X_test[log_cols])
X_train_stage['campaign'] = np.log1p(X_train_stage['campaign'])
X_test_stage['campaign'] = np.log1p(X_test_stage['campaign'])

X_train_stage[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test_stage[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# Standardise numerical features using parameters learned from training only.
scaler = StandardScaler()
X_train_stage[num_cols] = scaler.fit_transform(X_train_stage[num_cols])
X_test_stage[num_cols] = scaler.transform(X_test_stage[num_cols])

# 3.4 Training-only mixed-data oversampling ----------------------------
# SMOTENC is preferred here because the dataset contains both numerical and
# categorical predictors. Unlike ordinary SMOTE after one-hot encoding,
# SMOTENC avoids interpolating fractional dummy-category states.
smotenc = SMOTENC(
    categorical_features=cat_cols,
    sampling_strategy='auto',   # tune in CV if a non-1:1 target is preferred
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_balanced, y_train_balanced = smotenc.fit_resample(
    X_train_stage,
    y_train
)

print("\n[Training-only SMOTENC]")
print(f"Before: {X_train_stage.shape}, positive rate={y_train.mean():.4f}")
print(
    f"After:  {X_train_balanced.shape}, "
    f"positive rate={np.mean(y_train_balanced):.4f}"
)
print(f"Test remains untouched by resampling: {X_test_stage.shape}")

# 3.5 One-hot encode categorical features AFTER SMOTENC ----------------
# Numerical variables are already scaled. The encoder is fitted on the
# resampled TRAINING data only and then applied unchanged to the test data.
model_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ],
    remainder='drop'
)

X_train_model = model_preprocessor.fit_transform(X_train_balanced)
X_test_model = model_preprocessor.transform(X_test_stage)

print("\n[Final model matrices]")
print(f"X_train_model: {X_train_model.shape}")
print(f"X_test_model:  {X_test_model.shape}")

# 3.6 PCA experiment: TRAIN-ONLY fit -----------------------------------
# PCA is evaluated as an alternative dimensionality-reduction experiment,
# not assumed to be the final choice. Tree-based/supervised feature selection
# can be preferred when interpretability is more important.
X_train_dense = (
    X_train_model.toarray()
    if hasattr(X_train_model, 'toarray')
    else np.asarray(X_train_model)
)
X_test_dense = (
    X_test_model.toarray()
    if hasattr(X_test_model, 'toarray')
    else np.asarray(X_test_model)
)

pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_dense)
X_test_pca = pca.transform(X_test_dense)

print("\n[PCA experiment - 95% explained variance]")
print(f"Original encoded features: {X_train_dense.shape[1]}")
print(f"PCA components retained:   {X_train_pca.shape[1]}")
print(f"Test PCA shape:            {X_test_pca.shape}")


# ============================================================
# 4. Figure 4.4 - Updated end-to-end preprocessing workflow
# ============================================================
def draw_workflow_figure(filename='Figure_4_4_leakage_safe_workflow_FINAL.png'):
    """Generate the final two-branch, leakage-safe workflow at 300 dpi."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15)
    ax.axis('off')

    def box(x, y0, text, fs=10):
        ax.text(
            x, y0, text, ha='center', va='center', fontsize=fs,
            bbox=dict(boxstyle='round,pad=0.55', fc='white', ec='black', lw=1.2)
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', lw=1.3)
        )

    ax.text(
        7, 14.65,
        'Figure 4.4. Leakage-Safe End-to-End Data Preprocessing Workflow',
        ha='center', va='center', fontsize=13, fontweight='bold'
    )

    # Shared stages before the split
    box(7, 13.5, 'Raw bank-additional-full.csv\n41,188 observations | 20 predictors + y')
    box(7, 11.9, 'Deterministic data-quality handling\nDrop duplicates | unknown -> missing')
    box(
        7, 10.2,
        'Semantic feature engineering & leakage prevention\n'
        'pdays=999 -> previously_contacted + elapsed-days feature\n'
        'Drop duration before modelling'
    )
    box(
        7, 8.45,
        'Define X / y and STRATIFIED TRAIN-TEST SPLIT\n'
        'No data-dependent preprocessing fitted before this point'
    )

    arrow(7, 12.95, 7, 12.45)
    arrow(7, 11.35, 7, 10.85)
    arrow(7, 9.55, 7, 9.05)

    ax.text(4.0, 7.75, 'TRAIN BRANCH', ha='center', va='center',
            fontsize=10.5, fontweight='bold')
    ax.text(11.0, 7.75, 'HELD-OUT TEST BRANCH', ha='center', va='center',
            fontsize=10.5, fontweight='bold')

    arrow(6.4, 7.95, 4.0, 7.2)
    arrow(7.6, 7.95, 11.0, 7.2)

    # Training branch
    box(
        4.0, 6.5,
        'Fit on TRAIN only\nMedian / most-frequent imputation\n'
        'log1p(campaign) | StandardScaler', fs=9.6
    )
    box(
        4.0, 4.8,
        'TRAIN-only imbalance treatment\n'
        'SMOTENC for mixed numerical + categorical predictors\n'
        'Tune sampling strategy using training/CV', fs=9.4
    )
    box(
        4.0, 3.1,
        'TRAIN-only representation / feature strategy\n'
        'One-Hot Encoding\nPCA experiment or supervised feature selection', fs=9.4
    )
    box(
        4.0, 1.45,
        'Model training & validation\n'
        'Logistic Regression / Random Forest / XGBoost', fs=9.6
    )

    arrow(4.0, 5.9, 4.0, 5.4)
    arrow(4.0, 4.2, 4.0, 3.7)
    arrow(4.0, 2.5, 4.0, 2.05)

    # Held-out test branch
    box(
        11.0, 6.5,
        'Hold out from fitting and resampling\nPreserve original class distribution', fs=9.6
    )
    box(
        11.0, 4.8,
        'Apply TRAIN-fitted transformations only\n'
        'Same imputation, scaling, encoding\nand selected feature mapping', fs=9.4
    )
    arrow(11.0, 5.9, 11.0, 5.4)

    # Final evaluation receives both the trained model and transformed test set
    box(
        8.0, 1.45,
        'Final held-out test evaluation\nPrecision | Recall | F1 | ROC-AUC | PR-AUC', fs=9.8
    )
    arrow(6.35, 1.45, 6.9, 1.45)
    arrow(11.0, 4.2, 8.55, 2.05)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Workflow figure saved: {filename}')


draw_workflow_figure()

print('\nDone. Updated figures and preprocessing outputs have been generated.')
