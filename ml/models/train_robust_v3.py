"""
ROBUST FLEXIBLE REVENUE LEAKAGE MODEL V3

Designed for heterogeneous CSV / Excel inputs.

The model is trained and evaluated with simulated
missing-field scenarios so it can handle companies
providing different subsets of billing information.

V3 improvements:
    - Missing-value indicators
    - Missing-field simulation in train AND test
    - Core billing fields protected
    - Same 37-feature interface
    - XGBoost
    - Threshold optimization
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from ml.flexible_features import (
    FLEXIBLE_NUMERIC_FEATURES,
    OPTIONAL_NUMERIC_FEATURES,
    FLEXIBLE_CATEGORICAL_FEATURES,
    FLEXIBLE_DATE_FEATURES,
)

from ml.preprocessing.feature_builder import build_features
from ml.preprocessing.date_features import build_date_features


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    "datasets/training/revenue_leakage_dataset.csv"
)

MODEL_DIR = "ml/models"

MODEL_PATH = (
    f"{MODEL_DIR}/robust_v3_model.pkl"
)

PREPROCESSOR_PATH = (
    f"{MODEL_DIR}/robust_v3_preprocessor.pkl"
)

THRESHOLD_PATH = (
    f"{MODEL_DIR}/robust_v3_threshold.pkl"
)

FEATURES_PATH = (
    f"{MODEL_DIR}/robust_v3_features.pkl"
)

METRICS_PATH = (
    f"{MODEL_DIR}/robust_v3_metrics.pkl"
)

THRESHOLD_RESULTS_PATH = (
    f"{MODEL_DIR}/robust_v3_threshold_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Probability that an optional field is missing
FIELD_MISSING_RATE = 0.35

# Probability that an entire group of fields is missing
GROUP_MISSING_RATE = 0.20


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ROBUST FLEXIBLE REVENUE LEAKAGE MODEL V3")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    keep_default_na=False
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# TARGET
# ============================================================

if "Leakage" not in df.columns:

    raise ValueError(
        "Leakage target column not found."
    )

df["Leakage"] = pd.to_numeric(
    df["Leakage"],
    errors="coerce"
)

df = df.dropna(
    subset=["Leakage"]
)

df["Leakage"] = (
    df["Leakage"]
    .astype(int)
)

print("\nTarget distribution:")
print(
    df["Leakage"].value_counts()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING FEATURES")
print("=" * 70)

df = build_features(df)

df = build_date_features(df)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = (
    FLEXIBLE_NUMERIC_FEATURES
    + OPTIONAL_NUMERIC_FEATURES
    + FLEXIBLE_DATE_FEATURES
)

CATEGORICAL_FEATURES = (
    FLEXIBLE_CATEGORICAL_FEATURES
)

NUMERIC_FEATURES = list(
    dict.fromkeys(
        NUMERIC_FEATURES
    )
)

CATEGORICAL_FEATURES = list(
    dict.fromkeys(
        CATEGORICAL_FEATURES
    )
)

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

print(
    f"\nTotal model features: "
    f"{len(ALL_FEATURES)}"
)


# ============================================================
# CREATE MISSING COLUMNS
# ============================================================

for feature in ALL_FEATURES:

    if feature not in df.columns:

        df[feature] = np.nan


# ============================================================
# X / y
# ============================================================

X = df[
    ALL_FEATURES
].copy()

y = df[
    "Leakage"
].copy()


# ============================================================
# CLEAN NUMERICAL
# ============================================================

for column in NUMERIC_FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# CLEAN CATEGORICAL
# ============================================================

for column in CATEGORICAL_FEATURES:

    X[column] = (
        X[column]
        .astype(str)
        .replace(
            [
                "",
                "nan",
                "None",
                "NaN"
            ],
            np.nan
        )
    )


# ============================================================
# SPLIT
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# CORE FEATURES
# ============================================================

CORE_FEATURES = [

    "Actual_Usage",
    "Billed_Usage",
    "Expected_Amount",
    "Billed_Amount",

]


# ============================================================
# MISSING FIELD SIMULATION
# ============================================================

def simulate_missing_schema(
    X_data,
    random_state
):
    """
    Simulate a company providing an incomplete schema.

    Core billing fields are never removed.
    """

    X_data = X_data.copy()

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Individual optional fields
    # --------------------------------------------------------

    optional_candidates = [

        feature

        for feature in ALL_FEATURES

        if feature not in CORE_FEATURES

    ]

    for feature in optional_candidates:

        mask = (
            rng.random(
                len(X_data)
            )
            < FIELD_MISSING_RATE
        )

        X_data.loc[
            mask,
            feature
        ] = np.nan


    # --------------------------------------------------------
    # Groups of fields that may disappear together
    # --------------------------------------------------------

    groups = [

        [
            "Tax_Expected",
            "Tax_Billed",
            "Tax_Difference",
            "Tax_Billing_Ratio",
        ],

        [
            "Discount_Allowed",
            "Discount_Applied",
            "Discount_Amount",
            "Discount_Difference",
            "Discount_Exceeded",
        ],

        [
            "Contract_Price",
            "Estimated_Usage",
        ],

        [
            "Customer_Type",
            "Industry",
            "Region",
        ],

        [
            "Service_Type",
            "Billing_Cycle",
        ],

        [
            "Currency",
            "Invoice_Status",
        ],

        [
            "Invoice_Year",
            "Invoice_Month",
            "Invoice_Day",
            "Invoice_DayOfWeek",
            "Invoice_Quarter",
        ],

    ]


    for group in groups:

        existing = [

            column

            for column in group

            if column in X_data.columns

            and column not in CORE_FEATURES

        ]

        if not existing:

            continue

        mask = (
            rng.random(
                len(X_data)
            )
            < GROUP_MISSING_RATE
        )

        X_data.loc[
            mask,
            existing
        ] = np.nan


    return X_data


# ============================================================
# SIMULATE TRAIN DATA
# ============================================================

print("\n")
print("=" * 70)
print("SIMULATING MISSING COMPANY FIELDS")
print("=" * 70)

X_train_simulated = simulate_missing_schema(
    X_train,
    random_state=42
)

X_test_simulated = simulate_missing_schema(
    X_test,
    random_state=2026
)

print(
    "\nTraining schema simulation applied."
)

print(
    "Testing schema simulation applied."
)

print(
    f"Individual missing rate: "
    f"{FIELD_MISSING_RATE:.0%}"
)

print(
    f"Group missing rate: "
    f"{GROUP_MISSING_RATE:.0%}"
)


# ============================================================
# MISSING INDICATORS
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING MISSING-VALUE INDICATORS")
print("=" * 70)


def add_missing_indicators(
    X_data
):

    X_data = X_data.copy()

    for feature in ALL_FEATURES:

        indicator_name = (
            f"{feature}_is_missing"
        )

        X_data[
            indicator_name
        ] = X_data[
            feature
        ].isna().astype(int)

    return X_data


X_train_simulated = (
    add_missing_indicators(
        X_train_simulated
    )
)

X_test_simulated = (
    add_missing_indicators(
        X_test_simulated
    )
)


# ============================================================
# FEATURE GROUPS AFTER INDICATORS
# ============================================================

INDICATOR_FEATURES = [

    f"{feature}_is_missing"

    for feature in ALL_FEATURES

]


MODEL_NUMERIC_FEATURES = (
    NUMERIC_FEATURES
    + INDICATOR_FEATURES
)

MODEL_CATEGORICAL_FEATURES = (
    CATEGORICAL_FEATURES
)


# ============================================================
# PREPROCESSOR
# ============================================================

print("\n")
print("=" * 70)
print("FITTING V3 PREPROCESSOR")
print("=" * 70)


numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )

    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            numeric_pipeline,

            MODEL_NUMERIC_FEATURES

        ),

        (
            "categorical",

            categorical_pipeline,

            MODEL_CATEGORICAL_FEATURES

        ),

    ],

    remainder="drop"

)


# ============================================================
# PREPROCESS
# ============================================================

X_train_processed = (
    preprocessor.fit_transform(
        X_train_simulated
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test_simulated
    )
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# XGBOOST
# ============================================================

print("\n")
print("=" * 70)
print("CREATING V3 XGBOOST MODEL")
print("=" * 70)


model = XGBClassifier(

    n_estimators=400,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1,

)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining V3 model...")

model.fit(
    X_train_processed,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = (
    model.predict_proba(
        X_test_processed
    )[:, 1]
)


# ============================================================
# ROC AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(
    "\nROC-AUC:",
    round(
        roc_auc,
        4
    )
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append({

        "Threshold":
            round(
                float(threshold),
                2
            ),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

    })

    print(
        f"{threshold:.2f}  "
        f"Accuracy={accuracy:.4f}  "
        f"Precision={precision:.4f}  "
        f"Recall={recall:.4f}  "
        f"F1={f1:.4f}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df[
        "F1_Score"
    ].idxmax()
]

best_threshold = float(
    best_row["Threshold"]
)


print("\n")
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_row['Accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{best_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_row['F1_Score']:.4f}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 70)
print("V3 ROBUST MODEL PERFORMANCE")
print("=" * 70)

print(
    "\nAccuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1 Score:",
    round(
        f1,
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc,
        4
    )
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "No Leakage",
            "Leakage"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

joblib.dump(
    best_threshold,
    THRESHOLD_PATH
)

joblib.dump(
    {
        "base_features":
            ALL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "indicator_features":
            INDICATOR_FEATURES,

        "model_numeric_features":
            MODEL_NUMERIC_FEATURES,

        "model_categorical_features":
            MODEL_CATEGORICAL_FEATURES,

        "core_features":
            CORE_FEATURES,

    },
    FEATURES_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "Robust Flexible XGBoost V3",

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "threshold":
        float(best_threshold),

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "base_feature_count":
        int(len(ALL_FEATURES)),

    "indicator_feature_count":
        int(len(INDICATOR_FEATURES)),

    "field_missing_rate":
        float(FIELD_MISSING_RATE),

    "group_missing_rate":
        float(GROUP_MISSING_RATE),

}


joblib.dump(
    metrics,
    METRICS_PATH
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("ROBUST FLEXIBLE V3 TRAINING COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(
    f"Model        : {MODEL_PATH}"
)

print(
    f"Preprocessor : {PREPROCESSOR_PATH}"
)

print(
    f"Threshold    : {THRESHOLD_PATH}"
)

print(
    f"Features     : {FEATURES_PATH}"
)

print(
    f"Metrics      : {METRICS_PATH}"
)

print(
    "\nV3 model is ready for sparse-schema testing."
)