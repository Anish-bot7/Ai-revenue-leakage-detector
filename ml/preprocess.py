import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "datasets/revenue_leakage_dataset.csv"

PREPROCESSOR_PATH = "ml/preprocessor.pkl"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False
    )

    print("\nDataset loaded successfully.")

    print("Shape:", df.shape)

    return df


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def inspect_data(df):

    print("\n========== DATA QUALITY CHECK ==========")

    print("\nShape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nTarget Distribution:")

    if "Leakage" in df.columns:
        print(df["Leakage"].value_counts())

    print("\nData Types:")
    print(df.dtypes)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print(
        f"\nRemoved {before - after} duplicate rows."
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    print("\n========== FEATURE ENGINEERING ==========")

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    date_columns = [
        "Invoice_Date",
        "Due_Date",
        "Registration_Date"
    ]

    for column in date_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Invoice Date Features
    # --------------------------------------------------------

    if "Invoice_Date" in df.columns:

        df["Invoice_Year"] = (
            df["Invoice_Date"].dt.year
        )

        df["Invoice_Month"] = (
            df["Invoice_Date"].dt.month
        )

        df["Invoice_Day"] = (
            df["Invoice_Date"].dt.day
        )

        df["Invoice_DayOfWeek"] = (
            df["Invoice_Date"].dt.dayofweek
        )

        df["Invoice_Quarter"] = (
            df["Invoice_Date"].dt.quarter
        )

    # --------------------------------------------------------
    # Due Date Features
    # --------------------------------------------------------

    if "Due_Date" in df.columns:

        df["Due_Year"] = (
            df["Due_Date"].dt.year
        )

        df["Due_Month"] = (
            df["Due_Date"].dt.month
        )

        df["Due_Day"] = (
            df["Due_Date"].dt.day
        )

    # --------------------------------------------------------
    # Invoice Due Period
    # --------------------------------------------------------

    if "Invoice_Date" in df.columns and "Due_Date" in df.columns:

        df["Payment_Term_Days"] = (
            df["Due_Date"] - df["Invoice_Date"]
        ).dt.days

    # --------------------------------------------------------
    # Customer Tenure
    # --------------------------------------------------------

    if (
        "Registration_Date" in df.columns
        and "Invoice_Date" in df.columns
    ):

        df["Customer_Tenure_Days"] = (
            df["Invoice_Date"]
            - df["Registration_Date"]
        ).dt.days

        df["Customer_Tenure_Years"] = (
            df["Customer_Tenure_Days"] / 365
        )

    # --------------------------------------------------------
    # Usage Difference
    # --------------------------------------------------------

    if (
        "Actual_Usage" in df.columns
        and "Billed_Usage" in df.columns
    ):

        df["Usage_Difference"] = (
            df["Actual_Usage"]
            - df["Billed_Usage"]
        )

        df["Usage_Billing_Ratio"] = np.where(
            df["Actual_Usage"] != 0,
            df["Billed_Usage"]
            / df["Actual_Usage"],
            0
        )

    # --------------------------------------------------------
    # Expected vs Billed Amount
    # --------------------------------------------------------

    if (
        "Expected_Amount" in df.columns
        and "Billed_Amount" in df.columns
    ):

        df["Amount_Difference"] = (
            df["Expected_Amount"]
            - df["Billed_Amount"]
        )

        df["Billing_Ratio"] = np.where(
            df["Expected_Amount"] != 0,
            df["Billed_Amount"]
            / df["Expected_Amount"],
            0
        )

    # --------------------------------------------------------
    # Tax Difference
    # --------------------------------------------------------

    if (
        "Tax_Expected" in df.columns
        and "Tax_Billed" in df.columns
    ):

        df["Tax_Difference"] = (
            df["Tax_Expected"]
            - df["Tax_Billed"]
        )

        df["Tax_Billing_Ratio"] = np.where(
            df["Tax_Expected"] != 0,
            df["Tax_Billed"]
            / df["Tax_Expected"],
            0
        )

    # --------------------------------------------------------
    # Discount Difference
    # --------------------------------------------------------

    if (
        "Discount_Allowed" in df.columns
        and "Discount_Applied" in df.columns
    ):

        df["Discount_Difference"] = (
            df["Discount_Applied"]
            - df["Discount_Allowed"]
        )

        df["Discount_Exceeded"] = np.where(
            df["Discount_Applied"]
            > df["Discount_Allowed"],
            1,
            0
        )

    # --------------------------------------------------------
    # Price Difference
    # --------------------------------------------------------

    if (
        "Expected_Amount" in df.columns
        and "Actual_Usage" in df.columns
        and "Billed_Amount" in df.columns
        and "Billed_Usage" in df.columns
    ):

        df["Expected_Unit_Revenue"] = np.where(
            df["Actual_Usage"] != 0,
            df["Expected_Amount"]
            / df["Actual_Usage"],
            0
        )

        df["Actual_Unit_Revenue"] = np.where(
            df["Billed_Usage"] != 0,
            df["Billed_Amount"]
            / df["Billed_Usage"],
            0
        )

        df["Unit_Revenue_Difference"] = (
            df["Expected_Unit_Revenue"]
            - df["Actual_Unit_Revenue"]
        )

    # --------------------------------------------------------
    # Usage Utilization
    # --------------------------------------------------------

    if (
        "Actual_Usage" in df.columns
        and "Estimated_Usage" in df.columns
    ):

        df["Usage_Utilization"] = np.where(
            df["Estimated_Usage"] != 0,
            df["Actual_Usage"]
            / df["Estimated_Usage"],
            0
        )

    # --------------------------------------------------------
    # Contract Price Difference
    # --------------------------------------------------------

    if (
        "Contract_Price" in df.columns
        and "Expected_Amount" in df.columns
    ):

        df["Contract_Amount_Difference"] = (
            df["Contract_Price"]
            - df["Expected_Amount"]
        )

    # --------------------------------------------------------
    # Peak Usage Ratio
    # --------------------------------------------------------

    if (
        "Peak_Usage" in df.columns
        and "Actual_Usage" in df.columns
    ):

        df["Peak_Usage_Ratio"] = np.where(
            df["Actual_Usage"] != 0,
            df["Peak_Usage"]
            / df["Actual_Usage"],
            0
        )

    # --------------------------------------------------------
    # Off-Peak Usage Ratio
    # --------------------------------------------------------

    if (
        "OffPeak_Usage" in df.columns
        and "Actual_Usage" in df.columns
    ):

        df["OffPeak_Usage_Ratio"] = np.where(
            df["Actual_Usage"] != 0,
            df["OffPeak_Usage"]
            / df["Actual_Usage"],
            0
        )

    # --------------------------------------------------------
    # Remove original date columns
    # --------------------------------------------------------

    df = df.drop(
        columns=[
            "Invoice_Date",
            "Due_Date",
            "Registration_Date"
        ],
        errors="ignore"
    )

    return df


# ============================================================
# REMOVE DATA LEAKAGE / UNNECESSARY COLUMNS
# ============================================================

def remove_unusable_columns(df):

    columns_to_remove = [

    # -------------------------
    # Identifiers
    # -------------------------

    "Invoice_ID",
    "Customer_ID",
    "Contract_ID",

    # -------------------------
    # Personal Information
    # -------------------------

    "Customer_Name",
    "Email",
    "Phone",

    # -------------------------
    # Target
    # -------------------------

    "Leakage",
    "Leakage_Type",
    "Leakage_Amount",

    # -------------------------
    # Direct Leakage Indicators
    # -------------------------

    "Missing_GST",
    "Usage_Mismatch",
    "Extra_Discount",
    "Wrong_Pricing",
    "Missing_Charge",
    "Duplicate_Invoice",
    "Contract_Violation",
    "Late_Billing",

    # -------------------------
    # Information Not Available
    # At Prediction Time
    # -------------------------

    "Payment_Status",

    # -------------------------
    # High Cardinality
    # -------------------------

    "City",
    "Account_Manager",
    "Plan_Name"

]
    existing_columns = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    print("\n========== REMOVING COLUMNS ==========")

    print("\nRemoved columns:")

    for column in existing_columns:
        print(" -", column)

    df = df.drop(
        columns=existing_columns
    )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\n========== FEATURES ==========")

    print("\nNumerical Features:")

    for column in numeric_columns:
        print(" -", column)

    print("\nCategorical Features:")

    for column in categorical_columns:
        print(" -", column)

    # --------------------------------------------------------
    # Numerical Pipeline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Categorical Pipeline
    # --------------------------------------------------------

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
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]
    )

    # --------------------------------------------------------
    # Combine Pipelines
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                numeric_pipeline,
                numeric_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )

        ]
    )

    return preprocessor


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def prepare_data():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    inspect_data(df)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(df)

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    df = create_features(df)

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    if "Leakage" not in pd.read_csv(DATA_PATH).columns:
        raise ValueError(
            "Target column 'Leakage' not found."
        )

    # Re-load target before removing it
    original_df = pd.read_csv(DATA_PATH)

    y = original_df["Leakage"]

    # Make sure duplicate removal doesn't misalign target
    y = y.loc[
        df.index
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Remove unwanted columns
    # --------------------------------------------------------

    df = remove_unusable_columns(df)

    X = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\n========== DATA SPLIT ==========")

    print("Training samples :", len(X_train))

    print("Testing samples  :", len(X_test))

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print(
        f"\nPreprocessor saved to: "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        "\nProcessed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    prepare_data()