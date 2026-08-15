import pandas as pd

df = pd.read_csv(
    "datasets/revenue_leakage_dataset.csv",
    keep_default_na=False
)

print("\n========== MISSING VALUES BY COLUMN ==========\n")

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

print("\nTotal missing values:", df.isnull().sum().sum())