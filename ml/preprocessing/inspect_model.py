import joblib


PREPROCESSOR_PATH = (
    "ml/models/preprocessor.pkl"
)


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print("=" * 70)
print("PREPROCESSOR INSPECTION")
print("=" * 70)


print(
    "\nPreprocessor type:"
)

print(
    type(preprocessor)
)


print(
    "\nTransformers:"
)

for name, transformer, columns in (
    preprocessor.transformers_
):

    print(
        f"\n{name}"
    )

    print(
        f"Columns: {len(columns)}"
    )

    print(
        columns
    )


print(
    "\nTotal input columns expected:"
)

total = sum(
    len(columns)
    for name, transformer, columns
    in preprocessor.transformers_
)

print(
    total
)