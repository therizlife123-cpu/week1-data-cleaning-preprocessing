# Week 1 — Data Acquisition, Cleaning, and Preprocessing
# UCI Adult Census Income Dataset

# Install dependencies first:
# pip install -r requirements.txt

from ucimlrepo import fetch_ucirepo
import pandas as pd


def iqr_outlier_summary(data, columns):
    """Return an IQR-based outlier summary for numerical columns."""
    rows = []
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (data[col] < lower) | (data[col] > upper)

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_pct": 100 * mask.mean(),
        })

    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)


def main():
    # 1. Acquire the public dataset from the UCI Machine Learning Repository.
    adult = fetch_ucirepo(id=2)
    df = pd.concat([adult.data.features, adult.data.targets], axis=1).copy()

    print("Original shape:", df.shape)
    print("\nOriginal columns:")
    print(df.columns.tolist())

    # 2. Standardize column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # 3. Strip whitespace from text values.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 4. Convert source-specific missing markers to proper missing values.
    df = df.replace({"?": pd.NA, "": pd.NA})

    print("\nMissing values after normalization:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # 5. Inspect duplicate rows.
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")

    # 6. Domain validity checks.
    print("\nDomain checks:")
    print("Age below 17:", (df["age"] < 17).sum())
    print("Age above 100:", (df["age"] > 100).sum())
    print("Non-positive hours-per-week:", (df["hours_per_week"] <= 0).sum())
    print("Non-positive education-num:", (df["education_num"] <= 0).sum())
    print("Negative capital-gain:", (df["capital_gain"] < 0).sum())
    print("Negative capital-loss:", (df["capital_loss"] < 0).sum())

    # 7. Handle missing categorical values explicitly.
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # 8. Handle numerical missing values with the median where necessary.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 9. Normalize the target labels.
    if "income" in df.columns:
        df["income"] = (
            df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        )

    # 10. Outlier audit using the IQR rule.
    outlier_cols = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    outlier_report = iqr_outlier_summary(df, outlier_cols)

    print("\nIQR outlier report:")
    print(outlier_report.to_string(index=False))

    # 11. Final validation.
    assert df.columns.is_unique
    assert df["age"].between(17, 100).all()
    assert (df["hours_per_week"] > 0).all()
    assert df["education_num"].notna().all()
    assert df["income"].isin(["<=50K", ">50K"]).all()

    print("\nFinal shape:", df.shape)
    print("Remaining missing values:", int(df.isna().sum().sum()))
    print("Final duplicate rows:", int(df.duplicated().sum()))
    print("\nTarget distribution:")
    print(df["income"].value_counts(normalize=True))
    print("\nAll core validation checks passed.")


if __name__ == "__main__":
    main()
