from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "Titanic-Dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_DATA = PROCESSED_DIR / "titanic_cleaned.csv"
SUMMARY_FILE = PROCESSED_DIR / "cleaning_summary.txt"


def cap_iqr_outliers(series: pd.Series) -> pd.Series:
    """Cap numeric outliers using the 1.5 IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def min_max_scale(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if np.isclose(max_value, min_value):
        return pd.Series(0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def preprocess_titanic(raw_path: Path = RAW_DATA) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(raw_path)
    cleaned_df = df.copy()
    original_shape = cleaned_df.shape
    original_missing = cleaned_df.isna().sum()

    steps = [
        f"Loaded raw dataset with {original_shape[0]} rows and {original_shape[1]} columns."
    ]

    duplicate_count = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    steps.append(f"Removed {duplicate_count} duplicate rows.")

    cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())
    cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])
    cleaned_df["Cabin_Known"] = cleaned_df["Cabin"].notna().astype(int)
    cleaned_df["Cabin_Deck"] = cleaned_df["Cabin"].fillna("Unknown").astype(str).str[0]
    cleaned_df.loc[cleaned_df["Cabin"].isna(), "Cabin_Deck"] = "Unknown"
    steps.append(
        "Handled missing values: Age=median, Embarked=mode, Cabin converted to useful indicators."
    )

    cleaned_df["FamilySize"] = cleaned_df["SibSp"] + cleaned_df["Parch"] + 1
    cleaned_df["IsAlone"] = (cleaned_df["FamilySize"] == 1).astype(int)
    steps.append("Created FamilySize and IsAlone features.")

    for column in ["Age", "Fare", "SibSp", "Parch", "FamilySize"]:
        cleaned_df[column] = cap_iqr_outliers(cleaned_df[column])
    steps.append("Capped numeric outliers with the 1.5 IQR rule.")

    for column in ["Age", "Fare", "SibSp", "Parch", "FamilySize"]:
        cleaned_df[f"{column}_Norm"] = min_max_scale(cleaned_df[column])
    steps.append("Normalized numeric features with min-max scaling.")

    cleaned_df["Sex"] = cleaned_df["Sex"].str.lower().map({"male": 0, "female": 1})
    embarked_dummies = pd.get_dummies(cleaned_df["Embarked"], prefix="Embarked").astype(int)
    deck_dummies = pd.get_dummies(cleaned_df["Cabin_Deck"], prefix="Deck").astype(int)
    cleaned_df = pd.concat([cleaned_df, embarked_dummies, deck_dummies], axis=1)
    steps.append("Encoded categorical columns: Sex, Embarked, and Cabin deck.")

    drop_columns = ["Name", "Ticket", "Cabin", "Embarked", "Cabin_Deck"]
    cleaned_df = cleaned_df.drop(columns=drop_columns)
    steps.append(f"Dropped high-cardinality/raw text columns: {', '.join(drop_columns)}.")

    missing_after = int(cleaned_df.isna().sum().sum())
    steps.append(
        f"Final cleaned dataset has {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns."
    )
    steps.append(
        f"Missing values before cleaning: {int(original_missing.sum())}; after cleaning: {missing_after}."
    )

    return cleaned_df, steps


def write_outputs(cleaned_df: pd.DataFrame, steps: list[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_DATA, index=False)

    summary_lines = ["Titanic Data Cleaning Summary", "=" * 30, ""]
    summary_lines.extend(f"- {step}" for step in steps)
    summary_lines.extend(["", "Final columns:", ", ".join(cleaned_df.columns)])
    SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    cleaned_df, steps = preprocess_titanic()
    write_outputs(cleaned_df, steps)
    print(f"Saved cleaned dataset to: {CLEAN_DATA}")
    print(f"Saved cleaning summary to: {SUMMARY_FILE}")
    print(f"Missing values in final dataset: {int(cleaned_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
