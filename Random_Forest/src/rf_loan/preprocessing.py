import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if 'Experience' in cleaned.columns:
        cleaned['Experience'] = cleaned['Experience'].clip(lower=0)
    return cleaned
