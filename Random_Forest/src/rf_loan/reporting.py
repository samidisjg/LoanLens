from pathlib import Path

import pandas as pd


def save_dataset_summary(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    lines = [
        'Random Forest Dataset Summary',
        '=' * 30,
        f'Shape: {df.shape}',
        '',
        'Missing Values:',
        df.isna().sum().to_string(),
        '',
        'Target Distribution:',
        df[target_column].value_counts().to_string(),
        '',
        'Descriptive Statistics:',
        df.describe().transpose().to_string(),
    ]
    output_path.write_text('\n'.join(lines), encoding='utf-8')
