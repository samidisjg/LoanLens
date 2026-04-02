import pandas as pd

from .config import ProjectConfig


def load_dataset(config: ProjectConfig) -> pd.DataFrame:
    df = pd.read_csv(config.dataset_path)
    df['Experience'] = df['Experience'].clip(lower=0)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    return df


def split_features_target(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[config.target_column] + config.drop_columns)
    y = df[config.target_column]
    return X, y
