from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RandomForestConfig:
    dataset_path: Path = Path('dataset/Bank_Personal_Loan_Modelling.csv')
    output_dir: Path = Path('outputs')
    figures_dir: Path = Path('outputs/figures')
    reports_dir: Path = Path('outputs/reports')
    artifacts_dir: Path = Path('artifacts')
    model_dir: Path = Path('artifacts/model')
    metadata_dir: Path = Path('artifacts/metadata')
    target_column: str = 'Personal Loan'
    drop_columns: list[str] = field(default_factory=lambda: ['ID', 'ZIP Code'])
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = 'f1'
    param_grid: dict = field(
        default_factory=lambda: {
            'n_estimators': [200],
            'max_depth': [6, 7],
            'min_samples_split': [15, 20],
            'min_samples_leaf': [6, 8],
            'max_features': ['sqrt'],
            'class_weight': [None, 'balanced'],
        }
    )
