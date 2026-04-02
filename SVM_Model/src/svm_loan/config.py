from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    dataset_path: Path = Path('dataset/Bank_Personal_Loan_Modelling.csv')
    output_dir: Path = Path('outputs/base_svm')
    figures_dir: Path = Path('outputs/base_svm/figures')
    reports_dir: Path = Path('outputs/base_svm/reports')
    artifacts_dir: Path = Path('artifacts/base_svm')
    model_dir: Path = Path('artifacts/base_svm/model')
    metadata_dir: Path = Path('artifacts/base_svm/metadata')
    target_column: str = 'Personal Loan'
    drop_columns: list[str] = field(default_factory=lambda: ['ID'])
    numeric_features: list[str] = field(default_factory=lambda: ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage'])
    categorical_features: list[str] = field(default_factory=lambda: [
        'ZIP Code', 'Family', 'Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard'
    ])
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = 'f1'
    svm_param_grid: dict = field(default_factory=lambda: {
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
    })
