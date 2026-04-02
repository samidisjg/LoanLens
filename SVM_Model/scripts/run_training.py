from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.svm_loan.config import ProjectConfig
from src.svm_loan.data import load_dataset, split_features_target
from src.svm_loan.modeling import build_model, train_and_evaluate
from src.svm_loan.visualization import (
    save_confusion_matrix,
    save_feature_importance,
    save_learning_curve_plot,
    save_roc_curve,
    save_train_test_metrics_plot,
    save_validation_curve_plot,
)


def main() -> None:
    config = ProjectConfig()
    df = load_dataset(config)
    X, y = split_features_target(df, config)
    model = build_model(config)
    results = train_and_evaluate(X, y, model, config)

    save_confusion_matrix(results['y_test'], results['y_pred'], config.figures_dir / 'confusion_matrix.png')
    save_roc_curve(results['y_test'], results['y_proba'], config.figures_dir / 'roc_curve.png')
    save_feature_importance(results['best_model'], results['X_test'], results['y_test'], config.figures_dir / 'feature_importance.png', config.random_state)
    save_learning_curve_plot(results['best_model'], results['X_train'], results['y_train'], config.figures_dir / 'learning_curve_svm.png', config.random_state)
    save_validation_curve_plot(results['best_model'], results['X_train'], results['y_train'], config.figures_dir / 'validation_curve_svm.png')
    save_train_test_metrics_plot(results['train_metrics'], results['metrics'], config.figures_dir / 'train_test_metrics_svm.png')

    print('Training complete.')
    print(f"Best parameters: {results['best_params']}")
    print(f'Train metrics: {results["train_metrics"]}')
    print(f'Test metrics: {results["metrics"]}')
    print(f'Metrics saved to: {config.output_dir}')


if __name__ == '__main__':
    main()
