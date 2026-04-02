from src.rf_loan.analysis import save_dataset_analysis
from src.rf_loan.config import RandomForestConfig
from src.rf_loan.data import load_dataset, split_features_target
from src.rf_loan.modeling import build_model, train_and_evaluate
from src.rf_loan.reporting import save_dataset_summary
from src.rf_loan.visualization import (
    save_confusion_matrix,
    save_feature_importance_plots,
    save_learning_curve_plot,
    save_precision_recall_curve,
    save_roc_curve,
    save_target_correlation_bar,
    save_validation_curve_plot,
)


def main() -> None:
    config = RandomForestConfig()
    df = load_dataset(config)
    X, y = split_features_target(df, config)

    save_dataset_summary(df=df, target_column=config.target_column, output_path=config.reports_dir / 'dataset_summary_rf.txt')
    save_dataset_analysis(df=df, target_column=config.target_column, output_dir=config.figures_dir)

    model = build_model(config)
    results = train_and_evaluate(X, y, model, config)

    save_confusion_matrix(results['y_test'], results['y_pred'], config.figures_dir / 'confusion_matrix_rf.png')
    save_roc_curve(results['y_test'], results['y_proba'], config.figures_dir / 'roc_curve_rf.png')
    save_precision_recall_curve(results['y_test'], results['y_proba'], config.figures_dir / 'precision_recall_curve_rf.png')
    save_target_correlation_bar(df, config.target_column, config.figures_dir / 'target_correlation_rf.png')
    save_feature_importance_plots(results['best_model'], results['X_test'], results['y_test'], config.figures_dir, config.random_state)
    save_learning_curve_plot(results['best_model'], results['X_train'], results['y_train'], config.figures_dir / 'learning_curve_rf.png', config.random_state)
    save_validation_curve_plot(results['best_model'], results['X_train'], results['y_train'], config.figures_dir / 'validation_curve_rf.png')

    print('Random Forest training complete.')
    print(f"Best parameters: {results['best_params']}")
    print(f'Train metrics: {results["train_metrics"]}')
    print(f'Test metrics: {results["metrics"]}')
    print(f'Outputs saved to: {config.output_dir}')


if __name__ == '__main__':
    main()
