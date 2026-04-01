from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_dataset_analysis(df: pd.DataFrame, target_column: str, output_dir: Path) -> None:
    sns.set_theme(style='whitegrid')

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target_column, hue=target_column, dodge=False, palette='Set2', legend=False, ax=ax)
    ax.set_title('Class Distribution of Personal Loan')
    ax.set_xlabel(target_column)
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(output_dir / 'class_distribution_rf.png', dpi=220)
    plt.close(fig)

    analysis_df = df.copy()
    analysis_df['Experience'] = analysis_df['Experience'].clip(lower=0)
    top_features = ['Income', 'CCAvg', 'Mortgage', 'Family']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, feature in zip(axes, top_features):
        sns.boxplot(data=analysis_df, x=target_column, y=feature, ax=ax)
        ax.set_title(f'{feature} vs {target_column}')
    fig.tight_layout()
    fig.savefig(output_dir / 'top_feature_distributions_rf.png', dpi=220)
    plt.close(fig)
