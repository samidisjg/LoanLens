import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd


MODEL_PATH = Path('artifacts/smote_rf/model/smote_random_forest_model.joblib')
FEATURES_PATH = Path('artifacts/smote_rf/metadata/feature_columns_smote_rf.json')


def load_feature_columns() -> list[str]:
    return json.loads(FEATURES_PATH.read_text(encoding='utf-8'))['feature_columns']


def build_sample_input() -> pd.DataFrame:
    sample = {
        'Age': 40,
        'Experience': 14,
        'Income': 120,
        'Family': 2,
        'CCAvg': 3.5,
        'Education': 2,
        'Mortgage': 0,
        'Securities Account': 0,
        'CD Account': 1,
        'Online': 1,
        'CreditCard': 1,
    }
    return pd.DataFrame([sample])


def build_input_from_args(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'Age': args.age,
            'Experience': args.experience,
            'Income': args.income,
            'Family': args.family,
            'CCAvg': args.ccavg,
            'Education': args.education,
            'Mortgage': args.mortgage,
            'Securities Account': args.securities_account,
            'CD Account': args.cd_account,
            'Online': args.online,
            'CreditCard': args.creditcard,
        }
    ])


def build_input_from_csv(csv_path: Path, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in CSV: {missing}')
    return df[feature_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict personal loan acceptance using the saved SMOTE + Random Forest model.')
    parser.add_argument('--csv', type=str, help='Path to a CSV file containing input rows.')
    parser.add_argument('--age', type=float)
    parser.add_argument('--experience', type=float)
    parser.add_argument('--income', type=float)
    parser.add_argument('--family', type=float)
    parser.add_argument('--ccavg', type=float)
    parser.add_argument('--education', type=float)
    parser.add_argument('--mortgage', type=float)
    parser.add_argument('--securities-account', dest='securities_account', type=float)
    parser.add_argument('--cd-account', dest='cd_account', type=float)
    parser.add_argument('--online', type=float)
    parser.add_argument('--creditcard', type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(MODEL_PATH)
    feature_columns = load_feature_columns()

    if args.csv:
        input_df = build_input_from_csv(Path(args.csv), feature_columns)
    else:
        values = [
            args.age,
            args.experience,
            args.income,
            args.family,
            args.ccavg,
            args.education,
            args.mortgage,
            args.securities_account,
            args.cd_account,
            args.online,
            args.creditcard,
        ]
        if all(value is None for value in values):
            input_df = build_sample_input()
        elif any(value is None for value in values):
            raise ValueError('Provide all feature arguments for single-row prediction, or use --csv.')
        else:
            input_df = build_input_from_args(args)

    input_df = input_df[feature_columns].copy()
    input_df['Experience'] = input_df['Experience'].clip(lower=0)

    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    result_df = input_df.copy()
    result_df['Predicted Class'] = predictions
    result_df['Probability Personal Loan = 1'] = probabilities
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()
