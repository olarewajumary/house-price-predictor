import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.explore import load_data, basic_info, check_numeric
from src.preprocess import handle_missing, encode_categoricals, scale_features
from src.model import split_data, train_model, evaluate_model


def main():
    df = load_data("data/train.csv")
    basic_info(df)
    check_numeric(df)

    df = handle_missing(df)
    defaults = df.drop(columns=['SalePrice']).median(numeric_only=True).to_dict()
    for col in df.select_dtypes(include=['object', 'str']).columns:
        defaults[col] = df[col].mode()[0]
    df, encoders = encode_categoricals(df)
    df, scaler = scale_features(df)
    

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(defaults, "artifacts/defaults.pkl")
    joblib.dump(list(X_train.columns), "artifacts/columns.pkl")


if __name__ == "__main__":
    main()