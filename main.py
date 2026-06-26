import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.explore import load_data, basic_info, check_numeric
from src.preprocess import handle_missing, encode_categoricals, scale_features
from src.model import split_data, train_model, evaluate_model


def main():
    df = load_data("data/train.csv")
    basic_info(df)
    check_numeric(df)

    df = handle_missing(df)
    df = encode_categoricals(df)
    df, scaler = scale_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()