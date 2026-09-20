import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def split_data(df, target='SalePrice'):
    X = df.drop(columns=[target])
    y = np.log1p(df[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    param_grid = {'alpha': [0.1, 1, 5, 10, 20, 50, 100]}
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    print(f"Best alpha found: {grid.best_params_['alpha']}")
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = np.expm1(predictions)
    y_test = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"\nRMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    return predictions