import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_info(df):
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTarget variable stats:\n{df['SalePrice'].describe()}")

def check_numeric(df):
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    print(f"\nNumeric columns: {len(numeric)}")
    print(f"Categorical columns: {len(categorical)}")
    return numeric, categorical