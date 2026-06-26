import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing(df):
    df = df.copy()

    no_feature_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in no_feature_cols:
        df[col] = df[col].fillna('None')

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    return df


def encode_categoricals(df):
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(df, target='SalePrice'):
    df = df.copy()
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler