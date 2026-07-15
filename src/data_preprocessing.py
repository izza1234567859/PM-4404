"""
data_preprocessing.py — Script preprocessing.

Berisi fungsi untuk:
- load_data(): membaca CSV mentah dari data/raw/.
- split_data(): encode target & split menjadi train/val/test (70/15/15,
  stratified), dilakukan SEBELUM fitting apa pun agar tidak ada data leakage.
- build_preprocessor(): membangun ColumnTransformer (imputasi + scaling +
  one-hot encoding) yang di-fit hanya pada data train.

Kolom yang ditangani:
- Numerik (Age, Job, Credit amount, Duration, Credit_per_Month): imputasi
  median + StandardScaler.
- Kategorikal biasa (Sex, Housing, Purpose, Age_Group, Job_Skill): imputasi
  modus + OneHotEncoder.
- Kategorikal dengan missing bermakna (Saving accounts, Checking account):
  imputasi konstanta "none" (bukan modus, karena kosong kemungkinan besar
  berarti "tidak punya rekening") + OneHotEncoder.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import RANDOM_STATE

NUMERIC_FEATURES = ['Age', 'Job', 'Credit amount', 'Duration', 'Credit_per_Month']
MISSING_MEANINGFUL_FEATURES = ['Saving accounts', 'Checking account']
CATEGORICAL_FEATURES = ['Sex', 'Housing', 'Purpose', 'Age_Group', 'Job_Skill']


def load_data(path: str = '../data/raw/german_credit_data_updated.csv') -> pd.DataFrame:
    """Muat dataset German Credit Data mentah dari CSV."""
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame):
    """Encode target (1=Good->0, 2=Bad->1) lalu split 70/15/15 (stratified).

    Return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = df.copy()
    df['target'] = df['Credit Risk'].map({1: 0, 2: 1})

    drop_cols = [c for c in ['Unnamed: 0', 'Credit Risk', 'target'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['target']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """Bangun ColumnTransformer preprocessing (belum di-fit)."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    missing_meaningful_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ('miss', missing_meaningful_transformer, MISSING_MEANINGFUL_FEATURES)
    ])
    return preprocessor
