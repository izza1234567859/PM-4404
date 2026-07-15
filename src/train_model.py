"""
train_model.py — Script training.

Menjalankan seluruh alur: load data -> split -> feature engineering ->
preprocessing -> training 3 model (Logistic Regression, Random Forest,
XGBoost) -> hyperparameter tuning (GridSearchCV / RandomizedSearchCV) ->
evaluasi di validation set -> simpan model & artefak ke models/.

Jalankan dari dalam folder src/:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import RANDOM_STATE, feature_engineering, evaluate_model
from data_preprocessing import load_data, split_data, build_preprocessor

MODELS_DIR = '../models'
PROCESSED_DIR = '../data/processed'


def build_models(preprocessor, bad_ratio: float) -> dict:
    """Bangun 3 pipeline model (preprocessor + classifier)."""
    return {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced',
                                               random_state=RANDOM_STATE))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                                          scale_pos_weight=bad_ratio))
        ])
    }


PARAM_GRIDS = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'Random Forest': {
        'classifier__n_estimators': [200, 300, 500],
        'classifier__max_depth': [4, 6, 8, 12, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'classifier__n_estimators': [200, 300, 500],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0]
    }
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print('1) Load data...')
    df = load_data()

    print('2) Split data (70/15/15, stratified)...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print('3) Feature engineering...')
    X_train_fe = feature_engineering(X_train)
    X_val_fe = feature_engineering(X_val)
    X_test_fe = feature_engineering(X_test)

    print('4) Build preprocessor & model pipelines...')
    preprocessor = build_preprocessor()
    bad_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(preprocessor, bad_ratio)

    print('5) Hyperparameter tuning (5-fold Stratified CV, scoring=roc_auc)...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_estimators = {}
    for name, pipe in models.items():
        print(f'\n=== Tuning {name} ===')
        if name == 'Logistic Regression':
            search = GridSearchCV(pipe, PARAM_GRIDS[name], scoring='roc_auc', cv=cv, n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipe, PARAM_GRIDS[name], n_iter=25, scoring='roc_auc',
                                         cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train_fe, y_train)
        best_estimators[name] = search.best_estimator_
        print(f'Best CV ROC-AUC: {search.best_score_:.4f}')
        print(f'Best params: {search.best_params_}')

    print('\n6) Evaluate on validation set...')
    val_results = [evaluate_model(m, X_val_fe, y_val, name) for name, m in best_estimators.items()]
    comparison_df = pd.DataFrame(val_results).set_index('Model').round(4)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    print(comparison_df)

    best_model_name = comparison_df['ROC-AUC'].idxmax()
    best_model = best_estimators[best_model_name]
    print(f'\n>>> Model terpilih: {best_model_name} <<<')

    print('\n7) Simpan artefak ke models/ dan data/processed/...')
    joblib.dump(best_estimators, os.path.join(MODELS_DIR, 'all_models.pkl'))
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(best_model.named_steps['preprocessor'], os.path.join(MODELS_DIR, 'preprocessing.pkl'))
    comparison_df.to_csv(os.path.join(MODELS_DIR, 'comparison_val_results.csv'))

    X_val_fe.to_csv(os.path.join(PROCESSED_DIR, 'X_val_fe.csv'), index=False)
    X_test_fe.to_csv(os.path.join(PROCESSED_DIR, 'X_test_fe.csv'), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

    print('Selesai. Model terbaik disimpan di models/best_model.pkl')


if __name__ == '__main__':
    main()
