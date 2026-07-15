"""
utils.py — Fungsi utilitas yang dipakai bersama oleh notebooks (02, 03)
dan script-script di src/ (train_model.py, evaluate_model.py).

Berisi:
- RANDOM_STATE: seed konsisten untuk reproducibility di seluruh pipeline.
- feature_engineering(): fungsi deterministik untuk membuat fitur turunan
  (Credit_per_Month, Age_Group, Job_Skill), aman diterapkan ke train/val/test
  tanpa risiko data leakage.
- evaluate_model(): menghitung metrik evaluasi standar (Accuracy, Precision,
  Recall, F1, ROC-AUC) untuk kelas "Bad" (1).
"""

import pandas as pd

RANDOM_STATE = 42


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan ke dataframe fitur mentah.

    - Credit_per_Month = Credit amount / Duration -> proxy beban cicilan bulanan.
    - Age_Group        = kategori umur (Young/Adult/Middle/Senior).
    - Job_Skill        = label kategori dari kode ordinal `Job` (0-3).

    Fungsi ini murni deterministik (tidak melakukan fit dari data), sehingga
    aman dipanggil langsung pada X_train, X_val, maupun X_test.
    """
    data = data.copy()
    data['Credit_per_Month'] = data['Credit amount'] / data['Duration']

    data['Age_Group'] = pd.cut(
        data['Age'], bins=[0, 25, 40, 60, 100],
        labels=['Young(<=25)', 'Adult(26-40)', 'Middle(41-60)', 'Senior(60+)']
    ).astype(str)

    job_map = {0: 'unskilled_nonresident', 1: 'unskilled_resident',
               2: 'skilled', 3: 'highly_skilled'}
    data['Job_Skill'] = data['Job'].map(job_map)
    return data


def evaluate_model(model, X, y, name: str) -> dict:
    """Evaluasi model klasifikasi biner pada kelas Bad (1).

    Mengembalikan dict berisi Accuracy, Precision, Recall, F1, dan ROC-AUC.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'Model': name,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision (Bad)': precision_score(y, y_pred),
        'Recall (Bad)': recall_score(y, y_pred),
        'F1 (Bad)': f1_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_prob)
    }
