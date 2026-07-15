"""
evaluate_model.py — Script evaluasi.

Memuat model terbaik (models/best_model.pkl) beserta test set yang sudah
diproses (data/processed/), lalu menghitung metrik evaluasi akhir
(Accuracy, Precision, Recall, F1, ROC-AUC) dan menyimpan confusion matrix
ke reports/.

Jalankan dari dalam folder src/ (setelah train_model.py):
    python evaluate_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils import evaluate_model

MODELS_DIR = '../models'
PROCESSED_DIR = '../data/processed'
REPORTS_DIR = '../reports'


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print('Load model & test data...')
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    comparison_df = pd.read_csv(os.path.join(MODELS_DIR, 'comparison_val_results.csv'), index_col=0)
    best_model_name = comparison_df['ROC-AUC'].idxmax()

    X_test_fe = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test_fe.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()

    print(f'Model dievaluasi: {best_model_name}')
    result = evaluate_model(best_model, X_test_fe, y_test, best_model_name)
    print('\n=== Evaluasi Akhir di Test Set ===')
    for k, v in result.items():
        if k != 'Model':
            print(f'{k}: {v:.4f}')

    y_pred = best_model.predict(X_test_fe)
    print(f'\nClassification Report - {best_model_name} (Test Set):')
    print(classification_report(y_test, y_pred, target_names=['Good(0)', 'Bad(1)']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Good(0)', 'Bad(1)'], yticklabels=['Good(0)', 'Bad(1)'])
    plt.title(f'Confusion Matrix - {best_model_name} (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, 'confusion_matrix_test.png')
    plt.savefig(out_path)
    print(f'\nConfusion matrix disimpan ke {out_path}')


if __name__ == '__main__':
    main()
