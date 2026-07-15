# Capstone Project — Credit Risk Analysis
Credit Risk Analysis (Analisis Risiko Kredit) adalah proses yang dilakukan untuk menilai dan mengukur kemungkinan seorang nasabah atau peminjam gagal memenuhi kewajiban finansialnya (gagal bayar). Dalam proyek ini, analisis dilakukan menggunakan pendekatan Data Science dan Machine Learning untuk mengklasifikasikan nasabah ke dalam kategori Good Risk (lancar) atau Bad Risk (macet) berdasarkan karakteristik dan riwayat finansial mereka. Analisis, preprocessing, dan pemodelan risiko kredit menggunakan **German Credit Data**. Project ini diorganisir mengikuti struktur repository capstone
data mining.

## 🔑 5 Insight Utama dari EDA
1. **Data imbalance.** Kelas target tidak seimbang: 69.7% nasabah *good*
   (label 1) vs 30.3% *bad* (label 2) — model dievaluasi dengan
   Recall/F1/ROC-AUC, bukan hanya Accuracy.
2. **Jumlah kredit & durasi berkorelasi kuat (r=0.63)** dan keduanya lebih
   tinggi pada nasabah *bad*.
3. **Status rekening giro (Checking account) adalah sinyal risiko terkuat** —
   bad-rate menurun tajam seiring saldo membaik.
4. **Tujuan kredit (Purpose) memengaruhi risiko** — `vacation/others` dan
   `repairs` memiliki bad-rate tertinggi.
5. **Kepemilikan rumah (Housing) berpola serupa** — nasabah dengan rumah
   sendiri punya bad-rate paling rendah.

Detail lengkap ada di `notebooks/01_eda.ipynb`.
## 📁 Struktur Repository

```
capstone-project-data-mining/
├── data/
│   ├── raw/                  # Data mentah (letakkan german_credit_data_updated.csv di sini)
│   ├── processed/            # Data yang sudah diproses (dibuat otomatis oleh 02_modeling.ipynb)
│   └── external/             # Data referensi eksternal
├── notebooks/
│   ├── 01_eda.ipynb          # EDA dan data quality check
│   ├── 02_modeling.ipynb     # Feature engineering, preprocessing, training, tuning, evaluasi
│   └── 03_interpretation.ipynb  # Feature importance, SHAP, evaluasi akhir di test set
├── src/
│   ├── data_preprocessing.py # load_data(), split_data(), build_preprocessor()
│   ├── train_model.py        # Script training end-to-end (CLI)
│   ├── evaluate_model.py     # Script evaluasi model tersimpan (CLI)
│   └── utils.py               # RANDOM_STATE, feature_engineering(), evaluate_model()
├── models/
│   ├── best_model.pkl        # Model terbaik (dibuat oleh 02_modeling.ipynb / train_model.py)
│   └── preprocessing.pkl     # Preprocessing pipeline dari model terbaik
├── app/
│   ├── app.py                 # Aplikasi Streamlit utama (prediksi risiko kredit)
│   ├── pages/                 # Halaman tambahan Streamlit (opsional)
│   └── assets/                 # Gambar, CSS, dll.
├── reports/
│   ├── final_report.pdf       # Laporan akhir (lihat catatan di bawah)
│   └── presentation.pptx      # Slide presentasi (lihat catatan di bawah)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Cara Menjalankan

1. **Setup environment**
   ```bash
   pip install -r requirements.txt (library)
   ```

2. **Siapkan data** — letakkan `german_credit_data_updated.csv` di `data/raw/`.
3. **Jalankan notebooks secara berurutan** (dari folder `notebooks/`):
   - `01_eda.ipynb` — eksplorasi & data quality check.
   - `02_modeling.ipynb` — feature engineering, training, tuning, evaluasi
     validation set. Menyimpan model ke `models/` dan data terproses ke
     `data/processed/`.
   - `03_interpretation.ipynb` — memuat kembali model tersimpan (tanpa
     training ulang), interpretasi (feature importance / SHAP), dan evaluasi
     akhir di test set.

   Alternatif via CLI (setara dengan 02 & 03 tapi tanpa visualisasi inline):
   ```bash
   cd src
   python train_model.py
   python evaluate_model.py
   ```

4. **Jalankan aplikasi Streamlit** (setelah model tersimpan di `models/best_model.pkl`):
   ```bash
   streamlit run app/app.py
   ```

## 🧠 Ringkasan Metodologi

- **Split:** 70% train / 15% validation / 15% test, stratified, dilakukan
  sebelum fitting apa pun untuk mencegah data leakage.
- **Feature engineering:** `Credit_per_Month`, `Age_Group`, `Job_Skill`
  (fungsi deterministik, aman diterapkan ke train/val/test).
- **Preprocessing:** imputasi median + `StandardScaler` untuk numerik;
  imputasi modus + `OneHotEncoder` untuk kategorikal biasa; imputasi
  konstanta `"none"` untuk kategorikal dengan missing bermakna
  (`Saving accounts`, `Checking account`).
- **Model:** Logistic Regression, Random Forest, XGBoost — masing-masing
  di-tuning dengan `GridSearchCV`/`RandomizedSearchCV` (5-fold Stratified CV,
  scoring `roc_auc`).
- **Pemilihan model terbaik:** kombinasi ROC-AUC & Recall kelas *Bad* pada
  validation set (bukan Accuracy semata, karena data imbalanced dan *false
  negative* — menyetujui kredit nasabah berisiko — adalah kesalahan paling
  mahal).
- **Interpretasi:** feature importance / koefisien + SHAP pada model
  terbaik.
