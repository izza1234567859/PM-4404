"""
app.py — Aplikasi Streamlit utama untuk prediksi Credit Risk.

Memuat model terbaik (models/best_model.pkl) hasil training di
notebooks/02_modeling.ipynb (atau src/train_model.py), menerima input data
nasabah dari form, dan menampilkan prediksi risiko kredit (Good/Bad) beserta
probabilitasnya.

Jalankan dari root folder project:
    streamlit run app/app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import pandas as pd
import streamlit as st

from utils import feature_engineering

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

# Batas validasi input manual
AGE_MIN, AGE_MAX = 19, 75
CREDIT_AMOUNT_MAX = 20000

st.set_page_config(page_title='Credit Risk Predictor', page_icon='💳', layout='centered')


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def explain_prediction(age, housing, saving, checking, credit, duration, purpose):

    explanations = []

    # Saving Account
    if saving in ["moderate", "quite rich", "rich"]:
        explanations.append("✅ Saving Account berada pada kategori menengah hingga tinggi.")
    else:
        explanations.append("⚠️ Saving Account masih rendah.")

    # Checking Account
    if checking in ["moderate", "rich"]:
        explanations.append("✅ Checking Account menunjukkan kondisi keuangan yang baik.")
    else:
        explanations.append("⚠️ Checking Account masih rendah.")

    # Housing
    if housing == "own":
        explanations.append("✅ Memiliki rumah sendiri.")
    elif housing == "rent":
        explanations.append("ℹ️ Tinggal di rumah sewa.")
    else:
        explanations.append("⚠️ Tempat tinggal bebas (free).")

    # Age
    if 25 <= age <= 60:
        explanations.append("✅ Berada pada usia produktif.")
    else:
        explanations.append("⚠️ Usia berada di luar rentang produktif.")

    # Credit Amount vs Duration
    monthly = credit / duration

    if monthly <= 300:
        explanations.append("✅ Jumlah kredit masih proporsional terhadap lama pinjaman.")
    elif monthly <= 500:
        explanations.append("⚠️ Jumlah kredit cukup besar dibanding durasi.")
    else:
        explanations.append("❌ Jumlah kredit terlalu besar dibanding durasi pinjaman.")

    explanations.append(f"📌 Tujuan Kredit : {purpose}")

    return explanations


def main():
    st.title('💳 Credit Risk Predictor')
    st.write(
        'Aplikasi ini memprediksi apakah seorang nasabah berisiko **Good** '
        '(layak kredit) atau **Bad** (tidak layak kredit) berdasarkan data '
        'German Credit Data, menggunakan model terbaik hasil training di '
        '`notebooks/02_modeling.ipynb`.'
    )

    model = load_model()
    if model is None:
        st.error(
            'Model belum ditemukan di `models/best_model.pkl`. '
            'Jalankan `notebooks/02_modeling.ipynb` atau `python src/train_model.py` '
            'terlebih dahulu untuk melatih dan menyimpan model.'
        )
        return

    st.sidebar.header('Data Nasabah')
    age = st.sidebar.number_input(
        'Age', min_value=0, max_value=120, value=35, step=1
    )
    age_invalid = age < AGE_MIN or age > AGE_MAX
    if age_invalid:
        st.sidebar.warning(
            f'⚠️ Usia melebihi rentang yang didukung model ({AGE_MIN}–{AGE_MAX} tahun).'
        )

    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    job = st.sidebar.selectbox(
        'Job (skill level)', options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: '0 - unskilled non-resident', 1: '1 - unskilled resident',
            2: '2 - skilled', 3: '3 - highly skilled'
        }[x]
    )
    housing = st.sidebar.selectbox('Housing', ['own', 'rent', 'free'])
    saving_accounts = st.sidebar.selectbox(
        'Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'none']
    )
    checking_account = st.sidebar.selectbox(
        'Checking account', ['little', 'moderate', 'rich', 'none']
    )
    credit_amount = st.sidebar.number_input(
        'Credit amount', min_value=100, value=2000, step=100
    )
    credit_invalid = credit_amount > CREDIT_AMOUNT_MAX
    if credit_invalid:
        st.sidebar.warning(
            f'⚠️ Jumlah kredit melebihi batas maksimum yang didukung model '
            f'({CREDIT_AMOUNT_MAX:,}).'
        )

    duration = st.sidebar.slider('Duration (months)', 4, 72, 24)
    purpose = st.sidebar.selectbox(
        'Purpose',
        ['radio/TV', 'education', 'furniture/equipment', 'car', 'business',
         'domestic appliances', 'repairs', 'vacation/others']
    )

    input_df = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': None if saving_accounts == 'none' else saving_accounts,
        'Checking account': None if checking_account == 'none' else checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose,
    }])

    st.subheader('Ringkasan Input')
    st.dataframe(input_df, use_container_width=True)
    if age_invalid or credit_invalid:
        pesan = []
        if age_invalid:
            pesan.append(f'usia melebihi rentang yang didukung ({AGE_MIN}–{AGE_MAX} tahun)')
        if credit_invalid:
            pesan.append(f'jumlah kredit melebihi batas maksimum ({CREDIT_AMOUNT_MAX:,})')
        st.error('❌ Data input melebihi batas yang didukung: ' + '; '.join(pesan) + '.')

    predict_clicked = st.button('Prediksi Risiko Kredit', type='primary')

    if predict_clicked:
        if age_invalid or credit_invalid:
            st.stop()

        input_fe = feature_engineering(input_df)
        pred = model.predict(input_fe)[0]
        prob_bad = model.predict_proba(input_fe)[0, 1]

        label = 'Bad (tidak layak)' if pred == 1 else 'Good (layak)'
        if pred == 1:
            st.error(f'Hasil Prediksi: **{label}**')
        else:
            st.success(f'Hasil Prediksi: **{label}**')
        st.metric('Probabilitas Risiko Bad', f'{prob_bad*100:.1f}%')


if __name__ == '__main__':
    main()