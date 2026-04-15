import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN

from preprocessing import encode_data, transform_input, preprocess_ui
from C45 import C45

# ==============================
# STYLE
# ==============================
st.set_page_config(layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #800000;
}
.block-container {
    padding-top: 2rem;
}
.big-button button {
    background-color: #f0a500;
    color: black;
    height: 55px;
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TEMPLATE CSV
# ==============================
model_columns = [
    "kepulauan asal lahir",
    "jurusan sekolah",
    "profil sekolah",
    "jalur pendaftaran",
    "ipk1",
    "ipk2",
    "ipk3",
    "jumlah matakuliah d/e/f",
    "jumlah sks d/e/f",
    "total sks semester 1-3",
    "status"
]

pred_columns = model_columns[:-1]

model_csv = pd.DataFrame(columns=model_columns).to_csv(index=False)
pred_csv = pd.DataFrame(columns=pred_columns).to_csv(index=False)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("MENU")
menu = st.sidebar.radio("Pilih Menu", ["MODELING", "UJI DATA"])

# ==============================
# MODELING
# ==============================
if menu == "MODELING":

    st.title("SISTEM PREDIKSI MAHASISWA BERISIKO")
    st.subheader("MODELING")

    col1, col2, col3 = st.columns(3)

    with col1:
        uploaded_file = st.file_uploader("UPLOAD DATA", type=["csv"])

    with col2:
        tampilkan = st.button("TAMPILKAN DATA")

    with col3:
        st.download_button("DOWNLOAD FORMAT", model_csv, "format_model.csv")

    st.divider()

    # ======================
    # PREVIEW DATA
    # ======================
    if uploaded_file is not None:

        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, sep=';', engine='python')

        if tampilkan:
            st.dataframe(df_raw)

    # ======================
    # PREPROCESS
    # ======================
    st.subheader("PREPROCESSING")

    if st.button("PREPROSES DATA") and uploaded_file is not None:

        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', engine='python')

        df = preprocess_ui(df)

        st.write("Data setelah seleksi fitur:")
        st.dataframe(df.head())

        df_encoded, _ = encode_data(df, "status")

        st.write("Preview encoding:")
        st.dataframe(df_encoded.head())

        st.session_state["data"] = df

        st.success("Preprocessing selesai ✔")

    st.divider()

    # ======================
    # PARAMETER
    # ======================
    col1, col2 = st.columns(2)

    with col1:
        kfold = st.number_input("K-Fold", 2, 10, 5)

    with col2:
        min_leaf = st.number_input("Min Samples Leaf", 1, 50, 5)

    col1, col2 = st.columns(2)

    proses = col1.button("PROSES DATA")
    simpan = col2.button("SIMPAN MODEL")

    # ======================
    # TRAINING
    # ======================
    if proses:

        if "data" not in st.session_state:
            st.error("Lakukan preprocessing dulu!")
        else:
            df = st.session_state["data"]
            target = "status"

            X = df.drop(target, axis=1)
            y = df[target]

            kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

            acc_no_smote = []
            acc_smote = []

            for train_idx, test_idx in kf.split(X, y):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                train_df = X_train.copy()
                train_df[target] = y_train

                train_df, encoders = encode_data(train_df, target)

                X_train = train_df.drop(target, axis=1)
                y_train = train_df[target]

                test_df = X_test.copy()
                test_df[target] = y_test
                test_df = transform_input(test_df, encoders)

                X_test = test_df.drop(target, axis=1)
                y_test = test_df[target]

                # TANPA SMOTE
                model1 = C45(min_samples_leaf=min_leaf)
                model1.fit(X_train, y_train)

                pred1 = model1.predict(X_test)
                acc_no_smote.append(accuracy_score(y_test, pred1))

                # DENGAN SMOTE
                sm = SMOTEENN(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)

                model2 = C45(min_samples_leaf=min_leaf)
                model2.fit(X_res, y_res)

                pred2 = model2.predict(X_test)
                acc_smote.append(accuracy_score(y_test, pred2))

            acc1 = np.mean(acc_no_smote)
            acc2 = np.mean(acc_smote)

            st.write(f"Tanpa SMOTE: {acc1:.4f}")
            st.write(f"SMOTE: {acc2:.4f}")

            use_smote = acc2 > acc1

            # ======================
            # FINAL MODEL
            # ======================
            full_df = df.copy()
            full_df, encoders = encode_data(full_df, target)

            X_full = full_df.drop(target, axis=1)
            y_full = full_df[target]

            if use_smote:
                sm = SMOTEENN(random_state=42)
                X_full, y_full = sm.fit_resample(X_full, y_full)

            final_model = C45(min_samples_leaf=min_leaf)
            final_model.fit(X_full, y_full)

            st.session_state["model"] = final_model
            st.session_state["encoders"] = encoders

            st.success("Model siap digunakan ✔")

    # ======================
    # SIMPAN SESSION
    # ======================
    if simpan:

        if "model" in st.session_state:
            st.session_state["saved_model"] = st.session_state["model"]
            st.session_state["saved_encoders"] = st.session_state["encoders"]

            st.success("Model disimpan di session ✔")
        else:
            st.error("Belum ada model!")

# ==============================
# UJI DATA
# ==============================
elif menu == "UJI DATA":

    st.title("SISTEM PREDIKSI MAHASISWA BERISIKO")
    st.subheader("UJI DATA")

    # =========================
    # INPUT MANUAL
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        ipk1 = st.text_input("IPK Semester 1")
        ipk2 = st.text_input("IPK Semester 2")
        ipk3 = st.text_input("IPK Semester 3")

        total_sks = st.number_input("Total SKS", min_value=0)
        sks_tidak_lulus = st.number_input("SKS Tidak Lulus", min_value=0)

    with col2:
        mk_tidak_lulus = st.number_input("MK Tidak Lulus", min_value=0)

        jalur = st.selectbox("Jalur Masuk", ["RAPORT", "TES"])
        jurusan = st.selectbox("Jurusan Sekolah", ["SMA", "SMK", "Tidak Keduanya"])
        profil = st.selectbox("Profil Sekolah", ["NEGERI", "SWASTA"])
        kepulauan = st.selectbox("Kepulauan", ["Jawa", "Sumatra", "Kalimantan", "Sulawesi", "Papua"])

    st.write("")

    # =========================
    # TOMBOL
    # =========================
    col1, col2, col3 = st.columns(3)

    prediksi = col1.button("CEK PREDIKSI")
    upload_file = col2.file_uploader("UPLOAD CSV", type=["csv"])
    download = col3.download_button(
        "DOWNLOAD FORMAT",
        data=pred_csv,
        file_name="format_prediksi.csv",
        mime="text/csv"
    )

    # =========================
    # LOAD MODEL
    # =========================
    def load_model():
        if "saved_model" in st.session_state:
            st.info("Model dari user")
            return st.session_state["saved_model"], st.session_state["saved_encoders"]

        try:
            import pickle
            with open("default_model.pkl", "rb") as f:
                st.info("Model default")
                return pickle.load(f)
        except:
            st.error("Tidak ada model tersedia!")
            st.stop()

    # =========================
    # PREDIKSI MANUAL
    # =========================
    if prediksi:

        model, encoders = load_model()

        input_df = pd.DataFrame([{
            "kepulauan asal lahir": kepulauan,
            "jurusan sekolah": jurusan,
            "profil sekolah": profil,
            "jalur pendaftaran": jalur,
            "ipk1": float(ipk1) if ipk1 else 0,
            "ipk2": float(ipk2) if ipk2 else 0,
            "ipk3": float(ipk3) if ipk3 else 0,
            "jumlah matakuliah d/e/f": mk_tidak_lulus,
            "jumlah sks d/e/f": sks_tidak_lulus,
            "total sks semester 1-3": total_sks
        }])

        input_df = transform_input(input_df, encoders)

        pred = model.predict(input_df)[0]
        hasil = "BERISIKO" if pred == 1 else "TIDAK BERISIKO"

        st.success(f"HASIL: {hasil}")

    # =========================
    # PREDIKSI CSV
    # =========================
    if upload_file is not None:

        model, encoders = load_model()

        try:
            upload_file.seek(0)
            df = pd.read_csv(upload_file, sep=';', engine='python')

            df.columns = df.columns.str.strip().str.lower()

            # validasi kolom
            missing = [col for col in pred_columns if col not in df.columns]

            if missing:
                st.error(f"Kolom kurang: {missing}")
                st.stop()

            df = df[pred_columns]

            df_encoded = transform_input(df, encoders)

            preds = model.predict(df_encoded)

            df["hasil"] = ["BERISIKO" if p == 1 else "TIDAK BERISIKO" for p in preds]

            st.success("Prediksi batch selesai ✔")
            st.dataframe(df)

            csv_result = df.to_csv(index=False)

            st.download_button(
                "DOWNLOAD HASIL",
                data=csv_result,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")