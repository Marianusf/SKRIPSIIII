import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Import modul buatan sendiri
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from preprocessing import preprocess, get_cat_indices, TARGET, SELECTED_FEATURES
from C45 import C45

# ==============================
# 1. KONFIGURASI & STYLE
# ==============================
st.set_page_config(layout="wide", page_title="Sistem Prediksi Mahasiswa")

st.markdown("""
<style>
    /* Background Sidebar Merah Marun */
    [data-testid="stSidebar"] { background-color: #800000; }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: white; }
    
    /* Tombol Utama Warna Emas */
    .stButton>button {
        background-color: #f0a500;
        color: black;
        border-radius: 5px;
        font-weight: bold;
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover { background-color: #d99400; color: white; }
    
    /* Input Fields Border */
    .stTextInput>div>div>input, .stNumberInput>div>div>input { border-color: #f0a500; }

    /* Judul */
    h1, h2, h3 { color: #f0a500; text-align: center; font-family: 'Arial'; }
    
    /* Container Padding */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if "data_processed" not in st.session_state:
    st.session_state["data_processed"] = None

# Temp Model: Hasil training yang belum dikomit/disimpan user
if "temp_model" not in st.session_state:
    st.session_state["temp_model"] = None
if "temp_results" not in st.session_state:
    st.session_state["temp_results"] = None

# Active Model: Model yang valid untuk digunakan di Uji Data
if "active_model" not in st.session_state:
    st.session_state["active_model"] = None

# Urutan kolom fitur untuk prediksi (tanpa Target)
PRED_COLUMNS = [c for c in SELECTED_FEATURES if c != TARGET]

# ==============================
# 2. SIDEBAR MENU
# ==============================
st.sidebar.title("MENU UTAMA")
menu = st.sidebar.radio("Pilih Menu", ["MODELING", "UJI DATA"])

# ==============================
# MENU A: MODELING (TRAINING)
# ==============================
if menu == "MODELING":
    st.title("SISTEM PREDIKSI MAHASISWA BERISIKO")
    st.subheader("MODELING & TRAINING")

    # --- A. UPLOAD DATA ---
    c1, c2, c3 = st.columns([2, 1, 1])
    uploaded_file = c1.file_uploader("UPLOAD DATA LATIH (CSV)", type=["csv"])
    
    # Tombol Download Format
    fmt_csv = pd.DataFrame(columns=SELECTED_FEATURES).to_csv(index=False, sep=';')
    c3.download_button("📥 FORMAT CSV", fmt_csv, "format_training.csv")

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python') 
        except:
            st.error("Gagal membaca file. Pastikan format CSV benar.")
            st.stop()

        with st.expander("🔍 Lihat Data Mentah"):
            st.dataframe(df_raw.head())

        st.divider()
        
        # --- B. PREPROCESSING ---
        st.subheader("1. TAHAP PREPROCESSING")
        
        if st.button("▶️ JALANKAN PREPROSES"):
            with st.spinner("Sedang membersihkan data..."):
                try:
                    df_proc, _ = preprocess(df_raw)
                    st.session_state["data_processed"] = df_proc
                    st.session_state["temp_model"] = None # Reset hasil training lama jika data baru
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi error saat preprocessing: {e}")

        if st.session_state["data_processed"] is not None:
            st.success(f"Preprocessing Selesai! Total Data: {len(st.session_state['data_processed'])} baris")
            with st.expander("📄 Lihat Data Hasil Preprocess"):
                st.dataframe(st.session_state["data_processed"].head())

            st.divider()

            # --- C. TRAINING ---
            st.subheader("2. PARAMETER & TRAINING")
            
            col1, col2 = st.columns(2)
            k_val = col1.number_input("Jumlah K-Fold", 2, 10, 5)
            leaf_val = col2.number_input("Min Samples Leaf", 1, 50, 5)

            if st.button("🚀 PROSES DATA (TRAINING)"):
                with st.spinner("Sedang melatih model..."):
                    df = st.session_state["data_processed"]
                    X, y = df.drop(TARGET, axis=1), df[TARGET]
                    cat_idx = get_cat_indices(X)

                    kf = StratifiedKFold(n_splits=k_val, shuffle=True, random_state=42)
                    f1_norm, f1_smote = [], []

                    # Loop K-Fold
                    for train_idx, test_idx in kf.split(X, y):
                        XT, Xt = X.iloc[train_idx], X.iloc[test_idx]
                        yT, yt = y.iloc[train_idx], y.iloc[test_idx]

                        # Normal
                        model_n = C45(min_samples_leaf=leaf_val)
                        model_n.fit(XT, yT)
                        f1_norm.append(f1_score(yt, model_n.predict(Xt), average='macro'))

                        # SMOTE-ENN
                        sm_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
                        sm_enn = SMOTEENN(smote=sm_nc, random_state=42)
                        try:
                            XR, yR = sm_enn.fit_resample(XT, yT)
                            model_s = C45(min_samples_leaf=leaf_val)
                            model_s.fit(XR, yR)
                            f1_smote.append(f1_score(yt, model_s.predict(Xt), average='macro'))
                        except:
                            f1_smote.append(0)

                    # Pilih Pemenang
                    mean_norm = np.mean(f1_norm)
                    mean_smote = np.mean(f1_smote)
                    
                    X_f, y_f = X, y
                    metode = "Normal (Tanpa Balancing)"
                    
                    if mean_smote > mean_norm:
                        sm_nc_f = SMOTENC(categorical_features=cat_idx, random_state=42)
                        sm_enn_f = SMOTEENN(smote=sm_nc_f, random_state=42)
                        X_f, y_f = sm_enn_f.fit_resample(X, y)
                        metode = "SMOTE-ENN (Balancing)"

                    # Final Training (Disimpan di TEMP dulu)
                    final_m = C45(min_samples_leaf=leaf_val)
                    final_m.fit(X_f, y_f)

                    st.session_state["temp_model"] = final_m
                    st.session_state["temp_results"] = {
                        "norm": mean_norm, "smote": mean_smote, "winner": metode
                    }
                    st.rerun()

            # TAMPILKAN HASIL TRAINING (Jika ada di Temp)
            if st.session_state["temp_model"] is not None:
                res = st.session_state["temp_results"]
                st.info(f"🏆 Metode Terpilih: **{res['winner']}**")
                
                m1, m2 = st.columns(2)
                m1.metric("F1-Score (Normal)", f"{res['norm']:.4f}")
                m2.metric("F1-Score (SMOTE-ENN)", f"{res['smote']:.4f}", delta=f"{res['smote']-res['norm']:.4f}")
                
                st.warning("⚠️ Model ini belum aktif. Klik Simpan di bawah untuk menggunakannya di menu Uji Data.")

                # --- LOGIKA SIMPAN MODEL (COMMIT) ---
                # Tombol ini memindahkan Temp Model -> Active Model
                if st.button("💾 SIMPAN MODEL KE SESI (AKTIFKAN)"):
                    st.session_state["active_model"] = st.session_state["temp_model"]
                    
                    # Opsional: Simpan fisik juga jika mau, tapi yang penting session-nya
                    with open("default_model.pkl", "wb") as f:
                        pickle.dump(st.session_state["temp_model"], f)
                        
                    st.success("✅ Model BERHASIL diaktifkan untuk sesi ini! Sekarang Anda bisa ke menu UJI DATA untuk mencoba prediksi.")

# ==============================
# MENU B: UJI DATA (PREDIKSI)
elif menu == "UJI DATA":

    st.title("UJI PREDIKSI")
    model_used = None
    if st.session_state["active_model"] is not None:
        model_used = st.session_state["active_model"]
        st.success(
            "⚡ Menggunakan Model: HASIL TRAINING SESI INI"
        )
    elif os.path.exists("default_model.pkl"):
        with open("default_model.pkl", "rb") as f:
            model_used = pickle.load(f)
        st.info(
            "📂 Menggunakan Model: DEFAULT / TERDAHULU"
        )

    else:

        st.error(
            "❌ Belum ada model! Silakan training model terlebih dahulu."
        )
        st.stop()
    st.divider()
    
    st.header("1. PREDIKSI MANUAL")
    st.caption(
        "Masukkan data mahasiswa untuk memprediksi status."
    )

    with st.form(key="form_manual_prediction"):

        c1, c2 = st.columns(2)
        with c1:

            ipk1 = st.number_input(
                "IPK Semester 1",
                min_value=0.0,
                max_value=4.0,
                value=0.0,
                step=0.01
            )

            ipk2 = st.number_input(
                "IPK Semester 2",
                min_value=0.0,
                max_value=4.0,
                value=0.0,
                step=0.01
            )

            ipk3 = st.number_input(
                "IPK Semester 3",
                min_value=0.0,
                max_value=4.0,
                value=0.0,
                step=0.01
            )

            total_sks = st.number_input(
                "Total SKS Semester 1-3",
                min_value=0,
                max_value=100,
                value=0
            )

            sks_d = st.number_input(
                "Jumlah SKS D/E/F",
                min_value=0,
                max_value=60,
                value=0
            )

        with c2:

            mk_d = st.number_input(
                "Jumlah Matakuliah D/E/F",
                min_value=0,
                max_value=50,
                value=0
            )

            jalur = st.selectbox(
                "Jalur Pendaftaran",
                [
                    "Tes",
                    "Raport",
                    "Lainnya"
                ]
            )

            jurusan = st.selectbox(
                "Jurusan Sekolah",
                [
                    "SMA",
                    "SMK",
                    "Homeschooling",
                    "Lainnya"
                ]
            )

            profil = st.selectbox(
                "Profil Sekolah",
                [
                    "Negeri",
                    "Swasta",
                    "Lainnya"
                ]
            )

            kepulauan = st.selectbox(
                "Kepulauan Asal",
                [
                    "Jawa",
                    "Sumatera",
                    "Bali",
                    "Kalimantan",
                    "Sulawesi",
                    "Papua & Maluku",
                    "Lainnya"
                ]
            )

        submit_manual = st.form_submit_button(
            "🔍 CEK STATUS",
            type="primary"
        )

        if submit_manual:

            errors = []
            warnings = []

            if sks_d > total_sks:

                errors.append(
                    "Jumlah SKS D/E/F tidak boleh melebihi total SKS."
                )

            if total_sks == 0 and (
                ipk1 > 0 or
                ipk2 > 0 or
                ipk3 > 0
            ):

                errors.append(
                    "IPK tidak mungkin ada jika total SKS masih 0."
                )

            if mk_d > total_sks:

                errors.append(
                    "Jumlah matakuliah gagal terlalu besar dibanding total SKS."
                )
            if errors:

                for err in errors:
                    st.error(err)

                st.stop()
            for warn in warnings:
                st.warning(warn)

            input_data = pd.DataFrame([{
                'ipk1': ipk1,
                'ipk2': ipk2,
                'ipk3': ipk3,
                'total sks semester 1-3': total_sks,
                'jumlah sks d/e/f': sks_d,
                'jumlah matakuliah d/e/f': mk_d,
                'jalur pendaftaran': jalur,
                'jurusan sekolah': jurusan,
                'profil sekolah': profil,
                'kepulauan asal lahir': kepulauan
            }])

            input_data, _ = preprocess(input_data)

            if TARGET in input_data.columns:

                input_data = input_data.drop(
                    columns=[TARGET]
                )

            for col in PRED_COLUMNS:

                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[PRED_COLUMNS]

            try:

                pred = model_used.predict(
                    input_data
                )[0]
                st.divider()
                if pred == 1:
                    st.error(
                        "### ⚠️ HASIL: BERPOTENSI SISIP"
                    )
                else:
                    st.success(
                        "### ✅ HASIL: AMAN (TIDAK SISIP)"
                    )
            except Exception as e:
                st.error(
                    f"Gagal memprediksi: {e}"
                )
    st.divider()

    st.header("2. PREDIKSI BATCH")
    template_batch = pd.DataFrame(
        columns=PRED_COLUMNS
    )
    csv_template = template_batch.to_csv(
        index=False,
        sep=';'
    )
    st.download_button(
        "📥 DOWNLOAD TEMPLATE",
        csv_template,
        "template_prediksi.csv"
    )


    batch_file = st.file_uploader(
        "Upload File CSV",
        type=["csv"]
    )
    if batch_file is not None:
        try:
            batch_file.seek(0)
            df_batch = pd.read_csv(
                batch_file,
                sep=None,
                engine='python'
            )
            st.write("Preview Data:")
            st.dataframe(df_batch.head())
            if st.button(
                "🚀 JALANKAN BATCH PREDICTION"
            ):
                with st.spinner("Memproses..."):

                    X_batch, _ = preprocess(df_batch)

                    if TARGET in X_batch.columns:

                        X_batch = X_batch.drop(
                            columns=[TARGET]
                        )

                    for col in PRED_COLUMNS:

                        if col not in X_batch.columns:
                            X_batch[col] = 0

                    X_batch = X_batch[PRED_COLUMNS]
                    y_pred = model_used.predict(
                        X_batch
                    )
                    df_result = df_batch.copy()
                    df_result["STATUS_PREDIKSI"] = [
                        "SISIP" if x == 1
                        else "AMAN"
                        for x in y_pred
                    ]
                    st.success(
                        "Prediksi batch selesai!"
                    )
                    st.dataframe(df_result)

                    res_csv = df_result.to_csv(
                        index=False,
                        sep=';'
                    )
                    st.download_button(
                        "📥 DOWNLOAD HASIL",
                        res_csv,
                        "hasil_prediksi_batch.csv"
                    )
        except Exception as e:
            st.error(f"Error: {e}")