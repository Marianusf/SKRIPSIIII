import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Import modul buatan sendiri
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] { display: none; }
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

TemplateTraining = [
    "nim","kepulauan asal lahir",
    "jurusan sekolah","profil sekolah","jalur pendaftaran",
    "ipk1", "ipk2", "ipk3", "ipk4",  
    "jumlah matakuliah d/e/f","jumlah sks d/e/f","ipk 60 sks",
    "total sks semester 1-3",
    "total sks semester 1-4",
    TARGET
]
TemplatePrediksi = [
    "nim","kepulauan asal lahir",
    "jurusan sekolah","profil sekolah","jalur pendaftaran",
    "ipk1", "ipk2", "ipk3", "ipk4",  
    "jumlah matakuliah d/e/f","jumlah sks d/e/f","ipk 60 sks",
    "total sks semester 1-3",
    "total sks semester 1-4",
]

VALIDASITRAINING = [c.lower() for c in SELECTED_FEATURES + [TARGET]]
VALIDASIPREDIKSI = [c.lower() for c in SELECTED_FEATURES if c != TARGET]
# ==============================
# 2. SIDEBAR MENU
# ==============================
st.sidebar.title("MENU UTAMA")
menu = st.sidebar.radio("Pilih Menu", ["MODELING", "UJI DATA"])

if menu == "MODELING":
    st.title("SISTEM PREDIKSI MAHASISWA BERISIKO SISIP PROGRAM STUDI")
    st.subheader("MODELING & TRAINING")

    # --- 1. UPLOAD DATA ---
    c1, c2, c3 = st.columns([2, 1, 1])
    uploaded_file = c1.file_uploader("UPLOAD DATA LATIH (CSV)", type=["csv"])
    csv_tmpl = pd.DataFrame(columns=TemplateTraining).to_csv(index=False, sep=';')
    c3.download_button("📥 FORMAT LATIH", csv_tmpl, "template_training.csv", help="Format lengkap: Identitas + Fitur + Target")

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python') 
            uploaded_cols = [c.lower().strip() for c in df_raw.columns]
            missing_cols = [c for c in VALIDASITRAINING if c not in uploaded_cols]

            if missing_cols:
                st.error("⛔ **FILE DITOLAK: Struktur Data Tidak Sesuai! Silahkan Download dan gunakan template **")
                st.warning(f"File training wajib memiliki kolom berikut (Case Insensitive):\n\n`{', '.join(missing_cols)}`")
                st.stop()
            with st.expander("🔍 Lihat Data Mentah"):
                st.dataframe(df_raw.head())

            st.divider()
        
         # --- 2. PREPROCESSING ---
            st.subheader("1. TAHAP PREPROCESSING")
            
            if st.button("▶️ JALANKAN PREPROSES"):
                with st.spinner("Sedang membersihkan data..."):
                    try:
                        # Fungsi preprocess Anda (Pastikan menghandle penghapusan kolom Identity)
                        df_proc, _ = preprocess(df_raw)
                        st.session_state["data_processed"] = df_proc
                        st.session_state["temp_model"] = None 
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error Preprocessing: {e}")
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()

        if "data_processed" in st.session_state and st.session_state["data_processed"] is not None:
            df = st.session_state["data_processed"]
            
            # --- FITUR DIAGNOSA DATA (X-RAY) ---
            st.info("📊 **Laporan Distribusi Data:**")
            
            # Hitung jumlah Aman vs Sisip
            counts = df[TARGET].value_counts()
            c_aman = counts.get(0, 0) # Asumsi 0 = Aman
            c_sisip = counts.get(1, 0) # Asumsi 1 = Sisip (Target)
            
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Total Data", len(df))
            col_d2.metric("Jumlah 'Aman' (0)", c_aman)
            col_d3.metric("Jumlah 'Sisip' (1)", c_sisip) # <--- PERHATIKAN ANGKA INI

            # VALIDASI KRITIS
            if c_sisip < 2:
                st.error(f"⛔ **DATA TIDAK BISA DIPROSES!**\nJumlah data 'Sisip' hanya {c_sisip}. Minimal harus ada 2 data agar bisa dibagi (1 Latih, 1 Uji).")
                st.stop()
            elif c_sisip < 10:
                st.warning("⚠️ **PERINGATAN:** Data 'Sisip' sangat sedikit (< 10). Hasil SMOTE mungkin tidak akurat.")
            st.divider()
            # --- 3. KONFIGURASI ---
            st.subheader("2. KONFIGURASI TRAINING")
            c_param1, c_param2 = st.columns(2)
            with c_param1:
                k_val = st.number_input("Jumlah K-Fold", 2, 10, 5)
            with c_param2:
                leaf_val = st.number_input("Min Samples Leaf", 1, 50, 5)
            test_size = 0.2 

            # --- 4. EKSEKUSI TRAINING ---
            if st.button("🚀 MULAI TRAINING", type="primary"):
                with st.spinner("Sedang training..."):
                    X = df.drop(TARGET, axis=1)
                    y = df[TARGET]
                    cat_idx = get_cat_indices(X)

                    # TAHAP A: SPLIT 80:20
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )

                    # TAHAP B: K-FOLD (Hanya di Training)
                    kf = StratifiedKFold(n_splits=k_val, shuffle=True, random_state=42)
                    scores_norm, scores_smote = [], []

                    try:
                        for train_ix, val_ix in kf.split(X_train, y_train):
                            XT, Xv = X_train.iloc[train_ix], X_train.iloc[val_ix]
                            yT, yv = y_train.iloc[train_ix], y_train.iloc[val_ix]

                            # Normal
                            m1 = C45(min_samples_leaf=leaf_val)
                            m1.fit(XT, yT)
                            scores_norm.append(f1_score(yv, m1.predict(Xv), average='macro'))

                            # SMOTE
                            try:
                                # Pakai k=1 jika data latih < 6
                                k_neigh = 1 if yT.value_counts().min() < 6 else 5
                                sm = SMOTEENN(smote=SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neigh), random_state=42)
                                XR, yR = sm.fit_resample(XT, yT)
                                m2 = C45(min_samples_leaf=leaf_val)
                                m2.fit(XR, yR)
                                scores_smote.append(f1_score(yv, m2.predict(Xv), average='macro'))
                            except:
                                scores_smote.append(0)
                    except Exception as e:
                        st.error(f"Gagal saat K-Fold: {e}")
                        st.stop()

                    mean_norm = np.mean(scores_norm)
                    mean_smote = np.mean(scores_smote)

                    # TAHAP C: FINAL TEST
                    winner = "Normal"
                    final_score = 0
                    
                    if mean_smote > mean_norm and mean_smote > 0:
                        winner = "SMOTE-ENN"
                        # Retrain Full Train set
                        k_neigh = 1 if y_train.value_counts().min() < 6 else 5
                        sm_full = SMOTEENN(smote=SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neigh), random_state=42)
                        XT_final, yT_final = sm_full.fit_resample(X_train, y_train)
                        
                        m_final = C45(min_samples_leaf=leaf_val)
                        m_final.fit(XT_final, yT_final)
                    else:
                        m_final = C45(min_samples_leaf=leaf_val)
                        m_final.fit(X_train, y_train)

                    final_score = f1_score(y_test, m_final.predict(X_test), average='macro')

                    # TAHAP D: DEPLOYMENT (100% Data)
                    m_deploy = C45(min_samples_leaf=leaf_val)
                    if "SMOTE" in winner:
                         try:
                             k_neigh = 1 if y.value_counts().min() < 6 else 5
                             sm_deploy = SMOTEENN(smote=SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neigh), random_state=42)
                             XA, yA = sm_deploy.fit_resample(X, y)
                             m_deploy.fit(XA, yA)
                         except:
                             m_deploy.fit(X, y)
                    else:
                         m_deploy.fit(X, y)

                    st.session_state["temp_model"] = m_deploy
                    st.session_state["temp_results"] = {
                        "norm": mean_norm, "smote": mean_smote, "test": final_score, "winner": winner
                    }

            # HASIL
            if "temp_results" in st.session_state and st.session_state["temp_results"] is not None:
                res = st.session_state["temp_results"]
                st.success("✅ Training Selesai!")
                
                # Tampilkan Metrik
                c1, c2, c3 = st.columns(3)
                c1.metric("Validasi (Normal)", f"{res['norm']:.4f}")
                c2.metric("Validasi (SMOTE)", f"{res['smote']:.4f}")
                c3.metric("⭐ SKOR UJI (FINAL)", f"{res['test']:.4f}")
                
                st.info(f"🏆 Metode Terbaik: **{res['winner']}**")
                if st.button("⚡ SIMPAN MODEL (SESI INI SAJA)"):
                    model_to_use = st.session_state["temp_model"]
                    
                    # Tempelkan Metadata (Agar info muncul di menu Uji)
                    model_to_use.metadata = {
                        "f1_score": res['test'],
                        "total_data": len(df),
                        "algoritma": res['winner'] + " (Sesi Live)"
                    }
                    
                    # 1. Simpan ke Session State (RAM)
                    st.session_state["active_model"] = model_to_use
                    
                    st.success("✅ Model AKTIF! Silakan pindah ke menu 'UJI DATA'.")
                    st.warning("⚠️ Catatan: Model ini hanya hidup sementara. Jika browser di-refresh, aplikasi akan kembali menggunakan model Default (File).")


elif menu == "UJI DATA":
    st.title("🎯 UJI PREDIKSI (TESTING)")
    model_used = None

    if "active_model" in st.session_state and st.session_state["active_model"] is not None:
        model_used = st.session_state["active_model"]
        st.success("⚡ Menggunakan Model: HASIL TRAINING BARU (LIVE)")
    elif os.path.exists("default_model.pkl"):
        with open("default_model.pkl", "rb") as f:
            model_used = pickle.load(f)
        st.info("📂 Menggunakan Model: TERSIMPAN (DEFAULT) F1-SCORE : 0.9330")
    else:
        st.error("❌ Belum ada model! Silakan ke menu MODELING untuk training dulu.")
        st.stop()

    st.divider()

# Fungsi untuk reset form input
    def reset_callback(): 
        keys = ["ipk1", "ipk2", "ipk3", "total_sks", "sks_d", "mk_d", 
                "jalur", "jurusan", "profil", "kepulauan"]
        for k in keys:
            st.session_state[k] = None

    st.subheader("1. PREDIKSI MANUAL SATUAN")
    with st.form(key="form_manual"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🎓 Akademik")
            ipk1 = st.number_input("IPK Semester 1", 0.0, 4.0, value=None, step=0.01, placeholder="Cth: 3.50", key="ipk1")
            ipk2 = st.number_input("IPK Semester 2", 0.0, 4.0, value=None, step=0.01, placeholder="Cth: 3.45", key="ipk2")
            ipk3 = st.number_input("IPK Semester 3", 0.0, 4.0, value=None, step=0.01, placeholder="Cth: 3.20", key="ipk3")
            total_sks = st.number_input("Total SKS (Sem 1-3)", 0, 100, value=None, step=1, placeholder="Cth: 60", key="total_sks")
            sks_d = st.number_input("Jumlah SKS Nilai D/E/F", 0, 60, value=None, step=1, placeholder="Ketik 0 jika tidak ada", key="sks_d")
        with c2:
            st.markdown("##### 👤 Profil & Matakuliah")
            mk_d = st.number_input("Jumlah Matakuliah D/E/F", 0, 50, value=None, step=1, placeholder="Ketik 0 jika tidak ada", key="mk_d")
            
            jalur = st.selectbox("Jalur Pendaftaran", ["Tes", "Raport", "Lain-lain"], index=None, placeholder="Pilih Jalur...", key="jalur")
            jurusan = st.selectbox("Jurusan Sekolah", ["SMA", "SMK", "Home schooling", "Lain-lain"], index=None, placeholder="Pilih Jurusan...", key="jurusan")
            profil = st.selectbox("Profil Sekolah", ["Negeri", "Swasta", "Lain-lain"], index=None, placeholder="Pilih Profil...", key="profil")
            kepulauan = st.selectbox("Kepulauan Asal", ["Jawa", "Sumatera", "Bali", "Kalimantan", "Sulawesi", "Papua & Maluku", "Lain-lain"], index=None, placeholder="Pilih Asal...", key="kepulauan")
        col_reset, col_submit = st.columns([1, 1])
        with col_reset:
            st.form_submit_button("🔃 Reset Field", on_click=reset_callback, type="secondary")
        with col_submit:
            submit = st.form_submit_button("🔍 CEK STATUS MAHASISWA", type="primary")

        if submit:
            # A. Cek Kelengkapan
            input_map = {
                "IPK Sem 1": ipk1, "IPK Sem 2": ipk2, "IPK Sem 3": ipk3,
                "Total SKS": total_sks, "SKS Gagal": sks_d, "MK Gagal": mk_d,
                "Jalur": jalur, "Jurusan": jurusan, "Profil": profil, "Asal": kepulauan
            }
            field_kosong = [label for label, nilai in input_map.items() if nilai is None]
            if field_kosong:
                list_str = ", ".join(field_kosong)
                st.warning(f"⚠️ **Data Belum Lengkap!** Mohon isi kolom berikut: **{list_str}**")
            # B. Cek Logika Angka
            elif total_sks is not None and sks_d is not None:
                if sks_d > total_sks:
                    st.error("⛔ **Logika Salah:** SKS Gagal tidak boleh lebih besar dari Total SKS!")
                elif total_sks == 0 and (ipk1 + ipk2 + ipk3 > 0):
                    st.error("⛔ **Logika Aneh:** Total SKS 0 tapi memiliki IPK. Mohon cek kembali.")  
                else:
                    # C. EKSEKUSI PREDIKSI
                    input_data = pd.DataFrame([{
                        'ipk1': ipk1, 'ipk2': ipk2, 'ipk3': ipk3,
                        'total sks semester 1-3': total_sks,
                        'jumlah sks d/e/f': sks_d, 'jumlah matakuliah d/e/f': mk_d,
                        'jalur pendaftaran': jalur, 'jurusan sekolah': jurusan,
                        'profil sekolah': profil, 'kepulauan asal lahir': kepulauan
                    }])

                    try:
                        df_clean, _ = preprocess(input_data)
                        X_final = df_clean.reindex(columns=VALIDASIPREDIKSI, fill_value=0)
                        pred = model_used.predict(X_final)[0]
                        
                        st.divider()
                        if pred == 1:
                            st.error(f"### ⚠️ HASIL: RISIKO SISIP!")
                        else:
                            st.success(f"### ✅ HASIL: AMAN (TIDAK SISIP)")                            
                    except Exception as e:
                        st.error(f"Gagal memprediksi: {e}")
    st.divider()
    st.subheader("2. PREDIKSI BATCH (BANYAK DATA)")
    csv_tmpl = pd.DataFrame(columns=TemplatePrediksi).to_csv(index=False, sep=';')
    st.download_button("📥 FORMAT PREDIKSI", csv_tmpl, "template_prediksi.csv", help="Format: Identitas + Fitur (Tanpa kolom Status)")
    # --- 2. UPLOAD FILE ---
    batch_file = st.file_uploader("Upload File CSV Data Mahasiswa", type=["csv"])
    if batch_file is not None:
        # --- A. VALIDASI & LOAD DATA ---
        try:
            batch_file.seek(0)
            df_batch = pd.read_csv(batch_file, sep=None, engine='python')
            # Cek Kolom
            uploaded_cols = [c.lower().strip() for c in df_batch.columns]
            missing_cols = [c for c in VALIDASIPREDIKSI if c not in uploaded_cols]
            if missing_cols:
                st.error("⛔ **FILE DITOLAK: Struktur Data Tidak Sesuai! Silahkan Download dan gunakan template **")
                st.warning(f"Sistem membutuhkan kolom berikut:\n\n`{', '.join(missing_cols)}`")
                st.stop() # Berhenti jika validasi gagal
            # Jika Lolos
            st.success("✅ File Valid! Siap diproses.")
            st.info("📄 Preview Data Asli")
            st.dataframe(df_batch.head())
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()
        st.divider()
        if st.button("🚀 JALANKAN PREDIKSI BATCH", type="primary"):
            if model_used is None:
                st.error("⚠️ Model belum dimuat! Harap training ulang.")
            else:
                with st.spinner("Sedang memproses data..."):
                    try:
                        # 1. Preprocessing
                        X_batch_clean, _ = preprocess(df_batch)
                        st.success("✅ Data berhasil dibersihkan!")
                        with st.expander("Lihat Hasil Preprocessing:"):
                            st.dataframe(X_batch_clean.head()) 
                        # 2. Buang Target (Safety Net)
                        if TARGET in X_batch_clean.columns:
                            X_batch_clean = X_batch_clean.drop(columns=[TARGET])
                        # 3. ALIGNMENT KOLOM (PENTING!)
                        X_final = X_batch_clean.reindex(columns=VALIDASIPREDIKSI, fill_value=0)
                        # 4. Prediksi
                        y_pred = model_used.predict(X_final)
                        # 5. Gabungkan Hasil ke Data Asli
                        df_result = df_batch.copy()
                        df_result["STATUS_PREDIKSI"] = ["RISIKO SISIP" if x == 1 else "AMAN" for x in y_pred]
                        cols_to_drop = [c for c in df_result.columns if c.lower().strip() == TARGET.lower().strip()]
                        if cols_to_drop:
                            df_result = df_result.drop(columns=cols_to_drop)
                        # Tampilkan & Download
                        st.success(f"✅ Prediksi Selesai! ({len(df_result)} Data)")
                        st.dataframe(df_result)
                        st.download_button(
                            "📥 DOWNLOAD HASIL LENGKAP",
                            df_result.to_csv(index=False, sep=';'),
                            "hasil_prediksi_mahasiswa.csv"
                        )
                    except Exception as e:
                        st.error(f"Gagal saat kalkulasi: {e}")