import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

SELECTED_FEATURES = [
    "kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran", 
    "ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3", 
    "status"
]

NUMERIC_COLS = ["ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3"]
CATEGORICAL_COLS = ["kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran"]
TARGET = "status"

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        print(f"[ERROR] Gagal load data: {e}")
        return None

def select_features(df):
    df.columns = df.columns.str.strip().str.lower()
    valid_cols = [col for col in SELECTED_FEATURES if col in df.columns]
    return df[valid_cols].copy()

def get_cat_indices(df_x):
    return [df_x.columns.get_loc(c) for c in CATEGORICAL_COLS if c in df_x.columns]

# ==========================================
# TAHAP 1: CLEANING
# ==========================================
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    # NUMERIK
    for col in NUMERIC_COLS:
        if col in df.columns:
            if df[col].dtype == 'O':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            val = df[col].median()
            if pd.isna(val): val = 0
            df[col] = df[col].fillna(val)
    # KATEGORI
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace('nan', np.nan)
            modus = df[col].mode()
            val = modus[0] if not modus.empty else "lain-lain"
            df[col] = df[col].fillna(val)

    return df

# ==========================================
# TAHAP 2: TRANSFORMATION
# ==========================================
def transform_data(df):
    df = df.copy()
    if "kepulauan asal lahir" in df.columns:
        m = {'jawa': 1, 'sumatera': 2,'bali & ntt':3, 'kalimantan': 4, 'sulawesi': 5, 
        'papua & maluku': 6, 'lain-lain': 7}
        df["kepulauan asal lahir"] = df["kepulauan asal lahir"].map(m).fillna(7)
    if "jurusan sekolah" in df.columns:
        m = {'sma': 1, 'smk': 2, 'homeschooling': 3, 'home schooling': 3, 'lain-lain': 4}
        df["jurusan sekolah"] = df["jurusan sekolah"].map(m).fillna(4)
    if "profil sekolah" in df.columns:
        m = {'negeri': 0, 'swasta': 1, 'lain-lain': 2}
        df["profil sekolah"] = df["profil sekolah"].map(m).fillna(2)
    if "jalur pendaftaran" in df.columns:
        m = {'raport': 0, 'tes': 1, 'lain-lain': 2}
        df["jalur pendaftaran"] = df["jalur pendaftaran"].map(m).fillna(2)
    if TARGET in df.columns:
        target_clean = df[TARGET].astype(str).str.strip().str.lower()
        map_target = {
            'tidak sisip': 0, 'sisip': 1, 
        }
        df[TARGET] = target_clean.map(map_target).fillna(0)
    # Seleksi Akhir & Type Casting
    features_to_use = [c for c in SELECTED_FEATURES if c in df.columns]
    df_final = df[features_to_use].copy()

    for col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

    return df_final

def preprocess(df_raw):
    df_clean = clean_data(df_raw)
    df_ready = transform_data(df_clean)
    return df_ready, {0: 'TIDAK SISIP', 1: 'SISIP'}
