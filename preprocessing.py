import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

SELECTED_ATRIBUT = [
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

# 1. select_atribut: Ambil kolom yang valid dan terdaftar di SELECTED_ATRIBUT
def select_atribut(df):
    # Mengambil hanya kolom yang valid dan terdaftar di SELECTED_ATRIBUT
    valid_cols = [col for col in SELECTED_ATRIBUT if col in df.columns]
    return df[valid_cols].copy()

# 2. bersih_data: Bersihkan data dari missing value dan standarisasi format
def bersih_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    # 1. Bersihkan Kolom Target terlebih dahulu dari baris bising/kosong
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
        df = df[df[TARGET].isin(['tidak sisip', 'sisip'])].copy()
        
    # 2. Imputasi Kolom Numerik
    for col in NUMERIC_COLS:
        if col in df.columns:
            # Atasi jika ada angka yang ditulis menggunakan koma (format Indonesia)
            if df[col].dtype == 'O':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Isi missing value dengan Median, jika gagal isi 0
            median_val = df[col].median()
            if pd.isna(median_val): 
                median_val = 0
            df[col] = df[col].fillna(median_val)
            
    # 3. Imputasi Kolom Kategori
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace('nan', np.nan)
            
            # Isi missing value dengan Modus, jika tidak ada isi 'lain-lain'
            modus = df[col].mode()
            modus_val = modus[0] if not modus.empty else "lain-lain"
            df[col] = df[col].fillna(modus_val)
            
    return df

# TAHAP 3: TRANSFORMATION (Konversi ke Angka)
def transform_data(df):
    df = df.copy()
    if "kepulauan asal lahir" in df.columns:
        m = {'jawa': 1, 'sumatera': 2, 'bali & ntt': 3, 'kalimantan': 4, 
             'sulawesi': 5, 'papua & maluku': 6, 'lain-lain': 7}
        df["kepulauan asal lahir"] = df["kepulauan asal lahir"].map(m).fillna(7)
        
    if "jurusan sekolah" in df.columns:
        m = {'sma': 1, 'smk': 2, 'homeschooling': 3, 'lain-lain': 4}
        df["jurusan sekolah"] = df["jurusan sekolah"].map(m).fillna(4)
        
    if "profil sekolah" in df.columns:
        m = {'negeri': 1, 'swasta': 2, 'lain-lain': 3}
        df["profil sekolah"] = df["profil sekolah"].map(m).fillna(3)
        
    if "jalur pendaftaran" in df.columns:
        m = {'raport': 1, 'tes': 2, 'lain-lain': 3}
        df["jalur pendaftaran"] = df["jalur pendaftaran"].map(m).fillna(3)

    if TARGET in df.columns:
        map_target = {'tidak sisip': 0, 'sisip': 1}
        df[TARGET] = df[TARGET].map(map_target)
    features_to_use = [c for c in SELECTED_ATRIBUT if c in df.columns]
    
    return df[features_to_use].copy()

def get_cat_indices(df_x):
    return [df_x.columns.get_loc(c) for c in CATEGORICAL_COLS if c in df_x.columns]

def preprocess(df_raw):
    df_clean = bersih_data(df_raw)
    df_ready = transform_data(df_clean)
    return df_ready, {0: 'TIDAK SISIP', 1: 'SISIP'}
