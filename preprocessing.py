import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

SELECTED_FEATURES = ["kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran", "ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3", "status"]
NUMERIC_COLS = ["ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3"]
CATEGORICAL_COLS = ["kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran"]
TARGET = "status"

def load_data(path):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    return df
def select_features(df):
    return df[
        [c for c in SELECTED_FEATURES if c in df.columns]
    ].copy()

def preprocess(df):
    df = df.copy()
    
    # 1. Bersihkan Nama Kolom (Penting!)
    df.columns = df.columns.str.strip().str.lower()
    # selected fitur
    df = df[[c for c in SELECTED_FEATURES if c in df.columns]]
    # 2. HANDLING NUMERIC
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # 3. MANUAL MAPPING (Gunakan case-insensitive agar lebih aman)
    # Kita paksa data menjadi lowercase dulu sebelum di-replace
    if "kepulauan asal lahir" in df.columns:
        # Masukkan semua variasi ejaan yang mungkin ada di CSV ke dalam huruf kecil
        map_pulau = {
            'jawa': 1, 
            'sumatra': 2, 'sumatera': 2, 
            'bali': 3, 'nusa tenggara': 3, 'bali & ntt': 3, 'bali & nusa tenggara': 3,
            'kalimantan': 4, 
            'sulawesi': 5, 
            'papua': 6, 'maluku': 6, 'papua & maluku': 6,
            'lain-lain': 7, 'lainnya': 7, 'luar negeri': 7
        }
        # Kecilkan data dulu, hapus spasi, baru di-map. Jika tidak ada di list, beri angka 7.
        df["kepulauan asal lahir"] = df["kepulauan asal lahir"].astype(str).str.strip().str.lower().map(map_pulau).fillna(7)

    if "jurusan sekolah" in df.columns:
        map_jurusan = {
            'sma': 1, 
            'smk': 2, 
            'homeschooling': 3, 'home schooling': 3, 'tidak keduanya': 3
        }
        df["jurusan sekolah"] = df["jurusan sekolah"].astype(str).str.strip().str.lower().map(map_jurusan).fillna(3)

    if "profil sekolah" in df.columns:
        map_profil = {'negeri': 0, 'swasta': 1}
        df["profil sekolah"] = df["profil sekolah"].astype(str).str.strip().str.lower().map(map_profil).fillna(2)

    if "jalur pendaftaran" in df.columns:
        map_jalur = {'raport': 0,'tes': 1}
        df["jalur pendaftaran"] = df["jalur pendaftaran"].astype(str).str.strip().str.lower().map(map_jalur).fillna(2)

    # --- TARGET ---
    if TARGET in df.columns:
        map_target = {'tidak sisip': 0, 'sisip': 1}
        df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map(map_target).fillna(0)

    # 5. PAKSA KONVERSI KE ANGKA (Langkah Terakhir)
    df = df.infer_objects(copy=False)
    for col in CATEGORICAL_COLS + ([TARGET] if TARGET in df.columns else []):
        if col in df.columns:
            # Jika replace gagal, baris ini akan mengubah sisa teks menjadi NaN lalu jadi 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df.reset_index(drop=True), {0: 'TIDAK SISIP', 1: 'SISIP'}

def get_cat_indices(df_x):
    return [df_x.columns.get_loc(c) for c in CATEGORICAL_COLS if c in df_x.columns]
