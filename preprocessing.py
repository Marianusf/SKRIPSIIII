import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================
# FITUR YANG DIPAKAI
# =========================
SELECTED_FEATURES = [
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

TARGET = "status"


# =========================
# LOAD & VALIDATE DATA
# =========================
def load_data(path):

    try:
        df = pd.read_csv(path, sep=';')
        if len(df.columns) == 1:
            raise Exception("Separator salah")
    except:
        df = pd.read_csv(path, sep=',')

    df.columns = df.columns.str.strip().str.lower()

    print("Kolom terbaca:", df.columns.tolist())

    missing = [col for col in SELECTED_FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    df = df[SELECTED_FEATURES]

    return df


# =========================
# PREPROCESS UNTUK UI
# =========================
def preprocess_ui(df):

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if TARGET not in df.columns:
        raise ValueError(f"Kolom status tidak ditemukan: {df.columns.tolist()}")

    missing = [col for col in SELECTED_FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    df = df[SELECTED_FEATURES]

    return df


# =========================
# ENCODING TRAIN
# =========================
def encode_data(df, target_col):

    df = df.copy()
    encoders = {}

    for col in df.columns:

        if df[col].dtype == "object":

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

            encoders[col] = {
                "encoder": le,
                "classes": set(le.classes_)
            }

    return df, encoders


# =========================
# TRANSFORM DATA BARU (LEBIH AMAN)
# =========================
def transform_input(df, encoders):

    df = df.copy()

    for col in df.columns:

        if col in encoders:

            le = encoders[col]["encoder"]
            known_classes = encoders[col]["classes"]

            def safe_transform(x):
                x = str(x)
                if x in known_classes:
                    return le.transform([x])[0]
                else:
                    # fallback ke kelas pertama (lebih stabil dari -1)
                    return le.transform([list(known_classes)[0]])[0]

            df[col] = df[col].astype(str).apply(safe_transform)

    return df