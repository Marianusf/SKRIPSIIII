import pickle

from preprocessing import load_data, encode_data
from C45 import C45

# =========================
# LOAD DATA
# =========================

df = load_data("dataset.csv")

target = "status"

# =========================
# ENCODING (FULL DATA)
# =========================

df, encoders = encode_data(df, target)

X = df.drop(target, axis=1)
y = df[target]

# =========================
# TRAIN MODEL TERBAIK
# =========================

model = C45(min_samples_leaf=10)
model.fit(X, y)

# =========================
# SIMPAN MODEL
# =========================

with open("default_model.pkl", "wb") as f:
    pickle.dump((model, encoders), f)

print("✅ Default model berhasil dibuat!")











