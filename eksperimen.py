import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC

from preprocessing import TARGET, load_data, preprocess, get_cat_indices, select_features
from C45 import C45
# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================

# Load dataset asli
df_raw = load_data("Dataset Mahasiswa Sisipan.csv")

print("\n=== ATRIBUT DATASET ASLI ===")
print(df_raw.columns.tolist())

# Informasi dataset asli
print("\n=== INFORMASI DATASET ASLI ===")
print(f"Jumlah Baris  : {df_raw.shape[0]}")
print(f"Jumlah Kolom  : {df_raw.shape[1]}")

# Nama atribut
print("\n=== DAFTAR ATRIBUT DATASET ASLI ===")
for i, col in enumerate(df_raw.columns, 1):
    print(f"{i}. {col}")

# Tipe data
print("\n=== TIPE DATA ATRIBUT ===")
print(df_raw.dtypes.to_string())

# Missing value
print("\n=== JUMLAH MISSING VALUE ===")
print(df_raw.isnull().sum().to_string())

# ==========================================
# SELEKSI FITUR
# ==========================================

df_selected = select_features(df_raw)

print("\n=== FITUR TERPILIH ===")
for i, col in enumerate(df_selected.columns, 1):
    print(f"{i}. {col}")

print("\n=== DATASET SETELAH SELEKSI FITUR ===")
print(f"Jumlah Baris  : {df_selected.shape[0]}")
print(f"Jumlah Kolom  : {df_selected.shape[1]}")

print("\n5 Data Pertama:")
print(df_selected.head().to_string())

# ==========================================
# PREPROCESSING
# ==========================================

df, target_map = preprocess(df_selected)

print("\n=== DATASET SETELAH PREPROCESSING ===")
print(df.head().to_string())
print(df.dtypes.to_string())
# ==========================================
# PEMISAHAN FITUR & TARGET
# ==========================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

cat_idx = get_cat_indices(X)

print("\n=== FITUR YANG DIGUNAKAN MODEL ===")
print(X.columns.tolist())

print("\n=== DISTRIBUSI KELAS ASLI ===")
print(y.value_counts().to_string())

X = df.drop(columns=[TARGET])
y = df[TARGET]
cat_idx = get_cat_indices(X)


# ==========================================
# 2. SPLIT DATA AWAL (HOLD-OUT 80:20)
# ==========================================
# Pisahkan 20% data sebagai Data Testing Akhir (suci/belum pernah dilihat model)
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Data Latih (K-Fold): {X_train_full.shape[0]} baris")
print(f"Data Uji Final (Hold-out): {X_test_final.shape[0]} baris")

# ==========================================
# 3. DEMO INFORMATION GAIN (PADA DATA LATIH)
# ==========================================
print("\n=== RANKING FITUR (INFORMATION GAIN) ===")
# Resample sejenak pada data latih untuk melihat IG yang seimbang
sm_nc_demo = SMOTENC(categorical_features=cat_idx, random_state=42)
sm_enn_demo = SMOTEENN(smote=sm_nc_demo, random_state=42)
XR_demo, yR_demo = sm_enn_demo.fit_resample(X_train_full, y_train_full)

X_res_df = pd.DataFrame(XR_demo, columns=X.columns)
model_temp = C45()
ig_results = model_temp.information_gain_all_features(X_res_df, yR_demo)
for i, (feat, score) in enumerate(ig_results.items(), 1):
    print(f"{i}. {feat:25}: {score:.6f}")

# ==========================================
# 4. LOOP EKSPERIMEN (K-FOLD PADA DATA LATIH)
# ==========================================
kfold_list = [3, 5, 10]
leaf_list = [5,10,20,30]
results_table = []

print("\n=== MEMULAI EKSPERIMEN K-FOLD (MENCARI PARAMETER TERBAIK) ===")

for k in kfold_list:
    for leaf in leaf_list:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        m = {
            "normal": {"f1": [], "acc": []}, 
            "smote": {"f1": [], "acc": []}
        }

        for train_idx, val_idx in kf.split(X_train_full, y_train_full):
            XT, Xv = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            yT, yv = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            #A. METODE NORMAL
            model = C45(min_samples_leaf=leaf)
            model.fit(XT, yT)
            p = model.predict(Xv)
            m["normal"]["f1"].append(f1_score(yv, p, average='macro'))
            m["normal"]["acc"].append(accuracy_score(yv, p))

            #B. METODE SMOTE-ENN
            s_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
            s_enn = SMOTEENN(smote=s_nc, random_state=42)
            XR, yR = s_enn.fit_resample(XT, yT)
            
            model_s = C45(min_samples_leaf=leaf)
            model_s.fit(XR, yR)
            ps = model_s.predict(Xv)
            m["smote"]["f1"].append(f1_score(yv, ps, average='macro'))
            m["smote"]["acc"].append(accuracy_score(yv, ps))

        for method in ["normal", "smote"]:
            results_table.append({
                "K": k,
                "Leaf": leaf,
                "Method": method,
                "Acc_mean": np.mean(m[method]["acc"]),
                "F1_mean": np.mean(m[method]["f1"]), 
            })

results_df = pd.DataFrame(results_table)
print("\n=== HASIL EKSPERIMEN (DIURUTKAN BERDASARKAN F1) ===")
print(results_df.sort_values(by="F1_mean", ascending=False))

# ==========================================
# 5. PENGUJIAN AKHIR (HOLD-OUT TEST)
# ==========================================
best = results_df.sort_values(by="F1_mean", ascending=False).iloc[0]
print(f"\n--- Model Terpilih Untuk Uji Final: {best['Method']} | Leaf: {best['Leaf']} ---")

# Latih ulang model terbaik pada SELURUH Data Latih (80%)
X_final_train, y_final_train = X_train_full, y_train_full
if best["Method"] == "smote":
    snc = SMOTENC(categorical_features=cat_idx, random_state=42)
    senn = SMOTEENN(smote=snc, random_state=42)
    X_final_train, y_final_train = senn.fit_resample(X_train_full, y_train_full)

final_model = C45(min_samples_leaf=int(best["Leaf"]))
final_model.fit(X_final_train, y_final_train)

# Prediksi pada Data Uji Final (20%)
y_pred_final = final_model.predict(X_test_final)

print("\n=== HASIL EVALUASI AKHIR (PADA DATA TESTING 20%) ===")
print(f"Accuracy  : {accuracy_score(y_test_final, y_pred_final):.4f}")
print(f"Recall    : {recall_score(y_test_final, y_pred_final, average='macro'):.4f}")
print(f"F1-Score  : {f1_score(y_test_final, y_pred_final, average='macro'):.4f}")

print("\nConfusion Matrix (Hold-out Test):")
print(confusion_matrix(y_test_final, y_pred_final))

# ==========================================
# 6. SIMPAN MODEL & CETAK POHON
# ==========================================
with open('default_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("\n=== STRUKTUR POHON KEPUTUSAN FINAL ===")
final_model.print_tree()
