import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt


# Import fungsi dari preprocessing.py yang sudah kita perbaiki
from preprocessing import (TARGET,load_data,preprocess,get_cat_indices,select_atribut,)
from C45 import C45
print("\n=== 1. LOADING DATA ===")
df_raw = load_data("Dataset Mahasiswa Sisipan2.csv")

if df_raw is None:
    print("Gagal membaca data. Pastikan file CSV ada di folder yang sama.")
    exit()

print(f"Total Data Awal: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
print("\n=== INFO KOLOM, TIPE, & CONTOH DATA ===")
isi_data = pd.DataFrame({
    'Tipe Data': df_raw.dtypes,
    'Contoh Isi': df_raw.iloc[0]
})
print(isi_data)
print("\nJumlah Missing Values:")
print(df_raw.isnull().sum())

print("\n=== 2. SELEKSI ATRIBUT ===")
df_selected = select_atribut(df_raw)
print("Atribut Terpilih (ke bawah):")
print("\n".join(df_selected.columns)) # List atribut ke bawah
print(f"\nSisa Data: {df_selected.shape[0]} baris")

print("\n=== 3. PREPROCESSING ===")
df, target_map = preprocess(df_selected)
print("Data Siap Training:")
print(df.head().to_string(index=False))
print("\nTipe Data Akhir:\n")
print(df.dtypes.to_string())

# split fitur dan target
if TARGET not in df.columns:
    print(f"Error: Kolom Target '{TARGET}' hilang!")
    exit()

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Cari index kolom kategorikal
cat_idx = get_cat_indices(X)
print("\nDistribusi Kelas:")
print(y.value_counts().to_string())


print("\n=== 4. SPLIT DATA (80% Latih : 20% Uji Final) ===")
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.20,
    stratify=y,
    random_state=42
)
print(f"Data Latih: {len(X_train_full)} baris")
print(f"Data Uji Final : {len(X_test_final)} baris")


print("\n=== 5. MULAI EKSPERIMEN (MENCARI METODE TERBAIK) ===")
kfold_list = [3, 5, 10]
leaf_list = [1,2,3,4,5,6,7,8,9,10]
results_table = []


# K-Fold loop di lapisan paling luar
for k in kfold_list:
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # Siapkan penampung memori untuk menyimpan hasil semua fold yang sudah di-SMOTE-ENN
    # agar tidak perlu dihitung berulang-ulang secara mubazir
    folds_data = []
    # Jalankan loop pembagian fold dan proses SMOTE-ENN HANYA SEKALI per nilai K
    fold_idx = 1
    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        XT, Xv = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        yT, yv = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        try:
            s_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
            s_enn = SMOTEENN(smote=s_nc, random_state=42)
            # Proses SMOTE-ENN dijalankan murni sekali di sini
            XR, yR = s_enn.fit_resample(XT, yT)
            # CETAK INFO RESAMPLING HANYA PADA FOLD PERTAMA DARI K-FOLD INI
            if fold_idx == 1:
                XR_smote, yR_smote = s_nc.fit_resample(XT, yT)
                print(f"   INFO RESAMPLING UNTUK K-FOLD = {k}")
                print(f"==========================================")
                print(f"Data train asli      : {len(XT)} baris")
                print(f"Data setelah SMOTE   : {len(XR_smote)} baris")
                print(f"Data setelah SMOTEENN: {len(XR)} baris")  
                dist_df = pd.DataFrame({
                    'Asli': yT.value_counts(),
                    'SMOTE': pd.Series(yR_smote).value_counts(),
                    'SMOTE-ENN': pd.Series(yR).value_counts()
                }).fillna(0).astype(int)
                print("\nPerubahan Distribusi Kelas:")
                print(dist_df)
                print(f"==========================================\n")
            # Simpan data yang sudah matang ke dalam list penampung
            folds_data.append((XT, yT, XR, yR, Xv, yv))
        except Exception as e:
            print(f"Error Preprocessing di K={k}, Fold={fold_idx}: {e}")
        fold_idx += 1
        
    # SEKARANG KITA JALANKAN PERULANGAN LEAF MENGGUNAKAN DATA YANG SUDAH JADI
    for leaf in leaf_list:
        m = {
            "normal": {"f1": [], "acc": [], "rec": [], "pr": []},
            "smoteenn": {"f1": [], "acc": [], "rec": [], "pr": []}
        }  
        # Susur kembali list data yang sudah matang tadi
        for XT, yT, XR, yR, Xv, yv in folds_data:
            # 1. Model tanpa resampling (Normal)
            model = C45(min_samples_leaf=leaf)
            model.fit(XT, yT)
            p = model.predict(Xv)
            m["normal"]["f1"].append(f1_score(yv, p, zero_division=0))
            m["normal"]["acc"].append(accuracy_score(yv, p))
            m["normal"]["rec"].append(recall_score(yv, p, zero_division=0))   
            m["normal"]["pr"].append(precision_score(yv, p, zero_division=0))  
            
            # 2. Model dengan SMOTE-ENN (Tinggal fit model C4.5 nya saja, data XR dan yR sudah tersedia!)
            model_s = C45(min_samples_leaf=leaf)
            model_s.fit(XR, yR)
            ps = model_s.predict(Xv)
            m["smoteenn"]["f1"].append(f1_score(yv, ps, zero_division=0))
            m["smoteenn"]["acc"].append(accuracy_score(yv, ps))
            m["smoteenn"]["rec"].append(recall_score(yv, ps, zero_division=0))
            m["smoteenn"]["pr"].append(precision_score(yv, ps, zero_division=0))
            
        # Hitung rata-rata hasil untuk disimpan ke tabel hasil akhir
        for method in ["normal", "smoteenn"]:
            results_table.append({
                "K": k,
                "Leaf": leaf,
                "Method": method,
                "Acc_mean": np.mean(m[method]["acc"]) if m[method]["acc"] else 0,
                "Rec_mean": np.mean(m[method]["rec"]) if m[method]["rec"] else 0,
                "Pr_mean": np.mean(m[method]["pr"]) if m[method]["pr"] else 0,
                "F1_mean": np.mean(m[method]["f1"]) if m[method]["f1"] else 0
            })

# Tampilkan rangkuman seluruh eksperimen
if results_table:
    results_df = pd.DataFrame(results_table)
    print("\n=== HASIL LENGKAP EKSPERIMEN ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:.4f}'.format
    print(results_df.sort_values(by="F1_mean", ascending=False).to_string(index=False))
else:
    print("Gagal, tidak ada hasil eksperimen.")
    exit()

# Mengambil parameter terbaik berdasarkan F1-Score tertinggi
best = results_df.sort_values(by="F1_mean", ascending=False).iloc[0]
print("\n=== 6. UJI FINAL ===")
print(f"Parameter Terbaik -> Metode: {best['Method'].upper()} | K-Fold: {int(best['K'])} | Leaf: {int(best['Leaf'])}")

# Inisialisasi data latih final
X_final_train = X_train_full
y_final_train = y_train_full

if best["Method"] == "smoteenn":
    try:
        senn=SMOTEENN(smote=SMOTENC(categorical_features=cat_idx, random_state=42), random_state=42)
        X_final_train, y_final_train = senn.fit_resample(X_train_full, y_train_full)
        print("-> Resampling SMOTE + ENN (SMOTEENN) diterapkan pada data latih final.")
    except Exception as e:
        print(f"Gagal melakukan resampling final: {e}")

# Melatih model final dengan parameter terbaik
final_model = C45(min_samples_leaf=int(best["Leaf"]))
final_model.fit(X_final_train, y_final_train)

# Prediksi data uji final (hold-out 20%)
y_pred_final = final_model.predict(X_test_final)

# Evaluasi performa akhir model
acc_final = accuracy_score(y_test_final, y_pred_final)
pr_final = precision_score(y_test_final, y_pred_final)
rec_final = recall_score(y_test_final, y_pred_final)
f1_final = f1_score(y_test_final, y_pred_final)

print("\nHASIL EVALUASI MODEL FINAL:")
print(f"Akurasi  : {acc_final:.4f}")
print(f"Presisi : {pr_final:.4f}")
print(f"Recall    : {rec_final:.4f}")
print(f"F1-Skor  : {f1_final:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_final, y_pred_final)
unique_classes = np.unique(y_test_final)
cm_df = pd.DataFrame(
    cm, 
    index=[f'Aktual {c}' for c in unique_classes], 
    columns=[f'Prediksi {c}' for c in unique_classes]
)
print(cm_df.to_string())

print("\n=== 7. PENYIMPANAN MODEL ===")
with open('default_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model berhasil disimpan ke 'default_model.pkl'")
print("\n=== STRUKTUR POHON KEPUTUSAN ===")
final_model.print_tree()

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["Tidak Sisip", "Sisip"]
# )
# fig, ax = plt.subplots(figsize=(6,6))
# disp.plot(ax=ax, cmap="Blues", values_format="d")
# ax.set_title("Confusion Matrix: Pengujian Model Final C4.5")
# ax.set_xlabel("Prediksi")
# ax.set_ylabel("Data Aktual")
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

