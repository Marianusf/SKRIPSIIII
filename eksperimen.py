import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC

# Import fungsi dari preprocessing.py yang sudah kita perbaiki
from preprocessing import (TARGET,load_data,preprocess,get_cat_indices,select_features,)
from C45 import C45
print("\n=== 1. LOADING DATA ===")
df_raw = load_data("Dataset Mahasiswa Sisipan2.csv")

if df_raw is None:
    print("Gagal membaca data. Pastikan file CSV ada di folder yang sama.")
    exit()
    
# 1. load data
print(f"Total Data Awal: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
print("\n=== INFO KOLOM, TIPE, & CONTOH DATA ===")
isi_data = pd.DataFrame({
    'Tipe Data': df_raw.dtypes,
    'Contoh Isi': df_raw.iloc[0]
})
print(isi_data)
print("\nJumlah Missing Values:")
print(df_raw.isnull().sum())

print("\n=== 2. SELEKSI FITUR ===")
df_selected = select_features(df_raw)
print("Fitur Terpilih (ke bawah):")
print("\n".join(df_selected.columns)) # List fitur ke bawah
print(f"\nSisa Data: {df_selected.shape[0]} baris")

print("\n=== 3. PREPROCESSING ===")
df, target_map = preprocess(df_selected)
print("Data Siap Training:")
print(df.head().to_string())
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

min_class_count = y.value_counts().min()
# Jika data minoritas terlalu kecil
k_neighbors_val = 1 if min_class_count < 6 else 5

print("\n=== 4. SPLIT DATA (80% Latih : 20% Uji Final) ===")
# Stratify dimatikan jika ada kelas hanya 1 data
stratify_param = y if min_class_count > 1 else None
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.20,
    stratify=stratify_param,
    random_state=42
)
print(f"Data Latih: {len(X_train_full)} baris")
print(f"Data Uji Final : {len(X_test_final)} baris")


print("\n=== 5. CEK INFORMATION GAIN (PENTINGNYA FITUR) ===")
try:
    sm_demo = SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neighbors_val)
    sm_enn_demo = SMOTEENN(smote=sm_demo, random_state=42)
    XR_demo, yR_demo = sm_enn_demo.fit_resample(X_train_full, y_train_full)
    X_res_df = pd.DataFrame(XR_demo, columns=X.columns)
    model_temp = C45(min_samples_leaf=3)
    ig_results = model_temp.information_gain_all_features(X_res_df, yR_demo)
    ig_df = pd.DataFrame(list(ig_results.items()), columns=['Fitur', 'Gain Score'])
    print(ig_df.to_string(index=False))
except Exception as e:
    print(f"Gagal menghitung IG: {e}")

print("\n=== 6. MULAI EKSPERIMEN (MENCARI METODE TERBAIK) ===")
kfold_list = [3, 5, 10]
leaf_list = [3, 5, 7, 10]
results_table = []
# Maksimum K berdasarkan jumlah kelas minoritas
max_k = y_train_full.value_counts().min()
# Agar info resampling hanya tampil sekali
sudah_print = False
# K-Fold loop untuk tuning parameter
for k in kfold_list:
    if k > max_k:
        print(f"[SKIP] K-Fold {k} tidak bisa jalan karena kelas minoritas terlalu sedikit.")
        continue
        
    for leaf in leaf_list:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        # Penampung metrik evaluasi tiap fold
        m = {
            "normal": {"f1": [], "acc": [], "rec": [], "pr": []},
            "smote": {"f1": [], "acc": [], "rec": [], "pr": []}
        }
        try:
            for train_idx, val_idx in kf.split(X_train_full, y_train_full):
                XT, Xv = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
                yT, yv = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
                
                # 1. Model tanpa resampling (Normal)
                model = C45(min_samples_leaf=leaf)
                model.fit(XT, yT)
                p = model.predict(Xv)
                m["normal"]["f1"].append(f1_score(yv, p, average='macro'))
                m["normal"]["acc"].append(accuracy_score(yv, p))
                m["normal"]["rec"].append(recall_score(yv, p, average='macro'))   
                m["normal"]["pr"].append(precision_score(yv, p, average='macro'))  
                # 2. Model dengan SMOTE-ENN
                try:
                    s_nc = SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neighbors_val)
                    s_enn = SMOTEENN(smote=s_nc, random_state=42)
                    # Print info resampling hanya pada fold pertama saja
                    if not sudah_print:
                        XR_smote, yR_smote = s_nc.fit_resample(XT, yT)
                        XR, yR = s_enn.fit_resample(XT, yT)
                        print("\n=== CEK RESAMPLING FOLD ===")
                        print(f"Data train fold asli : {len(XT)} baris")
                        print(f"Data setelah SMOTE   : {len(XR_smote)} baris")
                        print(f"Data setelah SMOTEENN: {len(XR)} baris")
                        dist_df = pd.DataFrame({
                            'Asli': yT.value_counts(),
                            'SMOTE': pd.Series(yR_smote).value_counts(),
                            'SMOTE-ENN': pd.Series(yR).value_counts()
                        }).fillna(0).astype(int)
                        print("\nPerubahan Distribusi Kelas:")
                        print(dist_df)
                        sudah_print = True
                    else:
                        XR, yR = s_enn.fit_resample(XT, yT)
                        
                    model_s = C45(min_samples_leaf=leaf)
                    model_s.fit(XR, yR)
                    ps = model_s.predict(Xv)
                    
                    m["smote"]["f1"].append(f1_score(yv, ps, average='macro'))
                    m["smote"]["acc"].append(accuracy_score(yv, ps))
                    m["smote"]["rec"].append(recall_score(yv, ps, average='macro'))
                    m["smote"]["pr"].append(precision_score(yv, ps, average='macro'))
                    
                except Exception as e:
                    print(f"Error SMOTE pada K={k}, Leaf={leaf}: {e}")
                    m["smote"]["f1"].append(0)
                    m["smote"]["acc"].append(0)
                    m["smote"]["rec"].append(0)
                    m["smote"]["pr"].append(0)
                    
        except Exception as e:
            print(f"Skip K={k}, Leaf={leaf}: {e}")
            continue
            
        # Hitung rata-rata hasil fold untuk disimpan ke tabel
        for method in ["normal", "smote"]:
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
print("\n=== 7. UJI FINAL ===")
print(f"Parameter Terbaik -> Metode: {best['Method'].upper()} | K-Fold: {int(best['K'])} | Leaf: {int(best['Leaf'])}")

# Inisialisasi data latih final
X_final_train = X_train_full
y_final_train = y_train_full

# Terapkan resampling jika metode terbaiknya adalah SMOTE
if best["Method"] == "smote":
    try:
        snc = SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k_neighbors_val)
        senn = SMOTEENN(smote=snc, random_state=42)
        X_final_train, y_final_train = senn.fit_resample(X_train_full, y_train_full)
        print("-> Resampling SMOTE + ENN diterapkan pada data latih final.")
    except Exception as e:
        print(f"Gagal melakukan resampling final: {e}")

# Melatih model final dengan parameter terbaik
final_model = C45(min_samples_leaf=int(best["Leaf"]))
final_model.fit(X_final_train, y_final_train)

# Prediksi data uji final (hold-out 20%)
y_pred_final = final_model.predict(X_test_final)

# Evaluasi performa akhir model
acc_final = accuracy_score(y_test_final, y_pred_final)
pr_final = precision_score(y_test_final, y_pred_final, average='macro')
rec_final = recall_score(y_test_final, y_pred_final, average='macro')
f1_final = f1_score(y_test_final, y_pred_final, average='macro')

print("\nHASIL EVALUASI MODEL FINAL:")
print(f"Accuracy  : {acc_final:.4f}")
print(f"Precision : {pr_final:.4f}")
print(f"Recall    : {rec_final:.4f}")
print(f"F1-Score  : {f1_final:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_final, y_pred_final)
unique_classes = np.unique(y_test_final)
cm_df = pd.DataFrame(
    cm, 
    index=[f'Aktual {c}' for c in unique_classes], 
    columns=[f'Prediksi {c}' for c in unique_classes]
)
print(cm_df.to_string())


print("\n=== 8. PENYIMPANAN MODEL ===")
with open('default_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model berhasil disimpan ke 'default_model.pkl'")
print("\n=== STRUKTUR POHON KEPUTUSAN ===")
final_model.print_tree()