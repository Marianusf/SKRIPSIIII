import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.combine import SMOTEENN

from preprocessing import load_data, encode_data, transform_input
from C45 import C45


# =========================
# LOAD DATA
# =========================

df = load_data("Dataset Mahasiswa Sisipan - Copy of Data Final.csv")

target = "status"

X = df.drop(columns=[target])
y = df[target]


# =========================
# PARAMETER
# =========================

kfold_list = [3, 5, 10]
leaf_list = [5, 10, 20, 30]

results = []


# =========================
# LOOP EKSPERIMEN
# =========================

for k in kfold_list:
    for leaf in leaf_list:

        print(f"\n=== K={k} | Leaf={leaf} ===")

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        acc_no_smote = []
        prec_no_smote = []
        rec_no_smote = []

        acc_smote = []
        prec_smote = []
        rec_smote = []

        for train_idx, test_idx in kf.split(X, y):

            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # =========================
            # PREPROCESS (ANTI LEAKAGE)
            # =========================

            train_df = X_train.copy()
            train_df[target] = y_train

            test_df = X_test.copy()
            test_df[target] = y_test

            train_df, encoders = encode_data(train_df, target)
            test_df = transform_input(test_df, encoders)

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            # =========================
            # TANPA SMOTE
            # =========================

            model1 = C45(min_samples_leaf=leaf)
            model1.fit(X_train, y_train)

            pred1 = model1.predict(X_test)

            # HANDLE NONE
            majority = y_train.mode()[0]
            pred1 = [p if p is not None else majority for p in pred1]

            acc_no_smote.append(accuracy_score(y_test, pred1))
            prec_no_smote.append(precision_score(y_test, pred1, zero_division=0))
            rec_no_smote.append(recall_score(y_test, pred1, zero_division=0))

            # =========================
            # DENGAN SMOTE-ENN
            # =========================

            try:
                sm = SMOTEENN(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)

                model2 = C45(min_samples_leaf=leaf)
                model2.fit(X_res, y_res)

                pred2 = model2.predict(X_test)
                pred2 = [p if p is not None else majority for p in pred2]

                acc_smote.append(accuracy_score(y_test, pred2))
                prec_smote.append(precision_score(y_test, pred2, zero_division=0))
                rec_smote.append(recall_score(y_test, pred2, zero_division=0))

            except Exception as e:
                print("SMOTE ERROR:", e)
                continue  # ⬅️ ini lebih benar daripada isi 0

        # =========================
        # SIMPAN HASIL
        # =========================

        results.append({
            "K-Fold": k,
            "Min Leaf": leaf,

            "Acc Tanpa SMOTE": np.mean(acc_no_smote),
            "Prec Tanpa SMOTE": np.mean(prec_no_smote),
            "Recall Tanpa SMOTE": np.mean(rec_no_smote),

            "Acc SMOTE": np.mean(acc_smote) if acc_smote else 0,
            "Prec SMOTE": np.mean(prec_smote) if prec_smote else 0,
            "Recall SMOTE": np.mean(rec_smote) if rec_smote else 0,
        })


# =========================
# HASIL
# =========================

df_result = pd.DataFrame(results)

print("\n=== HASIL AKHIR ===")
print(df_result)

df_result.to_csv("hasil_eksperimen.csv", index=False)