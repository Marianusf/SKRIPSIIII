import numpy as np
import pandas as pd
from collections import Counter

class C45:
    def __init__(self, cat_features=[], min_samples_leaf=5, min_gain=0.001):
        """
        cat_features: List of indices (int) for categorical columns.
                      Contoh: [0, 2, 5] artinya kolom ke-0, 2, dan 5 adalah kategori.
        """
        self.tree = None
        self.cat_features = cat_features 
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.majority_class = None

    def entropy(self, y):
        if len(y) == 0: return 0
        # Pastikan tipe data integer untuk bincount
        y = np.array(y, dtype=int)
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def gain_ratio(self, parent, subsets):
        total = len(parent)
        if total == 0: return 0
        h_parent = self.entropy(parent)
        
        h_split, split_info = 0, 0
        for sub in subsets:
            if len(sub) == 0: continue
            w = len(sub) / total
            h_split += w * self.entropy(sub)
            split_info -= w * np.log2(w + 1e-9)
            
        gain = h_parent - h_split
        if split_info < 1e-9: return 0 # Hindari pembagian nol
        return gain / split_info

    # --- LOGIC 1: SPLIT NUMERIK (BINARY) ---
    def best_numeric_split(self, X_col, y):
        # Mengurutkan data untuk mencari threshold terbaik
        sorted_idx = np.argsort(X_col)
        X_col, y = X_col[sorted_idx], y[sorted_idx]
        best_g, best_t = -1, None
        
        # Cek setiap kemungkinan titik potong
        for i in range(1, len(X_col)):
            if X_col[i] == X_col[i-1]: continue
            thresh = (X_col[i] + X_col[i-1]) / 2
            
            left_mask = X_col <= thresh
            # Optimasi: Cek jumlah sampel sebelum hitung entropy
            if np.sum(left_mask) < self.min_samples_leaf or \
               (len(y) - np.sum(left_mask)) < self.min_samples_leaf:
                continue
                
            left, right = y[left_mask], y[~left_mask]
            g = self.gain_ratio(y, [left, right])
            
            if g > best_g: best_g, best_t = g, thresh
            
        return best_g, best_t

    # --- LOGIC 2: SPLIT KATEGORIKAL (MULTI-WAY) ---
    def calculate_categorical_gain(self, X_col, y):
        unique_vals = np.unique(X_col)
        # Jika variasi cuma 1, tidak bisa di-split
        if len(unique_vals) < 2: return -1, None
        
        subsets = []
        for val in unique_vals:
            subsets.append(y[X_col == val])
            
        # Cek min_samples_leaf untuk setiap cabang
        # (Opsional: C4.5 asli membiarkan ini, tapi kita safety check)
        if any(len(s) < 1 for s in subsets): # Minimal ada data
             pass 

        g = self.gain_ratio(y, subsets)
        return g, unique_vals

    def build_tree(self, X, y):
        y_arr = np.array(y)
        # Base Case 1: Semua label sama
        if len(set(y_arr)) <= 1: return int(y_arr[0])
        # Base Case 2: Data kurang dari min_samples
        if len(y_arr) < self.min_samples_leaf: return int(Counter(y_arr).most_common(1)[0][0])

        best_g = -1
        best_f = None
        best_criteria = None 
        split_type = "numeric" # default

        # Iterasi semua kolom
        for col_name in X.columns:
            col_idx = X.columns.get_loc(col_name) # Ambil index kolom (0, 1, 2...)
            X_val = X[col_name].values
            
            # CEK TIPE FITUR
            if col_idx in self.cat_features:
                # ---> Jalur Kategori (Multi-way)
                g, branches = self.calculate_categorical_gain(X_val, y_arr)
                if g > best_g:
                    best_g, best_f, best_criteria = g, col_name, branches
                    split_type = "categorical"
            else:
                # ---> Jalur Numerik (Binary)
                g, t = self.best_numeric_split(X_val, y_arr)
                if g > best_g:
                    best_g, best_f, best_criteria = g, col_name, t
                    split_type = "numeric"

        # Base Case 3: Tidak ada Gain yang bagus
        if best_g < self.min_gain or best_f is None:
            return int(Counter(y_arr).most_common(1)[0][0])

        # KONSTRUKSI NODE
        node = {
            "feature": best_f,
            "type": split_type,
            "samples": len(y_arr),
            "majority": int(Counter(y_arr).most_common(1)[0][0]) # Simpan untuk fallback
        }

        if split_type == "numeric":
            node["threshold"] = best_criteria
            l_idx = X[best_f] <= best_criteria
            r_idx = X[best_f] > best_criteria
            
            node["left"] = self.build_tree(X[l_idx], y[l_idx])
            node["right"] = self.build_tree(X[r_idx], y[r_idx])
        else:
            node["branches"] = {}
            unique_vals = best_criteria
            for val in unique_vals:
                idx = X[best_f] == val
                # Rekursif ke setiap nilai unik
                node["branches"][val] = self.build_tree(X[idx], y[idx])
                
        return node

    def fit(self, X, y):
        self.majority_class = int(Counter(y).most_common(1)[0][0])
        self.tree = self.build_tree(X, y)

    def predict_one(self, row, tree):
        if not isinstance(tree, dict): return tree
        
        feature_val = row[tree["feature"]]
        
        if tree["type"] == "numeric":
            if feature_val <= tree["threshold"]:
                return self.predict_one(row, tree["left"])
            else:
                return self.predict_one(row, tree["right"])
        else:
            # Logic Kategori
            # Cari cabang yang sesuai nilai
            if feature_val in tree["branches"]:
                return self.predict_one(row, tree["branches"][feature_val])
            else:
                # Fallback: Jika ketemu nilai kategori baru yang tidak ada saat training
                # Kembalikan majority class dari node saat ini
                return tree["majority"]

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for _, row in X.iterrows()])

    # --- VISUALISASI POHON ---
    def print_tree(self, tree=None, indent=""):
        node = tree if tree is not None else self.tree
        if not isinstance(node, dict):
            print(f"{indent}PREDIKSI: {node}")
            return

        if node["type"] == "numeric":
            print(f"{indent}IF {node['feature']} <= {node['threshold']:.3f}:")
            self.print_tree(node["left"], indent + "  | ")
            print(f"{indent}ELSE (> {node['threshold']:.3f}):")
            self.print_tree(node["right"], indent + "  | ")
        else:
            print(f"{indent}CASE {node['feature']}:")
            for val, child in node["branches"].items():
                print(f"{indent}  = {val}:")
                self.print_tree(child, indent + "    | ")
                
                
    def information_gain_all_features(self, X, y):
        res = {c: self.best_numeric_split(X[c].values, np.array(y))[0] for c in X.columns}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
