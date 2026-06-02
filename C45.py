import numpy as np
import pandas as pd
from collections import Counter

# Konfigurasi Atribut Global (Sesuaikan dengan Dataset Mahasiswa Anda)
NUMERIC_COLS = ["ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3"]
CATEGORICAL_COLS = ["kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran"]
TARGET = "status"

class C45:
    def __init__(self, min_samples_leaf=10, min_gain=0.0):
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

    def entropy(self, y):
        if len(y) == 0: return 0
        y = np.array(y, dtype=int)
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def gain_ratio(self, parent, subsets):
        total = len(parent)
        if total == 0: return 0
        h_parent = self.entropy(parent)
        
        if h_parent < 1e-9: return 0
        
        h_split = 0
        split_info = 0
        for sub in subsets:
            if len(sub) == 0: continue
            w = len(sub) / total
            h_split += w * self.entropy(sub)
            split_info -= w * np.log2(w)
            
        gain = h_parent - h_split
        
        if gain < 1e-9: return 0 
        if split_info < 1e-9: return 0 
        
        return gain / split_info

    def evaluate_categorical(self, X_col, y):
        """
        Gaya C4.5 Kolektif: Menghitung Gain Ratio secara utuh melibatkan semua cabang 
        terlebih dahulu agar nilai kepentingannya keluar murni (seperti tabel manual Anda)
        """
        unique_vals = np.unique(X_col)
        subsets = [y[X_col == val] for val in unique_vals]
                
        g = self.gain_ratio(y, subsets)
        return g, {"type": "categorical", "values": unique_vals}

    def evaluate_numeric(self, X_col, y):
        sorted_idx = np.argsort(X_col)
        X_col, y = X_col[sorted_idx], y[sorted_idx]
        best_g = -1.0
        best_info = None
        
        current_entropy = self.entropy(y)
        if current_entropy < 1e-9: 
            return -1.0, None  
        
        for i in range(1, len(X_col)):
            if X_col[i] == X_col[i-1]: continue
            thresh = (X_col[i] + X_col[i-1]) / 2
            left_mask = X_col <= thresh
            
            left, right = y[left_mask], y[~left_mask]
            
            # Saringan threshold numerik ekstrem di awal
            if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                continue
                
            g = self.gain_ratio(y, [left, right])
            
            if g > best_g and g > self.min_gain:  
                best_g = g
                best_info = {"type": "numeric", "threshold": thresh}
                
        return best_g, best_info

    def information_gain_all_features(self, X, y):
        y_arr = np.array(y)
        res = {}
        for c in X.columns:
            if c in CATEGORICAL_COLS:
                g, _ = self.evaluate_categorical(X[c].values, y_arr)
            else:
                g, _ = self.evaluate_numeric(X[c].values, y_arr)
            res[c] = g if g > 0 else 0.0
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    def build_tree(self, X, y, available_features, parent_majority=0):
        y_arr = np.array(y)
        
        if len(y_arr) == 0:
            return int(parent_majority)
            
        counts = Counter(y_arr).most_common(1)
        current_majority = int(counts[0][0]) if counts else int(parent_majority)
        
        # Kondisi Henti Mutlak 1: Data sudah homogen (Murni)
        if self.entropy(y_arr) < 1e-9 or len(set(y_arr)) <= 1:
            return current_majority
            
        # Kondisi Henti Mutlak 2 (Penerapan Eksperimen min_samples_leaf):
        # Cabang mana pun (Numerik/Kategorikal) jika total datanya di bawah batas leaf,
        # potong pertumbuhannya di sini dan langsung jadikan DAUN KEPUTUSAN!
        if len(y_arr) < self.min_samples_leaf or len(available_features) == 0: 
            return current_majority
        
        best_g = -1.0
        best_f = None
        best_info = None 
        
        for col_name in available_features:
            X_val = X[col_name].values
            if col_name in CATEGORICAL_COLS:
                g, info = self.evaluate_categorical(X_val, y_arr)
            else:
                g, info = self.evaluate_numeric(X_val, y_arr)
                
            if g > best_g:
                best_g, best_f, best_info = g, col_name, info

        if best_g <= self.min_gain or best_f is None or best_info is None:
            return current_majority

        node = {
            "feature": best_f,
            "split_info": best_info,
            "samples": len(y_arr),
            "majority": current_majority,
            "branches": {}
        }

        if best_info["type"] == "categorical":
            next_features = [f for f in available_features if f != best_f]
            for val in best_info["values"]:
                mask = X[best_f] == val
                node["branches"][val] = self.build_tree(X[mask], y[mask], next_features, current_majority)
        else:
            thresh = best_info["threshold"]
            left_mask = X[best_f] <= thresh
            right_mask = X[best_f] > thresh
            
            node["branches"]["left"] = self.build_tree(X[left_mask], y[left_mask], available_features, current_majority)
            node["branches"]["right"] = self.build_tree(X[right_mask], y[right_mask], available_features, current_majority)
            
        return node

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        all_features = list(X.columns)
        init_majority = int(Counter(y_series).most_common(1)[0][0]) if len(y_series) > 0 else 0
        self.tree = self.build_tree(X, y_series, all_features, init_majority)

    def predict_one(self, row, tree):
        if not isinstance(tree, dict): 
            return tree
            
        feature_val = row[tree["feature"]]
        info = tree["split_info"]
        
        if info["type"] == "categorical":
            if feature_val not in tree["branches"]:
                return tree["majority"]
            return self.predict_one(row, tree["branches"][feature_val])
        else:
            if feature_val <= info["threshold"]:
                return self.predict_one(row, tree["branches"]["left"])
            else:
                return self.predict_one(row, tree["branches"]["right"])

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for _, row in X.iterrows()])

    def print_tree(self, tree=None, indent=""):
        node = tree if tree is not None else self.tree
        if not isinstance(node, dict):
            print(f"{indent}PREDIKSI: {node}")
            return

        info = node["split_info"]
        
        if info["type"] == "categorical":
            vals = list(node["branches"].items())
            if len(vals) > 0:
                first_val, first_branch = vals[0]
                print(f"{indent}IF {node['feature']} == {first_val}:")
                self.print_tree(first_branch, indent + "  | ")
                
                for val, branch in vals[1:]:
                    print(f"{indent}ELSE IF {node['feature']} == {val}:")
                    self.print_tree(branch, indent + "  | ")
        else:
            print(f"{indent}IF {node['feature']} <= {info['threshold']:.3f}:")
            self.print_tree(node["branches"]["left"], indent + "  | ")
            print(f"{indent}ELSE (> {info['threshold']:.3f}):")
            self.print_tree(node["branches"]["right"], indent + "  | ")
