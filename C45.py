import numpy as np
import pandas as pd
from collections import Counter

NUMERIC_COLS = ["ipk1", "ipk2", "ipk3", "jumlah matakuliah d/e/f", "jumlah sks d/e/f", "total sks semester 1-3"]
CATEGORICAL_COLS = ["kepulauan asal lahir", "jurusan sekolah", "profil sekolah", "jalur pendaftaran"]
TARGET = "status"

class C45:
    def __init__(self, min_samples_leaf=1):
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        
    def entropy(self, y):
        if len(y) == 0: 
            return 0
        S = np.array(y, dtype=int)
        counts = np.bincount(S)
        
        # p_i = Proporsi sampel per kelas
        p_i = counts / len(S)
        # Entropy(S) = sum( -p_i * log2(p_i) )
        return float(np.sum([-p * np.log2(p) for p in p_i if p > 0]))

    def gain_ratio(self, parent, subsets):
        # S = Total baris data induk
        S = len(parent)
        if S == 0: 
            return 0
        
        Entropy_S = self.entropy(parent)
        if Entropy_S < 1e-9: 
            return 0
            
        # Variabel untuk menampung total Sigma (penjumlahan)
        sum_gain = 0
        sum_split_info = 0
        
        for sub in subsets:
            if len(sub) == 0: 
                continue
            # S_i = Jumlah data di subset anak cabang
            S_i = len(sub)
            
            # (INFO Gain: (|S_i| / |S|) * Entropy(S_i)
            sum_gain += (S_i / S) * self.entropy(sub)
            
            # (Split Info: (S_i / S) * log2(S_i / S)
            sum_split_info += (S_i / S) * np.log2(S_i / S)
            
        # Gain(S, A) = Entropy(S) - sum( (|S_i|/|S|) * Entropy(S_i) )
        Gain_S_A = Entropy_S - sum_gain
    
        Split_Info_S_A = -sum_split_info
        
        if Gain_S_A < 1e-9: 
            return 0 
        if Split_Info_S_A < 1e-9: 
            return 0 
        
        # Gain Ratio(S, A) = Gain(S, A) / Split Info(S, A)
        Gain_Ratio_S_A = Gain_S_A / Split_Info_S_A
        return Gain_Ratio_S_A

    def evaluate_categorical(self, X_col, y):
        unique_vals = np.unique(X_col)
        if len(unique_vals) < 2:
            return -1.0, None 
        subsets = [y[X_col == val] for val in unique_vals]
        for sub in subsets:
            if len(sub) < self.min_samples_leaf:
                return -1.0, None 
        g = self.gain_ratio(y, subsets)
        return g, {"type": "categorical", "values": unique_vals.tolist()}

    def evaluate_numeric(self, X_col, y):
        sorted_idx = np.argsort(X_col)
        X_col, y = X_col[sorted_idx], y[sorted_idx]
        best_g = -1.0
        best_info = None
        if len(y) < (2 * self.min_samples_leaf):
            return -1.0, None
            
        for i in range(1, len(X_col)):
            if X_col[i] == X_col[i-1]: 
                continue
            thresh = (X_col[i] + X_col[i-1]) / 2
            left_mask = X_col <= thresh
            
            left, right = y[left_mask], y[~left_mask]
            if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                continue  
            
            g = self.gain_ratio(y, [left, right])  
            if g > best_g:
                best_g = g
                best_info = {"type": "numeric", "threshold": thresh}         
        return best_g, best_info

    def build_tree(self, X, y, available_features, parent_majority=0):
        y_arr = np.array(y)
        if len(y_arr) == 0:
            return int(parent_majority)        
        counts = Counter(y_arr).most_common(1)
        current_majority = int(counts[0][0]) if counts else int(parent_majority) 
        if self.entropy(y_arr) < 1e-9 or len(set(y_arr)) <= 1:
            return current_majority     
        if len(y_arr) < self.min_samples_leaf or len(available_features) == 0: 
            return current_majority     
        best_g = -1.0
        best_f = None
        best_info = None 
        
        for col_name in available_features:
            X_val = X[col_name].values
            if len(np.unique(X_val)) <= 1:
                continue   
            if col_name in CATEGORICAL_COLS:
                g, info = self.evaluate_categorical(X_val, y_arr)
            else:
                g, info = self.evaluate_numeric(X_val, y_arr)        
            if g > best_g:
                best_g, best_f, best_info = g, col_name, info
        if best_f is None or best_info is None:
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
                mask = (X[best_f] == val).values
                if not np.any(mask):
                    node["branches"][val] = current_majority
                else:
                    node["branches"][val] = self.build_tree(X.loc[mask], y.loc[mask], next_features, current_majority)
        else:
            thresh = best_info["threshold"]
            left_mask = (X[best_f] <= thresh).values
            right_mask = (X[best_f] > thresh).values
            
            if not np.any(left_mask) or not np.any(right_mask):
                return current_majority
                
            node["branches"]["left"] = self.build_tree(X.loc[left_mask], y.loc[left_mask], available_features, current_majority)
            node["branches"]["right"] = self.build_tree(X.loc[right_mask], y.loc[right_mask], available_features, current_majority)
            
        return node

    def fit(self, X, y):
        X_clean = X.reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
        all_features = list(X_clean.columns)
        init_majority = int(Counter(y_series).most_common(1)[0][0]) if len(y_series) > 0 else 0
        self.tree = self.build_tree(X_clean, y_series, all_features, init_majority)

    def predict_one(self, row, tree):
        if not isinstance(tree, dict): 
            return tree  
        feature_val = row[tree["feature"]]
        info = tree["split_info"]
        
        # jika data kategori tapi nilai tidak ada di cabang, kembalikan mayoritas node saat ini
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
            for val, branch in vals:
                print(f"{indent}IF {node['feature']} == {val}:")
                self.print_tree(branch, indent + "  | ")
        else:
            print(f"{indent}IF {node['feature']} <= {info['threshold']:.3f}:")
            self.print_tree(node["branches"]["left"], indent + "  | ")
            print(f"{indent}ELSE (> {info['threshold']:.3f}):")
            self.print_tree(node["branches"]["right"], indent + "  | ")
