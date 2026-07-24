from graphviz import Digraph

class TreeVisualizer:
    def __init__(self):
        self.dot = Digraph(
            "DecisionTree",
            format="png")
        self.dot.attr(
            rankdir="TB",
            fontsize="14")
        self.node_id = 0

        # Nama atribut agar lebih mudah dibaca
        self.feature_names = {
            "ipk1": "IPK Semester 1",
            "ipk2": "IPK Semester 2",
            "ipk3": "IPK Semester 3",
            "jumlah matakuliah d/e/f": "Jumlah MK D/E/F",
            "jumlah sks d/e/f": "Jumlah SKS D/E/F",
            "total sks semester 1-3": "Total SKS Semester 1-3",
            "kepulauan asal lahir": "Kepulauan Asal",
            "profil sekolah": "Profil Sekolah",
            "jalur pendaftaran": "Jalur Pendaftaran",
            "jurusan sekolah": "Jurusan Sekolah"
        }

        self.category_names = {
            "kepulauan asal lahir": {
                1: "Jawa",
                2: "Sumatera",
                3: "Bali & NTT",
                4: "Kalimantan",
                5: "Sulawesi",
                6: "Papua & Maluku",
                7: "Lain-lain"},

            "jurusan sekolah": {
                1: "SMA",
                2: "SMK",
                3: "Homeschooling",
                4: "Lain-lain"},
            "profil sekolah": {
                1: "Negeri",
                2: "Swasta",
                3: "Lain-lain"},
            "jalur pendaftaran": {
                1: "Raport",
                2: "Tes",
                3: "Lain-lain"}}
        
    def visualize(self, tree):
        self.node_id = 0
        self.dot = Digraph("DecisionTree", format="png")
        self.dot.attr(
            rankdir="TB",
            fontsize="14")
        self._build(tree)
        return self.dot

    def _new_id(self):
        self.node_id += 1
        return f"node{self.node_id}"

    def _build(self, node, parent=None, edge_label=""):

        node_name = self._new_id()
        if not isinstance(node, dict):
            if node == 1:
                self.dot.node(
                    node_name,
                    "RISIKO SISIP",
                    shape="ellipse",
                    style="filled",
                    fillcolor="#ffb3b3",
                    fontsize="12")
            else:
                self.dot.node(
                    node_name,
                    "TIDAK SISIP",
                    shape="ellipse",
                    style="filled",
                    fillcolor="#b8f5b8",
                    fontsize="12")
        else:
            feature_key = node["feature"]
            feature = self.feature_names.get(
                feature_key,feature_key)
            split = node["split_info"]
            samples = node["samples"]
            if split["type"] == "numeric":
                threshold = split["threshold"]
                label = (
                    f"{feature}\n\n"
                    f"≤ {threshold:.3f}\n\n"
                    f"Samples : {samples}")
            else:
                label = (
                    f"{feature}\n\n"
                    f"Samples : {samples}")
            self.dot.node(
                node_name,
                label,
                shape="box",
                style="rounded,filled",
                fillcolor="#D6EAF8",
                fontsize="12")
            if split["type"] == "numeric":
                threshold = split["threshold"]
                self._build(
                    node["branches"]["left"],
                    node_name,
                    f"≤ {threshold:.3f}")
                self._build(
                    node["branches"]["right"],node_name,f"> {threshold:.3f}")
            else:
                mapping = self.category_names.get(feature_key,{})
                for value, child in node["branches"].items():
                    label = mapping.get(value, str(value))
                    self._build(child,node_name,label)

        if parent is not None:
            self.dot.edge(
                parent,
                node_name,
                label=edge_label
            )