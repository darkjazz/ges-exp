import joblib
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import umap
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from data import Data

class XgbClassifier:

    def __init__(self):
        self.data_dir = "./data"
        self.data = Data()

    def prepare_data(self, test_size=0.1):
        exclude = ["fmgrain", "lfdnoise1", "quadl", "softclip", "tanh", "standardl", "distort"]
        load_from = os.path.join(self.data_dir, "mels128.joblib")
        self.mels = joblib.load(load_from)
        # self.mels = np.log1p(np.nan_to_num(self.mels, nan=0.0))
        self.mels1d = self.mels.reshape(self.mels.shape[0], -1)
        print(f"loaded {len(self.mels)} mels")
        # self.emb = umap.UMAP(n_neighbors=37, min_dist=0.0, n_components=128).fit_transform(self.mels1d)
        self.emb = joblib.load(os.path.join(self.data_dir, "mels128_umap_embeddings.joblib"))
        print(f"umapped {self.emb.shape} embeddings")
        self.keys = joblib.load(load_from.replace(".joblib", "_keys.joblib"))
        print(f"loaded {len(self.keys)} keys")
        with open(os.path.join(self.data_dir, "all_ugen_tags.json")) as f:
            self.labels = json.load(f)
        print(f"loaded {len(self.labels)} labels")
        self.X, self.y = [], []
        for i, key in enumerate(self.keys):
            for label in self.labels.get(key, []):
                if label.lower() not in exclude:
                    self.X.append(self.emb[i])
                    self.y.append(label.lower())
        print(f"extracted {len(self.X)} observations")
        self.unique = list(set(self.y))
        self.unique.sort()
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(np.array(self.y))
        self.X = np.array(self.X)
        # self.ohe = OneHotEncoder(sparse_output=False)
        # self.y = self.ohe.fit_transform(np.array(self.y).reshape(-1, 1))
        # self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(np.array(self.X), self.y, test_size=test_size, random_state=17)

    def prepare_1d_data(self, features_file="mfcc.json", labels_file="all_ugen_tags.json", test_size=0.1):
        exclude = ["fmgrain", "lfdnoise1", "quadl", "softclip", "tanh", "standardl", "distort"]
        load_from = os.path.join(self.data_dir, features_file)
        if not os.path.exists(load_from):
            self.data.collect_features()
            with open(load_from, "w") as f:
                json.dump(self.data.features, f)
            print(f"wrote {len(self.data.features)} mfccs to {load_from}")
        else:
            with open(load_from) as f:
                self.data.features = json.load(f)
        with open(os.path.join(self.data_dir, labels_file)) as f:
            self.labels = json.load(f)
        self.X, self.y = [], []
        for i, key in enumerate(self.data.features):
            if key in self.labels:
                for label in self.labels[key]:
                    if label.lower() not in exclude:
                        self.X.append(self.data.features[key])
                        self.y.append(label.lower())
        print(f"extracted {len(self.X)} observations")
        self.unique = list(set(self.y))
        self.unique.sort()
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(np.array(self.y))
        self.X = np.array(self.X)
        # self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(np.array(self.X), np.array(self.y), test_size=test_size, random_state=17)

    def prepare_bbc_data(self):
        with open(os.path.join(self.data_dir, "bbc_features.json")) as f:
            self.tracks = json.load(f)
        self.X, self.y = [], []
        for track in self.tracks.values():
            for style in track.get("tags", []):
                ftr = eval(track["mfcc_mean"])
                ftr.extend(np.array(eval(track["mfcc_cov"])).diagonal().tolist())
                self.X.append(ftr)
                self.y.append(style)
        print(f"extracted {len(self.X)} observations")
        self.unique = list(set(self.y))
        self.unique.sort()
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(np.array(self.y))
        self.X = np.array(self.X)

    def fit(self):
        self.model = XGBClassifier(objective='multi:softprob', num_class=len(self.unique), n_estimators=1000, max_depth=10)
        self.model.fit(self.X, self.y)
        self.prdct = self.model.predict(self.X)
        self.acc = accuracy_score(self.y, self.prdct)
        print("Accuracy:", self.acc)
        print("\nClassification Report:")
        print(classification_report(self.y, self.prdct, target_names=np.array(self.unique)))
