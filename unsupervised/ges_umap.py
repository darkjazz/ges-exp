import os
import glob
import hdbscan
import joblib
import json
import librosa
import numpy as np
from pathlib import Path
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
    Conv2D,
    MaxPooling2D,
    LSTM
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

min_file_size = 1800000

class UmapHdbscan:

    def __init__(self, dump_data=False, data_dir="./"):
        self.dump_data = dump_data
        self.data_dir = data_dir

    def get_files_by_size(self, path):
        match = os.path.join(path, "*.wav")
        self.files = [ fn for fn in glob.glob(match) if os.path.getsize(fn) >= min_file_size ]

    def extract_mel_spectrograms(self, n_mels=128):
        mels = {}
        for file in self.files:
            try:
                y, sr = librosa.load(file, sr=None)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                name = Path(file).stem
                mels[name] = mel
                print(name)
            except Exception as e:
                print(e)
                continue
        min_shape = min(set([ m.shape[1] for m in mels.values() ]))
        self.mels = np.array([ a[:,:min_shape] for a in mels.values() ])
        self.keys = [ k for k in mels.keys()]
        if self.dump_data:
            joblib.dump(self.mels, os.path.join(self.data_dir, f"mels{n_mels}_vae.joblib"))
            joblib.dump(self.keys, os.path.join(self.data_dir, f"mels{n_mels}_vae_keys.joblib"))

    def prepare_data(self, load_from=None):
        if load_from:
            self.mels = joblib.load(load_from)
            self.keys = joblib.load(load_from.replace(".joblib", "_keys.joblib"))
        self.mels = np.log1p(np.nan_to_num(self.mels, nan=0.0))
        self.mels1d = self.mels.reshape(2910, -1)

    def cluster(self):
        self.clst_emb = umap.UMAP(n_neighbors=37, min_dist=0.0, n_components=2).fit_transform(self.mels1d)
        self.clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30)
        self.labels = self.clusterer.fit_predict(self.clst_emb)
        if self.dump_data:
            joblib.dump(self.clst_emb, os.path.join(self.data_dir, "1d_umap_emb.joblib"))
            joblib.dump(self.labels, os.path.join(self.data_dir, "1d_hdbscan_labels.joblib"))

class Autoencoder:

    def __init__(self, n_mels, n_steps):
        self.n_mels = n_mels
        self.n_steps = n_steps
        self.input_dim = n_mels * n_steps
        self.input_shape = (n_mels, n_steps)
        self.batch_size = 32

    def build(self):
        self.model = Sequential()
        self.model.add(Input(shape=self.input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(self.input_dim, activation='sigmoid'))
        self.model.add(Reshape(self.input_shape))
        optimiser = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimiser, loss='mse')

    def train(self, mels, epochs):
        mels = np.nan_to_num(mels, nan=0.0)
        mels = np.log1p(mels)
        self.model.fit(mels, mels, epochs=epochs, batch_size=self.batch_size, shuffle=True)
        self.encoder = Sequential(self.model.layers[:4])
        self.encoded = self.encoder.predict(mels)
        self.clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30)
        self.labels = self.clusterer.fit_predict(self.encoded)
        return self.labels

class UgenCRNN:
    def __init__(self):
        self.data_dir = "./data"

    def prepare_data(self, test_size=0.1):
        load_from = os.path.join(self.data_dir, "mels128.joblib")
        self.mels = joblib.load(load_from)
        self.mels = np.log1p(np.nan_to_num(self.mels, nan=0.0))
        print(f"loaded {len(self.mels)} mels")
        self.keys = joblib.load(load_from.replace(".joblib", "_keys.joblib"))
        print(f"loaded {len(self.keys)} keys")
        with open(os.path.join(self.data_dir, "ugen_tags.json")) as f:
            self.labels = json.load(f)
        print(f"loaded {len(self.labels)} labels")
        self.X, self.y = [], []
        for i, key in enumerate(self.keys):
            for label in self.labels[key]:
                self.X.append(self.mels[i])
                self.y.append(label.lower())
        print(f"extracted {len(self.X)} observations")
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(np.array(self.y))
        self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(np.array(self.X), self.y, test_size=test_size, random_state=17)
        self.X_tr = self.X_tr.reshape(*(*self.X_tr.shape, 1))
        self.X_ts = self.X_ts.reshape(*(*self.X_ts.shape, 1))
        self.y_tr = to_categorical(self.y_tr, num_classes=self.le.classes_.shape[0])
        self.y_ts = to_categorical(self.y_ts, num_classes=self.le.classes_.shape[0])

    def build_model(self):
        input_shape = (self.mels.shape[1], self.mels.shape[2], 1)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        for units in [64, 128, 256]:
            self.model.add(Conv2D(units, kernel_size=(3, 3), activation="relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Reshape((128, -1)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(LSTM(32))
        self.model.add(Dense(self.le.classes_.shape[0], activation="softmax"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(self.model.summary())

    def train(self, epochs, batch_size):
        self.model.fit(
            self.X_tr,
            self.y_tr,
            validation_data=(self.X_ts, self.y_ts),
            epochs=epochs,
            batch_size=batch_size
        )
