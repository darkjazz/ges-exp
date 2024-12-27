import couchdb
import joblib
import numpy as np
import os
import progressbar as pb
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    Dense,
    Masking,
    Reshape
)
from tensorflow.keras.optimizers import Adam


class GenomeLSTM:
    def __init__(self):
        self.data_dir = "/data"
        self.dbname = "ges_ld_00"
        self.db = couchdb.Server("http://admin:admin@127.0.0.1:5984/")[self.dbname]
        self.defname_view = self.db.view("application/defnamesByHeader")
        self.doc_view = self.db.view("application/docByDefName")

    def get_defnames(self, header):
        self.defnames = []
        for it in self.defname_view[header]:
            self.defnames.append(it.value)

    def get_row(self, key):
        genome = []
        params = []
        for it in self.doc_view[key]:
            doc = it.value
            genome = doc["ges:genome"]
            params = doc["ges:parameters"]["ges:literals"]
        return genome, params

    def get_genomes(self):
        self.genomes = {}
        self.params = {}
        prg = pb.ProgressBar(max_value=len(self.defnames))
        for _i, _key in enumerate(self.defnames):
            _genome, _params = self.get_row(_key)
            self.genomes[_key] = _genome
            self.params[_key] = _params
            prg.update(_i)
        prg.finish()

    # def build_targets(self):
    #     unique_ugens =

    def load_data(self, header):
        self.get_defnames(header)
        self.get_genomes()
        load_from = os.path.join(self.data_dir, "mels128.joblib")
        self.mels = joblib.load(load_from)
        self.mels = np.log1p(np.nan_to_num(self.mels, nan=0.0))
        print(f"loaded {len(self.mels)} mels")
        self.keys = joblib.load(load_from.replace(".joblib", "_keys.joblib"))
        print(f"loaded {len(self.keys)} keys")
        self.X, self.y = [], []
        for i, key in enumerate(self.keys):
            genome = self.genomes.get(key)
            if genome:
                self.X.append(self.mels[i])
                self.y.append(genome)
        print(f"extracted {len(self.X)} observations")
        ugens = list(set(value for values in self.y for value in values if len(value) > 1))
        ugens.sort()
        terminals = list(set(value for values in self.y for value in values if len(value) == 1))
        terminals.sort()
        target_map = { ugen: num for num, ugen in enumerate(ugens) }
        target_map.update({ term: num+50 for num, term in enumerate(terminals) })
        self.y = np.array([ [ float(target_map[it]) for it in genome ] for genome in self.y ])
        self.X = np.array(self.X)

    def build_model(self):
        input_shape = (self.mels.shape[1], self.mels.shape[2], 1)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        for units in [64, 128, 256]:
            self.model.add(Conv2D(units, kernel_size=(3, 3), activation="relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        # self.model.add(Dense(128, activation="relu"))
        # self.model.add(Reshape((128, -1)))
        # self.model.add(LSTM(64, return_sequences=True))
        # self.model.add(LSTM(32))
        self.model.add(Dense(self.y.shape[1], activation="softmax"))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        print(self.model.summary())
