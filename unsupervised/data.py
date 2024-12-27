import couchdb
import glob
import json
import librosa
import numpy as np
import os
import progressbar as pb
import soundfile as sf


class Data:
    def __init__(self, datadir="./data"):
        self.data_dir = datadir
        self.couch = couchdb.Server("http://admin:admin@127.0.0.1:5984/")
        self.db = self.couch["ges_ld_00"]
        self.ftr_db = self.couch["ges_synth_features_wav"]
        self.view = self.db.view("application/genomeByDefname")

    def get_row(self, key):
        genome = []
        genes = 0
        defname = ""
        for it in self.view[key]:
            doc = it.value
            genome = doc["ges:genome"]
            genes = doc["ges:numgenes"]
            defname = doc["ges:defname"]
        return genome, genes, defname

    def get_root_ugen_names(self, keys):
        self.root_names = {}
        prg = pb.ProgressBar(max_value=len(keys))
        for _i, _key in enumerate(keys):
            _genome, _genes, _defname = self.get_row(_key)
            self.root_names[_key] = [c for i, c in enumerate(_genome) if i%int(len(_genome)/_genes)==0]
            prg.update(_i)
        prg.finish()
        with open(os.path.join(self.data_dir, "ugen_tags.json"), "w") as f:
            json.dump(self.root_names, f)
        return self.root_names

    def get_all_root_ugen_names(self):
        self.root_names = {}
        prg = pb.ProgressBar(max_value=len(self.db))
        for _i, _key in enumerate(self.db):
            _doc = self.db[_key]
            try:
                _genome = _doc["ges:genome"]
                _genes = _doc["ges:environment"]["ges:numgenes"]
                _defname = _doc["ges:defname"]
                self.root_names[_defname] = [c for i, c in enumerate(_genome) if i%int(len(_genome)/_genes)==0]
            except:
                pass
            prg.update(_i)
        prg.finish()
        with open(os.path.join(self.data_dir, "all_ugen_tags.json"), "w") as f:
            json.dump(self.root_names, f)

    def collect_features(self):
        self.features = {}
        prg = pb.ProgressBar(max_value=len(self.ftr_db))
        for i, id in enumerate(self.ftr_db):
            doc = self.ftr_db[id]
            self.features[id] = doc["mfcc"]
            # self.features[id].extend(doc["contrast"])
            prg.update(i)
        prg.finish()
        print(f"loaded {len(self.features)} mfccs")


class AudioFix:
    def __init__(self):
        self.basedir = "/Users/kurivari/snd/evolver"
        self.input_dir = os.path.join(self.basedir, "test")
        self.output_dir = os.path.join(self.basedir, "wav")

    def process_audio_file(self, input_path, output_path):
        # Load the AIFF audio file using librosa
        audio, sample_rate = librosa.load(input_path, sr=None)

        # Resample to 48000 Hz if necessary
        if sample_rate != 48000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=48000)
            sample_rate = 48000

        # Replace NaNs and Infs
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp values between -1.0 and 1.0
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to 24-bit format by scaling the audio (since librosa uses float32)
        audio_24bit = (audio * (2**23 - 1)).astype(np.int32)

        # Save the processed audio as 24-bit WAV in the new location
        sf.write(output_path, audio_24bit, sample_rate, subtype='PCM_24')
        print(f"Processed and saved: {output_path}")

    def process_audio_files_in_directory(self):
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Loop through each audio file in the input directory
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(".aiff"):  # Handle AIFF files
                input_path = os.path.join(self.input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + ".wav"
                output_path = os.path.join(self.output_dir, output_filename)

                try:
                    self.process_audio_file(input_path, output_path)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
