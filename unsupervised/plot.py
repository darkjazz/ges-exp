import matplotlib.pyplot as plt
import joblib
import librosa
import numpy as np

def plot_umap_embedding():
    emb = joblib.load("./data/umap_emb.joblib")
    plt.scatter(emb[:, 0], emb[:, 1], s=0.1)
    plt.show(block=False)

def plot_umap_hdbscan_clusters():
    emb = joblib.load("./data/1d_umap_emb_wav.joblib")
    lbl = joblib.load("./data/1d_hdbscan_labels_wav.joblib")
    clustered = (lbl >= 0)
    plt.scatter(emb[~clustered, 0],
                emb[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=0.1,
                alpha=0.5)
    plt.scatter(emb[clustered, 0],
                emb[clustered, 1],
                c=lbl[clustered],
                s=0.1,
                cmap='Spectral')
    plt.show(block=False)

def plot_spectrogram(mels):
    s_db = librosa.power_to_db(mels, ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(s_db, sr=44100, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('mel frequency spectrogram')
    plt.tight_layout()
    plt.show(block=False)

def identify_spikes(mels):
    flattened = np.concatenate([spec.flatten() for spec in mels])
    plt.figure(figsize=(10, 6))
    plt.hist(flattened, bins=100, alpha=0.75, color='blue')
    plt.title('Distribution of Mel Spectrogram Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show(block=False)

def plot_exemplars():
    ex = joblib.load("1d_mfcc_exemplars_wav.joblib")
    plt.figure(figsize=(10, 6))
    # plt.scatter(ex[:, 0], ex[:,1], c='gray', label='exemplars', alpha=0.5)
    for i, e in enumerate(ex):
        plt.scatter(e[:, 0], e[:, 1], s=47, label=str(i))
        centroid = np.mean(e, axis=0)  # Calculate centroid of the exemplars for annotation
        plt.text(centroid[0], centroid[1], f'{i}', fontsize=8, ha='center', va='center', color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.title("mfcc exemplars")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, fontsize='small', markerscale=0.7)
    plt.show(block=False)
