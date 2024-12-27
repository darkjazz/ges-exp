import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras import metrics, losses, ops, optimizers, random


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class GesVAE:
    def __init__(self):
        self.dir = "./data"

    def load_data(self):
        self.mels = joblib.load(os.path.join(self.dir, "mels128_vae.joblib"))
        self.mels = self.mels.reshape(*(*self.mels.shape, 1))

    def create_model(self):
        self.latent_dim = 2

        inputs = Input(shape=self.mels.shape[1:], name='encoder_input')
        enc = Conv2D(32, kernel_size=3, activation='relu', strides=2, padding='same')(inputs)
        enc = Conv2D(64, kernel_size=3, activation='relu', strides=2, padding='same')(enc)
        enc = Flatten()(enc)
        enc = Dense(256, activation='relu')(enc)
        z_mean = Dense(self.latent_dim, name='z_mean')(enc)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(enc)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        dec = Dense(32 * int(self.mels.shape[2] / 4), activation='relu')(latent_inputs)
        dec = Reshape((32, int(self.mels.shape[2] / 4), 1))(dec)
        dec = Conv2DTranspose(64, kernel_size=3, activation='relu', strides=2, padding='same')(dec)
        dec = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(dec)
        dec = Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(dec)

        self.decoder = Model(latent_inputs, dec, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])

        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))

    def fit(self, batch_size, epochs):
        self.history = self.vae.fit(
            self.mels,
            batch_size=batch_size,
            epochs=epochs
        )

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction = tf.image.resize(reconstruction, size=(128, 897))
            reconstruction_loss = ops.mean(
                ops.sum(
                    losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_weight = 0.1  # Lower the KL weight
            kl_loss = kl_weight * (-0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            ))            # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



# import numpy as np

# # Sample from the latent space
# def sample_latent_space(latent_dim, num_samples=1):
#     z_samples = np.random.normal(size=(num_samples, latent_dim))
#     return z_samples

# # Generate new Mel spectrograms
# latent_samples = sample_latent_space(latent_dim, num_samples=10)
# generated_spectrograms = decoder.predict(latent_samples)

# print("Generated Mel Spectrograms Shape:", generated_spectrograms.shape)

# import matplotlib.pyplot as plt

# def plot_spectrogram(spectrogram, title="Generated Spectrogram"):
#     plt.figure(figsize=(10, 4))
#     plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
#     plt.colorbar()
#     plt.title(title)
#     plt.xlabel('Time')
#     plt.ylabel('Frequency')
#     plt.show()

# # Plot the first generated spectrogram
# plot_spectrogram(generated_spectrograms[0].squeeze())

# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def mel_to_audio(mel_spectrogram, sr=22050, n_fft=2048, hop_length=512):
#     # Invert the mel spectrogram to a linear spectrogram
#     mel_spectrogram = np.squeeze(mel_spectrogram)  # Remove single-dimensional entries
#     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram_db, sr=sr, n_fft=n_fft)

#     # Convert the linear spectrogram to an audio signal using Griffin-Lim
#     audio_signal = librosa.griffinlim(linear_spectrogram, hop_length=hop_length)
#     return audio_signal

# # Example usage
# generated_spectrogram = generated_spectrograms[0]  # Use one of the generated spectrograms
# audio = mel_to_audio(generated_spectrogram)

# # Save audio to file
# librosa.output.write_wav('generated_audio.wav', audio, sr=22050)

# # Plot the audio signal
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(audio, sr=22050)
# plt.title("Generated Audio Waveform")
# plt.show()
