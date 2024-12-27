import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import argparse

# Define the Generator class for WaveGAN
class WaveGANGenerator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(WaveGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(z_dim, 256, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator class for WaveGAN
class WaveGANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(WaveGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=25, stride=4, padding=11),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=11),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, kernel_size=25, stride=4, padding=11),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), -1).mean(dim=1, keepdim=True)


class WaveGAN:
    def __init__(self, snd_dir, sample_rate, waveform_scalar, epochs, batch_size, learning_rate, model_dir):
        self.snd_dir = snd_dir
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.z_dim = 100  # Latent space dimension
        self.waveform_length = sample_rate * waveform_scalar  # Length of audio waveform
        self.dataset = self.load_wav_files()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.generator = WaveGANGenerator(z_dim=self.z_dim, output_dim=self.waveform_length)
        self.discriminator = WaveGANDiscriminator(input_dim=self.waveform_length)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.model_dir = model_dir

    def load_wav_files(self):
        """Loads WAV files from a directory and prepares them for training."""
        dataset = []
        for file_name in os.listdir(self.snd_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(self.snd_dir, file_name)
                waveform, sample_rate = torchaudio.load(file_path)

                # Use only the first channel for stereo files
                waveform = waveform[0:1, :]

                # Resample if needed
                if sample_rate != self.sample_rate:
                    resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                    waveform = resample_transform(waveform)

                if waveform.shape[1] >= self.waveform_length:
                    waveform = waveform[:, :self.waveform_length]
                else:
                    waveform = torch.nn.functional.pad(waveform, (0, self.waveform_length - waveform.shape[1]))
                dataset.append(waveform)
        return torch.stack(dataset)

    def train(self):
        for epoch in range(self.epochs):
            for real_data in self.dataloader:
                batch_size = real_data.size(0)

                # Train Discriminator
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Real data loss
                real_data = real_data.view(batch_size, 1, -1)  # Reshape to (batch_size, channels, length)
                outputs = self.discriminator(real_data)
                d_loss_real = self.criterion(outputs, real_labels)

                # Fake data loss
                z = torch.randn(batch_size, self.z_dim, 1)  # Latent vector reshaped for ConvTranspose1d
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(outputs, fake_labels)

                # Backpropagate total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                z = torch.randn(batch_size, self.z_dim, 1)
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data)
                g_loss = self.criterion(outputs, real_labels)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                print(f"Epoch [{epoch+1}/{self.epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    def generate_audio(self, num_samples, output_dir, duration=1, sample_rate=44100):
        """Generate audio samples from the trained generator and save them as .wav files."""
        os.makedirs(output_dir, exist_ok=True)
        self.generator.eval()  # Set the generator to evaluation mode

        waveform_length = int(sample_rate * duration)

        with torch.no_grad():  # No gradient calculation needed for generation
            for i in range(num_samples):
                z = torch.randn(1, self.z_dim, 1)  # Generate a random latent vector
                fake_waveform = self.generator(z).squeeze(0).cpu()  # Generate audio and remove batch dimension
                fake_waveform = fake_waveform.clamp(-1, 1)  # Clamp values to [-1, 1]

                # Adjust waveform to desired length
                fake_waveform = torch.nn.functional.interpolate(
                    fake_waveform.unsqueeze(0).unsqueeze(0),
                    size=waveform_length,
                    mode="linear",
                    align_corners=False
                ).squeeze()

                # Save the waveform as a .wav file
                file_path = os.path.join(output_dir, f"generated_sample_{i + 1}.wav")
                torchaudio.save(file_path, fake_waveform.unsqueeze(0), sample_rate)
                print(f"Saved: {file_path}")

    def save_model(self):
        """Save the generator and discriminator models to the specified directory."""
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(self.model_dir, "generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, "discriminator.pth"))
        print(f"Models saved to {self.model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a WaveGAN model.")
    parser.add_argument('--snd_dir', type=str, required=True, help="Path to the directory containing WAV files.")
    parser.add_argument('--sample_rate', type=int, default=44100, help="Sample rate of training data")
    parser.add_argument('--waveform_scalar', type=int, default=7, help="Multiplier for waveform length")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate.")
    parser.add_argument('--output_dir', type=str, default="generated_samples", help="Directory to save generated samples.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of audio samples to generate.")
    parser.add_argument('--duration', type=int, default=5, help="Duration of generated audio in seconds.")
    parser.add_argument('--model_dir', type=str, default="saved_models", help="Directory to save the trained models.")

    args = parser.parse_args()

    wavegan = WaveGAN(
        args.snd_dir,
        args.sample_rate,
        args.waveform_scalar,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.model_dir
    )

    # Train the model
    wavegan.train()

    # Save the trained models
    wavegan.save_model()

    # Generate and save samples after training
    wavegan.generate_audio(args.num_samples, args.output_dir, duration=args.duration, sample_rate=args.sample_rate)
