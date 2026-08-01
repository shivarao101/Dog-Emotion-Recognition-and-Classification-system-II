import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
audio_path = "your_audio.wav"
y, sr = librosa.load(audio_path, sr=None)

# Compute Mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128,
    fmax=8000
)

# Convert to log scale (dB)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel_spec_db,
    sr=sr,
    x_axis='time',
    y_axis='mel',
    fmax=8000
)

plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
