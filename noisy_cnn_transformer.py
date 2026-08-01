import numpy as np 
import os 
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers, models
def extract_spectrogram(
    audio,
    sr,
    nperseg=512,
    noverlap=256,
    nfft=512,
    max_len=300   # 🔑 FIXED time dimension
):
    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling='spectrum',
        mode='magnitude'
    )

    # Log compression
    log_Sxx = np.log(Sxx + 1e-10)

    # 🔧 Fix time dimension
    if log_Sxx.shape[1] < max_len:
        pad_width = max_len - log_Sxx.shape[1]
        log_Sxx = np.pad(log_Sxx, ((0, 0), (0, pad_width)))
    else:
        log_Sxx = log_Sxx[:, :max_len]

    return log_Sxx

def extract_logmel(y, sr=16000, n_mels=64, max_len=300):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512
    )
    logmel = librosa.power_to_db(mel)

    # Fix time dimension
    if logmel.shape[1] < max_len:
        pad_width = max_len - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)))
    else:
        logmel = logmel[:, :max_len]

    return logmel

def load_dataset(dataset_path, sr=16000):
    X, y = [], []
    label_map = {"dog_bark_new": 0,"dog_growl_new": 1, "dog_grunt_new": 2, "dog_whinning_new": 3}

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for snr_folder in os.listdir(label_path):
            snr_path = os.path.join(label_path, snr_folder)
            if not os.path.isdir(snr_path):
                continue

            for file in os.listdir(snr_path):
                if not file.endswith(".wav"):
                    continue

                file_path = os.path.join(snr_path, file)
                signal, _ = librosa.load(file_path, sr=sr)

                features = extract_logmel(signal, sr)
                X.append(features)
                y.append(label_map[label])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y
X, y = load_dataset("./dog_wiener_denoised")

print("X shape:", X.shape)
print("y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
def transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1):
    # Self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Feed-forward network
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)

    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)
# Spectrogram
#input_shape = (257, 300, 1)
# Log-Mel
#input_shape = (64, 300, 1)
inputs = layers.Input(shape=(64, 300, 1))

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)

# CNN output: (None, H, W, 128)

# ✅ Correct reshape for Transformer
x = layers.Reshape((-1, 128))(x)   # (None, T, 128)

# Transformer encoder
x = transformer_encoder(
    x,
    head_size=64,
    num_heads=4,
    ff_dim=256
)

# Temporal pooling
x = layers.GlobalAveragePooling1D()(x)

# Classifier
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)
