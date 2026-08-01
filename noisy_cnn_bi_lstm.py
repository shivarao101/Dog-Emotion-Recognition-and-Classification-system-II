import numpy as np 
import os 
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Bidirectional,TimeDistributed,Flatten, LSTM, Dense, Dropout
)

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

                features = extract_spectrogram(signal, sr)
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


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 300, 1)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    # CNN output → (batch, 14, 73, 64)
    Reshape((73, 14 * 64)),   # ✅ FIXED

    Bidirectional(LSTM(64)),
    Dropout(0.5),

    Dense(4, activation='softmax')
])
model1 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(257, 300, 1)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    # CNN → BiLSTM bridge
    TimeDistributed(Flatten()),

    Bidirectional(LSTM(64)),
    Dropout(0.5),

    Dense(4, activation='softmax')
])
model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.summary()
history = model1.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)
