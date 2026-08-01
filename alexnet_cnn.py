import numpy as np
from scipy.io import wavfile
from scipy import signal
def load_audio(path, target_sr=16000):
    sr, audio = wavfile.read(path)

    # Convert to float
    audio = audio.astype(np.float32)

    # Stereo → Mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr)
        sr = target_sr

    # Normalize
    audio = (audio - np.mean(audio)) / np.std(audio)
    #audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio, sr
def extract_spectrogram(audio, sr):
    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window='hann',
        nperseg=512,
        noverlap=256,
        nfft=512,
        scaling='spectrum',
        mode='magnitude'
    )

    # Log compression
    log_Sxx = np.log(Sxx + 1e-10)

    return log_Sxx
import tensorflow as tf

def extract_logmel(audio, sr=16000, n_mels=64):
    stft = tf.signal.stft(
        audio,
        frame_length=512,
        frame_step=256,
        fft_length=512
    )

    spectrogram = tf.abs(stft)

    mel_weight = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sr,
        lower_edge_hertz=50,
        upper_edge_hertz=sr // 2
    )

    mel_spec = tf.tensordot(spectrogram, mel_weight, 1)
    mel_spec.set_shape(spectrogram.shape[:-1].concatenate(
        mel_weight.shape[-1:])
    )

    log_mel = tf.math.log(mel_spec + 1e-6)
    return log_mel.numpy()
import random

def spec_augment(spec,
                 freq_mask_param=10,
                 time_mask_param=20,
                 num_freq_masks=2,
                 num_time_masks=2):

    spec = tf.identity(spec)

    num_mel, num_frames = spec.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, num_mel - f))
        spec = tf.concat(
            [spec[:f0, :],
             tf.zeros((f, num_frames)),
             spec[f0 + f:, :]],
            axis=0
        )

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, num_frames - t))
        spec = tf.concat(
            [spec[:, :t0],
             tf.zeros((num_mel, t)),
             spec[:, t0 + t:]],
            axis=1
        )

    return spec

def pad_or_crop(spec, target_shape=(256, 256)):
    freq, time = spec.shape
    tf, tt = target_shape

    padded = np.zeros(target_shape)

    f_min = min(freq, tf)
    t_min = min(time, tt)

    padded[:f_min, :t_min] = spec[:f_min, :t_min]
    return padded
import os

def prepare_dataset(file_paths, labels):
    X = []
    y = []

    for path, label in zip(file_paths, labels):
        audio, sr = load_audio(path)
        spec = extract_spectrogram(audio, sr)
        spec = pad_or_crop(spec)

        X.append(spec)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    return X, y
def prepare_dataset1(file_paths, labels, training=True):
    X, y = [], []

    for path, label in zip(file_paths, labels):
        audio, sr = load_audio(path)

        logmel = extract_logmel(audio, sr)

        #if training:
            #logmel = spec_augment(logmel)

        # Fixed size (important)
        logmel = pad_or_crop(logmel, target_shape=(64, 256))

        X.append(logmel)
        y.append(label)

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y

import tensorflow as tf
from tensorflow.keras import models, layers

model = models.Sequential([

    layers.Input(shape=(64,256,1)),

    layers.Conv2D(64,(11,11),strides=4,activation='relu'),
    layers.MaxPooling2D((3,3),strides=2,padding='same'),

    layers.Conv2D(128,(5,5),padding='same',activation='relu'),
    layers.MaxPooling2D((3,3),strides=2,padding='same'),

    layers.Conv2D(192,(3,3),padding='same',activation='relu'),
    layers.Conv2D(192,(3,3),padding='same',activation='relu'),

    layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    layers.MaxPooling2D((3,3),strides=2,padding='same'),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
import os

def load_dataset_from_folder(root_dir):
    file_paths = []
    labels = []

    class_names = sorted(os.listdir(root_dir))  
    # ['bark', 'howl']

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)

        for file in os.listdir(class_dir):
            if file.lower().endswith(".wav"):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(label)

    return file_paths, labels, class_names
from sklearn.model_selection import train_test_split

root_dir = "dog_voice"

file_paths, labels, class_names = load_dataset_from_folder(root_dir)

X_train_files, X_test_files, y_train, y_test = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # VERY important for emotion datasets
)
X_train, y_train = prepare_dataset1(X_train_files, y_train)
X_test,  y_test  = prepare_dataset1(X_test_files,  y_test)
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)
for i, name in enumerate(class_names):
    print(f"{i} → {name}")
