import numpy as np 
import os 
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Bidirectional, LSTM, Dense, Dropout,
    TimeDistributed, Flatten, Softmax, Multiply, Lambda
)
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Model

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
#spectrogram
inputs = Input(shape=(64, 300, 1))  # freq can vary

# CNN
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# CNN → RNN bridge (NO RESHAPE)
x = TimeDistributed(Flatten())(x)

# BiLSTM
x = Bidirectional(LSTM(64, return_sequences=True))(x)

# Attention
attn = Dense(1, activation='tanh')(x)
attn = Softmax(axis=1)(attn)
x = Multiply()([x, attn])
x = GlobalAveragePooling1D()(x)

# Classifier
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)
y_pred_prob = model.predict(X_test)

# Convert probabilities → class labels
y_pred = np.argmax(y_pred_prob, axis=1)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
from sklearn.metrics import classification_report

report=classification_report(
    y_test,
    y_pred,
    target_names=['Dog-bark','Dog-growl', 'Dog-grunt', 'Dog-whining'],
    output_dict=True
)
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_report = pd.DataFrame(report).transpose()
df_report = df_report.drop(index='accuracy')
df_plot = df_report[['precision', 'recall', 'f1-score']]

plt.figure(figsize=(8, 5))
sns.heatmap(
    df_plot,   # remove accuracy row & support column
    annot=True,
    annot_kws={"weight": "bold", "size":12},
    cmap='YlGnBu',
    fmt=".2f"
)

plt.title("Classification Report",fontsize=12, fontweight='bold')
plt.ylabel("Classes",fontsize=12, fontweight='bold')
plt.xlabel("Metrics",fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("classreport.png", dpi=600)
plt.show()


plt.figure(figsize=(8,5))
sns.heatmap(cm, annot=True, fmt='d',annot_kws={"weight": "bold", "size":12},
            xticklabels=['Dog-bark','Dog-growl','Dog-grunt','Dog-whining'],
            yticklabels=['Dog-bark','Dog-growl','Dog-grunt','Dog-whining'],
            cmap='Blues')
plt.xlabel("Predicted",fontsize=12, fontweight='bold')
plt.ylabel("True",fontsize=12, fontweight='bold')
plt.title("Confusion Matrix",fontsize=12, fontweight='bold')
plt.savefig("confmat.png", dpi=600)
plt.show()

cm = confusion_matrix(y_test, y_pred)

# Class-wise accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Create DataFrame
class_labels = ['Dog-bark','Dog-growl','Dog-grunt','Dog-whining']
df_acc = pd.DataFrame({
    'Class': class_labels,
    'Accuracy': class_accuracy
})
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df_acc,
    x='Class',
    y='Accuracy'
)

plt.ylim(0, 1)
plt.title("Class-wise Accuracy",fontsize=12, fontweight='bold')
plt.ylabel("Accuracy",fontsize=12, fontweight='bold')
plt.xlabel("Class",fontsize=12, fontweight='bold')

# Add value labels
for index, value in enumerate(class_accuracy):
    plt.text(index, value + 0.02, f"{value:.2f}", ha='center')

plt.tight_layout()
plt.savefig("class_accuracy.png", dpi=600)
plt.show()
y_pred_prob = model.predict(X_test)
from sklearn.preprocessing import label_binarize

n_classes = 4
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
sns.set_style("whitegrid")
plt.figure(figsize=(7, 6))

class_names = ['Dog-bark','Dog-growl','Dog-grunt','Dog-whining']

for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        linewidth=3,
        label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
    )

# Diagonal reference line
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate",fontsize=12, fontweight='bold')
plt.ylabel("True Positive Rate",fontsize=12, fontweight='bold')
plt.title("Multi-class ROC Curve (One-vs-Rest)",fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig("roc_curve_multiclass.png", dpi=600)
plt.show()