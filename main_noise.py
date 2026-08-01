import os
import random
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# ------------------------------
# User settings
# ------------------------------
dog_dataset_path = "./dog_voice"      # Folder with 'bark/' and 'howl/'
urbansound_path = "./Urbansound8k"     # Root folder of UrbanSound8K
output_path = "./dog_noisy_dataset"    # Where augmented audio will be saved
sr = 16000                             # Sampling rate for all audio
snr_levels = [0, 5, 10]            # SNR levels in dB

# US8K noise classes to use
noise_classes = ["engine_idling", "jackhammer", "drilling", "air_conditioner"]

# ------------------------------
# Load UrbanSound8K noise
# ------------------------------
metadata = pd.read_csv(os.path.join(urbansound_path, "Urbansound8k.csv"))

noise_audio = {cls: [] for cls in noise_classes}

for idx, row in metadata.iterrows():
    class_name = row["class"]
    if class_name in noise_classes:
        fold = row["fold"]
        file_name = row["slice_file_name"]
        file_path = os.path.join(urbansound_path, f"fold{fold}", file_name)
        #y, _ = librosa.load(file_path, sr=sr)
        #noise_audio[class_name].append(y)
        if os.path.isfile(file_path):
            noise_audio[class_name].append(file_path)
for cls in noise_classes:
    print(f"{cls}: {len(noise_audio[cls])} clips loaded")

# ------------------------------
# Helper functions
# ------------------------------
def get_random_noise_segment(noise_files, length):
    noise_path = random.choice(noise_files)   # MUST be a string
    noise, _ = librosa.load(noise_path, sr=16000)

    if np.mean(noise**2) < 1e-6:
        raise ValueError("Silent noise clip detected")

    if len(noise) >= length:
        start = random.randint(0, len(noise) - length)
        return noise[start:start + length]
    else:
        return np.pad(noise, (0, length - len(noise)))


def mix_with_snr(signal, noise, snr_db):
    eps = 1e-8  # small constant to avoid divide-by-zero
    
    P_s = np.mean(signal**2) + eps
    P_n = np.mean(noise**2) + eps
  
    alpha = np.sqrt(P_s / (P_n * 10**(snr_db / 10)))
    
    mixed = signal + alpha * noise
    print(
    "Signal power:", np.mean(signal**2),
    "Noise power:", np.mean(noise**2),
    "Output max:", np.max(np.abs(mixed)))
    # Normalize to [-1, 1] to avoid clipping or silence
    max_val = np.max(np.abs(mixed)) + eps
    mixed = mixed / max_val
    
    return mixed


# ------------------------------
# Process dog dataset
# ------------------------------
os.makedirs(output_path, exist_ok=True)

for emotion in os.listdir(dog_dataset_path):
    emotion_path = os.path.join(dog_dataset_path, emotion)
    if not os.path.isdir(emotion_path):
        continue

    output_emotion_path = os.path.join(output_path, emotion)
    os.makedirs(output_emotion_path, exist_ok=True)

    for file_name in os.listdir(emotion_path):
        if not file_name.endswith(".wav"):
            continue

        dog_file_path = os.path.join(emotion_path, file_name)
        dog_y, _ = librosa.load(dog_file_path, sr=sr)
        length = len(dog_y)

        for snr_db in snr_levels:
            noise_class = random.choice(noise_classes)

            noise_segment = get_random_noise_segment(
                noise_audio[noise_class], length
            )
            noisy_dog = mix_with_snr(dog_y, noise_segment, snr_db)

            # ✅ SNR subfolder
            snr_folder = f"snr_{snr_db}"
            output_snr_path = os.path.join(output_emotion_path, snr_folder)
            os.makedirs(output_snr_path, exist_ok=True)

            out_file_name = (
                f"{os.path.splitext(file_name)[0]}_snr{snr_db}_{noise_class}.wav"
            )
            out_file_path = os.path.join(output_snr_path, out_file_name)

            sf.write(out_file_path, noisy_dog, sr)

        print(f"Processed {file_name} in {emotion} folder")

print("All files processed and saved to", output_path)

