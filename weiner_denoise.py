import numpy as np
import librosa

def wiener_filter(
    noisy,
    sr=16000,
    n_fft=512,
    hop_length=256,
    noise_frames=10,
    eps=1e-10
):
    """
    Frequency-domain Wiener filter
    """

    # STFT of noisy signal
    X = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag_X = np.abs(X)
    phase_X = np.angle(X)

    # Estimate noise power spectrum P_n(k)
    P_n = np.mean(mag_X[:, :noise_frames]**2, axis=1, keepdims=True)

    # Estimate signal power spectrum P_s(k)
    P_x = mag_X**2
    P_s = np.maximum(P_x - P_n, eps)

    # Wiener gain
    G = P_s / (P_s + P_n)

    # Apply Wiener filter
    S_hat = G * mag_X * np.exp(1j * phase_X)

    # Inverse STFT
    enhanced = librosa.istft(S_hat, hop_length=hop_length)

    return enhanced
import os
import soundfile as sf

input_root = "./dog_noisy_dataset"
output_root = "./dog_wiener_denoised"
os.makedirs(output_root, exist_ok=True)

for emotion in ["dog_bark_new","dog_growl_new", "dog_grunt_new", "dog_whinning_new"]:
    for snr in ["snr_0", "snr_5", "snr_10"]:
        in_dir = f"{input_root}/{emotion}/{snr}"
        out_dir = f"{output_root}/{emotion}/{snr}"
        os.makedirs(out_dir, exist_ok=True)

        for file in os.listdir(in_dir):
            if not file.endswith(".wav"):
                continue

            noisy, sr = librosa.load(os.path.join(in_dir, file), sr=16000)
            enhanced = wiener_filter(noisy)

            sf.write(os.path.join(out_dir, file), enhanced, sr)
