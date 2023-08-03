import numpy as np
import librosa
from SDAfunc import signal, sr, M, SDA_view
from audiomentations import Compose, LowPassFilter, HighPassFilter, PolarityInversion, Reverse, Normalize, TanhDistortion

# 定义增广效果列表
augmentations = [
    LowPassFilter(min_cutoff_freq=0, max_cutoff_freq=1000, p=1),
    HighPassFilter(min_cutoff_freq=5000, max_cutoff_freq=8000, p=1),
    PolarityInversion(p=1),
    Reverse(p=1),
    Normalize(p=1),
    TanhDistortion(min_distortion=0.01, max_distortion=0.5, p=1),
    TanhDistortion(min_distortion=0.5, max_distortion=1, p=1),
]

# 创建一个 Compose 实例，包含所有的增广效果
augmenter = Compose(augmentations)

# 应用所有的增广效果
augmented_signal = augmenter(samples=signal, sample_rate=44100)  # Increase sampling rate to 44100

# 使用 librosa 将音频信号转换为频谱图
D = librosa.stft(augmented_signal, n_fft=400, hop_length=160, win_length=400, center=False)
spectrogram = np.abs(D)

# 生成增广后的音频的梅尔频谱图
M_augmented = librosa.feature.melspectrogram(S=spectrogram, sr=44100, n_mels=64, fmax=44100/2)  # Reduce n_mels to 64 and set fmax to half of the sampling rate

# 显示原始音频和增广后的音频的梅尔频谱图
SDA_view(M, M_augmented, "Original", M_augmented, "Augmented")