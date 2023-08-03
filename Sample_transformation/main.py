from func.apply_audio_effects import apply_audio_effects
from func.audio_augmentation import audio_augmentation
from func.channel_shuffle_augmentation import channel_shuffle_augmentation
from func.SDAfunc import SDA_view
from func.spec_augmentation import spec_augmentation
from func.trim import trim
import numpy as np
from librosa import load, amplitude_to_db
from librosa.feature import melspectrogram
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise

if __name__ == '__main__':
    # 使用librosa库加载音频文件，并返回音频信号和采样率
    signal, sr = load("../lib/wav48/p225/p225_001.wav")

    # 生成梅尔频谱图
    S = melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)

    # 将梅尔频谱图转换为分贝形式
    M = amplitude_to_db(S, ref=np.max)

    signal = audio_augmentation(signal, sr, M)

    signal = trim(signal, sr)

    signal = channel_shuffle_augmentation(signal, sr)

    signal = spec_augmentation(signal, sr)

    signal = apply_audio_effects(signal, sr, M)

    # 生成梅尔频谱图
    M_augmented = melspectrogram(y=signal, sr=sr, power=1)

    # 显示原始频谱图和增强后的频谱图
    SDA_view(M, M_augmented, "Original", M_augmented, "Augmented")

    # 保存最终的音频
    sf.write('./output/final_audio.wav', signal, int(sr))
