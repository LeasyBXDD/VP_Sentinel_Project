import librosa
from audiomentations import Compose, Gain
from Sample_transformation.func.SDAfunc import SDA_view


# from main import M, signal, sr

def apply_audio_effects(signal, sr, M):
    # 应用音高移位
    signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=-6)
    # 应用时间拉伸
    signal = librosa.effects.time_stretch(signal, rate=0.75)
    # 应用增益调整
    augment = Compose([Gain(min_gain_in_db=-20, max_gain_in_db=0, p=1)])
    signal = augment(samples=signal, sample_rate=int(sr))

    # 应用音频处理步骤
    # signal_augmented = apply_audio_effects(signal, sr, M)

    # 生成梅尔频谱图
    # M_augmented = librosa.feature.melspectrogram(y=signal_augmented, sr=sr, power=1)

    # 显示原始频谱图和增强后的频谱图
    # SDA_view(M, M_augmented, "Original", M_augmented, "Augmented")

    return signal
