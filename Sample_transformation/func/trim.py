import librosa
from func.SDAfunc import SDA_view

def trim(signal, sr):
    # 使用默认的top_db（60dB）进行修剪
    signal_a1, _ = librosa.effects.trim(signal)
    name_a1 = "ST > 60db"  # 注意：librosa.effects.trim的默认top_db是60dB，不是40dB

    # 使用top_db=10进行修剪
    signal_a2, _ = librosa.effects.trim(signal, top_db=10)
    name_a2 = "ST > 10db"

    # 生成梅尔频谱图
    M = librosa.feature.melspectrogram(y=signal, sr=sr, power=1)
    M_a1 = librosa.feature.melspectrogram(y=signal_a1, sr=sr, power=1)
    M_a2 = librosa.feature.melspectrogram(y=signal_a2, sr=sr, power=1)

    # 显示原始频谱图和修剪后的频谱图
    # SDA_view(M, M_a1, name_a1, M_a2, name_a2)

    return signal