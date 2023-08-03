import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask

from SDAfunc import SDA_view, signal, sr

# 创建梅尔频谱图
M = librosa.feature.melspectrogram(y=signal, sr=sr, power=1)

# 创建通道混洗增强
augment = SpecCompose([SpecChannelShuffle(p=1)])

# 将一个通道的梅尔频谱图作为输入，并应用通道混洗增强
M1 = augment(np.array([M, M]))[0, :, :]
name1 = "C1"  # 为增强后的频谱图命名

# 将另一个通道的梅尔频谱图作为输入，并应用通道混洗增强
M2 = augment(np.array([M, M]))[1, :, :]
name2 = "C2"  # 为增强后的频谱图命名

# 显示原始频谱图和增强后的频谱图
SDA_view(M, M1, name1, M2, name2)
