from audiomentations import SpecCompose, SpecFrequencyMask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
from func.SDAfunc import SDA_view

def spec_augmentation(signal, sr):
    # 创建一个2行3列的网格布局
    gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])

    # 使用librosa库生成音频信号的梅尔频谱图
    M = librosa.feature.melspectrogram(y=signal, sr=sr, power=1)

    # 使用audiomentations库创建一个频率遮罩增强
    augment = SpecCompose([SpecFrequencyMask(p=1)])

    # 对梅尔频谱图应用频率遮罩增强
    M1 = augment(M)
    name1 = 'Frequency'  # 为增强后的频谱图命名

    # 对梅尔频谱图的转置应用频率遮罩增强，模拟时间遮罩效果
    M2 = augment(M.transpose()).transpose()
    name2 = 'Time'  # 为增强后的频谱图命名

    # 使用SDA_view函数显示原始频谱图和增强后的频谱图
    # SDA_view(M, M1, name1, M2, name2)

    return signal