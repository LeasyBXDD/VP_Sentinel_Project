import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

# 使用librosa库加载音频文件，并返回音频信号和采样率
# signal, sr = librosa.load("../content/data/wav48/p225/p225_001.wav")

# 生成梅尔频谱图
# S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)

# 将梅尔频谱图转换为分贝形式
# M = librosa.power_to_db(S, ref=np.max)

# 定义SDA_view函数，用于显示原始频谱图和两个增强频谱图
def SDA_view(M, M1, name1, M2, name2):
    # 创建一个新的图形窗口
    fig = plt.figure(figsize=(10, 3))  # ,dpi=300

    # 创建一个3列的网格布局
    gs = gridspec.GridSpec(nrows=1, ncols=3)

    # 在第一列中显示原始频谱图
    ax0 = fig.add_subplot(gs[0, 0])
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M, ref=np.max), y_axis="log", x_axis="time", ax=ax0
    )
    ax0.set_title("Original", fontsize=16)  # 设置标题
    ax0.axis("off")  # 关闭坐标轴显示

    # 在第二列中显示第一个增强频谱图
    ax1 = fig.add_subplot(gs[0, 1])
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M1, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax1,
    )
    ax1.set_title(name1, fontsize=16)  # 使用name1变量设置标题
    ax1.axis("off")  # 关闭坐标轴显示

    # 在第三列中显示第二个增强频谱图
    ax2 = fig.add_subplot(gs[0, 2])
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M2, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax2,
    )
    ax2.set_title(name2, fontsize=16)  # 使用name2变量设置标题
    ax2.axis("off")  # 关闭坐标轴显示

    # 显示图形
    plt.show()