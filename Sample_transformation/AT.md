# audioSampleTransformation

基于样本变换的样本增强模块

```
from audiomentations import Compose
```

[colab](https://colab.research.google.com/github/lucas-fpaiva/survey-audio-aug/blob/main/survey_aug.ipynb#scrollTo=7NLp7EUxrEh-)

## SDAfunc.py

代码的主要目标是音频处理和数据增强，以及将处理后的结果进行可视化。
下面我详细解释一下这些步骤的目的：

**音频加载和梅尔频谱图生成**

音频处理通常从加载音频文件开始，然后将音频信号转换为某种形式的表示，以便进一步处理。
这个代码中使用的表示方法是梅尔频谱图。

梅尔频谱图是一种将音频信号转换为二维图像的方法，
其中一维表示时间，另一维表示频率，而颜色表示特定频率在特定时间的强度。

这种表示方法可以帮助我们更好地理解音频信号的特性，并且可以利用图像处理和计算机视觉的技术来处理音频数据。

**数据增强**

数据增强是一种常用的技术，
可以通过创建训练数据的变体来增加数据集的大小和多样性，从而提高模型的性能和泛化能力。
在音频处理中，常见的数据增强技术包括添加噪声、改变音高或速度、频率遮罩和时间遮罩等。
这个代码中使用的是频率遮罩和频道混洗。

频率遮罩是一种模拟频率丢失的技术，可以提高模型对频率丢失的鲁棒性；
频道混洗则是一种模拟多声道音频环境的技术，可以提高模型的泛化能力。

**结果可视化**

结果可视化是数据分析和模型开发中的重要步骤。
通过可视化，我们可以更直观地理解数据的特性，检查数据预处理和增强的效果， 以及分析模型的性能。
在这个代码中，SDA_view函数生成了一个包含原始频谱图和两个增强频谱图的图表， 可以帮助我们理解数据增强的效果。

## SDASRM.py

这段代码的主要目标是展示如何使用`audiomentations`库进行音频数据的频谱增强，包括频率遮罩和时间遮罩，然后使用`SDA_view`函数将原始的梅尔频谱图以及增强后的梅尔频谱图进行可视化。

![image-20230726131140690](C:\Users\79355\AppData\Roaming\Typora\typora-user-images\image-20230726131140690.png)

黑色条块状的缺失正是数据增强操作的结果。这是由SpecFrequencyMask和SpecTimeMask这两种增强方法产生的。

**SpecFrequencyMask：**这种方法在频域（即梅尔频谱图的垂直轴）上选择一段范围，然后将这段范围内的所有值设为最小值（在可视化的时候，这部分会显示为黑色）。这相当于模拟了某个频段的音频信号丢失的情况。

**SpecTimeMask：**这种方法在时间域（即梅尔频谱图的水平轴）上选择一段范围，然后将这段范围内的所有值设为最小值（在可视化的时候，这部分会显示为黑色）。这相当于模拟了某个时间段的音频信号丢失的情况。

这两种方法都是频谱图数据增强中常用的手段，被称为频谱遮罩（Spectral Masking）。这些方法可以帮助模型更好地应对实际环境中可能出现的各种情况，比如信号丢失、设备故障等等，从而提高模型的鲁棒性。

## SDASCS.py

在这段代码中，我们使用`SpecChannelShuffle`函数来模拟音频信号在多个通道之间的混洗，然后将原始的梅尔频谱图以及混洗后的梅尔频谱图进行可视化。

## ST.py

在这段代码中，我们首先导入需要的模块和音频信号，然后使用 `librosa.effects.trim` 函数进行修剪。我们使用了两种不同的 `top_db` 参数值：默认的60dB和10dB。然后，我们生成原始的梅尔频谱图以及修剪后的梅尔频谱图，最后将原始的梅尔频谱图和修剪后的梅尔频谱图进行可视化。

## SP.py

该代码进行了以下三种音频处理：

1. **音高移位**：这是通过 `librosa.effects.pitch_shift` 函数实现的。它会改变音频信号的音高，但不会改变其速度。在这段代码中，音高被降低了6个半音。
2. **时间拉伸**：这是通过 `librosa.effects.time_stretch` 函数实现的。它会改变音频信号的播放速度，但不会改变其音高。在这段代码中，音频播放速度被减慢到了原来的75%。
3. **增益调整**：这是通过 `audiomentations.Compose` 和 `audiomentations.Gain` 实现的。它会改变音频信号的音量。在这段代码中，音量被随机增益在-20dB到0dB之间。

## high.py

这段代码是在做音频增广（audio augmentation）和频谱图生成。音频增广是一种在音频处理和机器学习中常用的技术，它可以通过在原始音频上添加各种随机变化来生成新的音频样本，从而增加数据集的大小和多样性。

以下是这段代码的详细解释：

1. 导入所需的库和函数：NumPy（用于数值计算），librosa（用于音频处理），SDAfunc（你自定义的模块，包含音频信号、采样率、梅尔频谱图和显示函数），audiomentations（用于音频增广）。
2. 定义一个增广效果列表，包括低通滤波器、高通滤波器、极性反转、反向播放、归一化和双曲正切失真。
3. 创建一个 Compose 实例，该实例会将所有的增广效果结合在一起。
4. 对原始音频信号应用所有的增广效果，生成增广后的音频信号。这里的采样率被设置为44100赫兹。
5. 使用 librosa.stft 函数将增广后的音频信号转换为短时傅里叶变换（STFT）的频谱图。
6. 计算频谱图的绝对值（也就是频谱的幅度）。
7. 使用 librosa.feature.melspectrogram 函数将幅度频谱图转换为梅尔频谱图。这里的参数设置为64个梅尔滤波器，最大频率为采样率的一半。
8. 最后，使用 SDAfunc 模块中的 SDA_view 函数显示原始音频和增广后的音频的梅尔频谱图。

这是一个对音频数据进行增广和频谱图生成的常见流程，可以用于音频分析、音频分类、语音识别等音频处理的任务中。

## main.py

```python
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
    signal, sr = load("./content/data/wav48/p225/p225_001.wav")

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

```

这段代码的目标是对输入的音频文件进行一系列的音频增强和预处理操作，并生成增强后的梅尔频谱图。以下是对代码各部分的详细解释：

1. **导入所需的库和函数**：代码开始处，首先导入了一些需要的库和函数，包括音频处理库librosa，音频文件写入库soundfile，以及一些自定义的音频增强函数。
2. **加载音频文件**：使用`librosa.load`函数加载音频文件`"./content/data/wav48/p225/p225_001.wav"`，得到音频信号`signal`和采样率`sr`。
3. **生成梅尔频谱图**：使用`librosa.feature.melspectrogram`函数对音频信号生成梅尔频谱图。
4. **将梅尔频谱图转换为分贝形式**：使用`librosa.amplitude_to_db`函数将梅尔频谱图的振幅转换为分贝表示，这是一种更符合人类听觉感知的表示方式。
5. **音频增强和预处理**：然后，音频信号经过一系列的增强和预处理操作，包括：
   - `audio_augmentation`: 进行音频增强。
   - `trim`: 对音频进行裁剪，通常是移除开始和结束的静音部分。
   - `channel_shuffle_augmentation`: 进行通道混洗增强，这是一种常用的数据增强技术，可以改善模型的泛化能力。
   - `spec_augmentation`: 进行频谱增强。
   - `apply_audio_effects`: 应用音频效果。具体的效果可能包括混响、回声等。
6. **生成增强后的梅尔频谱图**：对经过增强和预处理的音频信号重新生成梅尔频谱图。
7. **显示原始和增强后的梅尔频谱图**：使用`SDA_view`函数显示原始和增强后的梅尔频谱图。
8. **保存增强后的音频**：最后，使用`soundfile.write`函数将增强后的音频信号保存为音频文件。

