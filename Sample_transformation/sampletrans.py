import os
from func.apply_audio_effects import apply_audio_effects
from func.audio_augmentation import audio_augmentation
from func.channel_shuffle_augmentation import channel_shuffle_augmentation
from func.SDAfunc import SDA_view
from func.spec_augmentation import spec_augmentation
from func.trim import trim
from db_func import funcs
import numpy as np
from librosa import load, amplitude_to_db
from librosa.feature import melspectrogram
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise


def sample_transformation():
    output_dir = 'Adversarial_example_recognition'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 从数据库中获取音频文件路径
    audio_files = funcs.get_audio_files_from_database()

    for audio_file in audio_files:
        # 使用librosa库加载音频文件，并返回音频信号和采样率
        signal, sr = load(audio_file)

        # 生成梅尔频谱图
        S = melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)

        # 将梅尔频谱图转换为分贝形式
        M = amplitude_to_db(S, ref=np.max)

        signal = audio_augmentation(signal, sr, M)

        # 修剪音频信号
        signal = trim(signal, sr)

        # 对信号进行通道混洗
        signal = channel_shuffle_augmentation(signal, sr)

        # 对信号进行频谱图增强
        signal = spec_augmentation(signal, sr)

        # 对信号应用音频效果(音高移位\时间拉伸\增益调整)
        signal = apply_audio_effects(signal, sr, M)

        # 生成梅尔频谱图
        M_augmented = melspectrogram(y=signal, sr=sr, power=1)

        # 显示原始频谱图和增强后的频谱图
        # SDA_view(M, M_augmented, "Original", M_augmented, "Augmented")

        # 保存最终的音频到指定的输出目录
        output_file = os.path.join(output_dir, os.path.basename(audio_file))
        sf.write(output_file, signal, int(sr))
