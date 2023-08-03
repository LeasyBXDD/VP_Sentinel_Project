import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# 设置音频后端
torchaudio.set_audio_backend("soundfile")


# 定义GAN的判别器
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 第一层卷积层
#         self.avgpool = nn.AdaptiveAvgPool2d((8, 10))  # 平均池化层，将输入张量调整到指定的大小
#         self.fc1 = nn.Linear(8 * 10 * 8, 500)  # 第一层全连接层
#         self.fc2 = nn.Linear(500, 1)  # 第二层全连接层，输出一个单一的判别分数
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # 增加一个维度，从(batch_size, height, width)变为(batch_size, 1, height, width)
#         x = F.relu(self.conv1(x))  # 通过第一层卷积并应用ReLU激活函数
#         x = self.avgpool(x)  # 对卷积的输出进行平均池化
#         x = x.view(x.size(0), -1)  # 对张量进行reshape，方便输入全连接层
#         x = F.relu(self.fc1(x))  # 通过第一层全连接层并应用ReLU激活函数
#         x = torch.sigmoid(self.fc2(x))  # 通过第二层全连接层并应用Sigmoid激活函数，将输出限制在0到1之间
#         return x

class Opt:
    def __init__(self):
        self.channels = 1
        self.img_size = 32
        # add other parameters as needed

opt = Opt()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# 定义填充函数，如果音频片段小于期望长度，使用零进行填充
def pad_waveform(waveform, desired_length):
    if waveform.size(1) < desired_length:
        num_missing_samples = desired_length - waveform.size(1)
        last_dim_padding = (0, num_missing_samples)
        waveform = F.pad(waveform, last_dim_padding)
    return waveform


# 加载模型 Deep Convolutional GAN(DCGAN)
# https://github.com/eriklindernoren/PyTorch-GAN.git
# generator.pth
# discriminator.pth
discriminator = Discriminator()
# discriminator.load_state_dict(torch.load('../lib/discriminator.pth'))
discriminator.load_state_dict(torch.load('../lib/discriminator.pth', map_location=torch.device('cpu')))
discriminator.eval()  # 设置为评估模式

# 假设我们知道每个样本的长度是1秒
sample_rate = 16000  # 你需要替换这个为你的实际采样率
sample_length = sample_rate  # 1秒的样本长度

# 预测测试集中的样本
with torch.no_grad():  # 不需要计算梯度

    # 音频文件路径列表
    # audio_files = ['./audio1.wav', './audio2.wav', './audio3.wav']
    audio_files = ['../lib/wav48/p225/p255_001.wav', '../lib/wav48/p225/p255_002.wav', '../lib/wav48/p225/p255_003.wav']

    for audio_file in audio_files:

        # 加载并预处理测试数据
        waveform, _ = torchaudio.load(audio_file)

        # 分割波形
        samples = waveform.unfold(1, sample_length, sample_length).split(sample_length, dim=1)

        # 处理最后一个片段，如果最后一个音频片段的长度小于sample_length，使用零进行填充
        if waveform.size(1) % sample_length != 0:
            last_segment = waveform[:, waveform.size(1) // sample_length * sample_length:]
            last_segment = pad_waveform(last_segment, sample_length)
            samples.append(last_segment)

        # 对每个非空样本计算梅尔频谱
        mel_spectrograms = [torch.squeeze(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(sample)) for
                            sample in samples if sample.nelement() > 0]

        if len(mel_spectrograms) == 0:
            print("No mel spectrograms were generated from the test data. Skipping testing.")
        else:
            # 将音频样本直接输入到模型中
            inputs = torch.stack(mel_spectrograms)
            inputs = inputs.unsqueeze(1)  # 增加一个维度，从(batch_size, height, width)变为(batch_size, 1, height, width)

            outputs = discriminator(inputs)
            outputs = torch.round(outputs)  # round to 0 or 1
            print('Predicted for file {}: '.format(audio_file),
                  ' '.join('%5s' % outputs[j] for j in range(len(outputs))))
