import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# 设置音频后端
torchaudio.set_audio_backend("soundfile")


class Opt:
    def __init__(self):
        self.channels = 1
        self.img_size = 32


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


discriminator = Discriminator()
discriminator.load_state_dict(torch.load('../lib/discriminator.pth', map_location=torch.device('cpu')))
discriminator.eval()  # 设置为评估模式

sample_rate = 48000  # 你需要替换这个为你的实际采样率
sample_length = sample_rate  # 1秒的样本长度
desired_time_steps = 128

# 预测测试集中的样本
with torch.no_grad():  # 不需要计算梯度

    audio_files = ['../lib/wav48/p225/p225_002.wav',
                   '../lib/wav48/p225/p255_002.wav',
                   '../lib/wav48/p225/p255_002.wav']

    for audio_file in audio_files:

        waveform, _ = torchaudio.load(audio_file)
        samples = list(waveform.unfold(1, sample_length, sample_length).split(sample_length, dim=1))

        if waveform.size(1) % sample_length != 0:
            last_segment = waveform[:, waveform.size(1) // sample_length * sample_length:]
            last_segment = pad_waveform(last_segment, sample_length)
            samples.append(last_segment)

        mel_spectrograms = []

        for sample in samples:
            if sample.nelement() > 0:
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(sample)
                mel_spectrogram = mel_spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                # Interpolate only the height (n_mels) and width (time steps)
                mel_spectrogram = torch.nn.functional.interpolate(mel_spectrogram, size=(128, desired_time_steps))
                mel_spectrograms.append(mel_spectrogram.squeeze(0))  # remove batch dimension

        # Later when feeding to the discriminator
        inputs = torch.stack(mel_spectrograms)  # Now inputs already have channel dimension

        if len(mel_spectrograms) == 0:
            print("No mel spectrograms were generated from the test data. Skipping testing.")
        else:
            outputs = discriminator(inputs)
            outputs = torch.round(outputs)  # round to 0 or 1
            print('Predicted for file {}: '.format(audio_file),
                  ' '.join('%5s' % outputs[j] for j in range(len(outputs))))