import os
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import librosa
import numpy as np
import soundfile as sf
from temp.my_gan_model import MyGANModel, Generator, Discriminator  # 导入自定义GAN模型
import torch.optim as optim
from collections import OrderedDict

# 根据环境决定后续操作在哪个设备上运行，优先选择GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义各种恢复方法的权重
weight_clever = 0.7
weight_second_prob = 0.3


# 使用第二大可能性恢复标签
def recover_label_with_second_probability(model, denoised_mel_spectrogram):
    probabilities = model(denoised_mel_spectrogram)  # 获取模型对去噪音频的预测概率
    sorted_probabilities, sorted_labels = torch.sort(probabilities, descending=True)  # 按概率值降序排列
    second_most_probable_label = sorted_labels[1]  # 获取第二大可能性的标签
    return second_most_probable_label


# 定义Carlini & Wagner (C&W) 方法
def CW_attack(model, input, target, num_steps=1000, learning_rate=0.01):
    perturbation = torch.zeros_like(input).cuda()
    perturbation.requires_grad = True
    optimizer = optim.Adam([perturbation], lr=learning_rate)
    for step in range(num_steps):
        optimizer.zero_grad()  # 清空梯度
        perturbed_input = input + perturbation
        output = model(perturbed_input)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        perturbation.data = project_to_l2_ball(perturbation.data)
    perturbed_input = input + perturbation
    return perturbed_input


# 将输入张量投影到 L2 ball上
def project_to_l2_ball(x, eps=1.0):
    norm = torch.norm(x)
    if norm > eps:
        x = x / norm * eps
    return x


# 使用C&W攻击来计算每个类别的距离
def compute_distance_with_cw_attack(model, input, num_classes):
    distances = []
    for target_class in range(num_classes):
        target = torch.tensor([target_class]).cuda()
        perturbed_input = CW_attack(model, input, target)
        distance = torch.norm(input - perturbed_input)
        distances.append(distance)
    return distances


# 使用C&W计算距离函数来恢复标签
def recover_label_with_cw_attack(model, denoised_mel_spectrogram, num_classes):
    distances = compute_distance_with_cw_attack(model, denoised_mel_spectrogram, num_classes)  # 计算C&W攻击的距离
    recovered_label = torch.argmin(distances)  # 选择距离最小的标签作为恢复的标签
    return recovered_label


# 计算输出和权重
def recovery_system(outputs, weights, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluation_values = torch.tensor(outputs).to(device) * torch.tensor(weights).to(device)  # 计算评估值
    max_evaluation_value = torch.max(evaluation_values)  # 获取最大的评估值

    # 如果最大的评估值小于阈值threshold，则拒绝样本，进行手动审核或丢弃
    if max_evaluation_value < threshold:
        return None
    else:
        # 找出评估值等于最大评估值的输出标签
        possible_labels = [output for output, evaluation_value in zip(outputs, evaluation_values) if
                           evaluation_value == max_evaluation_value]
        if not possible_labels:
            return None
        # 从可能的标签中随机选择一个作为真实标签
        true_label = np.random.choice(possible_labels)
        return true_label


# 将音频文件转换为模型需要的输入格式
def prepare_input(audio_file):
    y, sr = librosa.load(audio_file)  # 读取音频文件
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # 计算音频的Mel频谱图
    mel_spectrogram = torch.from_numpy(mel_spectrogram)  # 将Mel频谱图转换为张量
    mel_spectrogram = mel_spectrogram.unsqueeze(0)  # 添加batch size维度
    mel_spectrogram = mel_spectrogram.unsqueeze(0)  # 添加channel维度
    mel_spectrogram = torch.nn.functional.interpolate(mel_spectrogram, 100)  # 将Mel频谱图调整到需要的大小
    mel_spectrogram = (mel_spectrogram / 255.0) * 2 - 1  # 将Mel频谱图归一化到[-1, 1]区间
    return mel_spectrogram


# 加载模型的状态
def load_my_state_dict(model, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('l1.0.'):  # 如果键以'l1.0.'开头
            name = k.replace('l1.0.', 'l1.')  # 用'l1.'替换'l1.0.'
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)  # 加载新的状态字典


# 用GAN去噪并进行标签恢复
def denoise_with_GAN(GAN_model, audio_file, sr, C=0.5, output_dir=None):
    input = prepare_input(audio_file)  # 准备输入
    input = input.to(device)  # 将输入移至设备
    input_var = Variable(input, requires_grad=True)  # 创建Variable
    GAN_model.G.eval()  # 设置生成器为评估模式
    denoised = GAN_model.G(input_var)  # 使用生成器去噪
    num_classes = GAN_model.num_classes  # 获取类别数量
    GAN_model.D.eval()  # 设置鉴别器为评估模式
    # 使用C&W方法恢复标签
    recovered_label_cw = recover_label_with_cw_attack(GAN_model.D, denoised, num_classes)
    # 使用第二大概率恢复标签
    recovered_label_second_prob = recover_label_with_second_probability(GAN_model.D, denoised)
    # 使用恢复系统得到最终的恢复标签
    recovered_label = recovery_system([recovered_label_cw, recovered_label_second_prob],
                                      [weight_clever, weight_second_prob], C)
    if recovered_label is None:  # 如果没有恢复到标签，则打印消息并返回
        print("No label recovered")
        return
    else:  # 否则，打印恢复的标签
        print("Recovered label: {}".format(recovered_label))
    denoised = denoised.squeeze(0).squeeze(0).detach().cpu().numpy()  # 调整张量形状并转换为NumPy数组
    denoised = librosa.feature.inverse.mel_to_audio(denoised, sr=sr)  # Convert Mel spectrogram back to audio
    sf.write(os.path.join(output_dir, "denoised.wav"), denoised, sr)  # 保存去噪后的音频
    GAN_model.G.train()  # 设置生成器为训练模式
    GAN_model.D.train()  # 设置鉴别器为训练模式


# 用Carlini & Wagner(L2)方法生成对抗样本
def CarliniWagner(file, output_dir=None):
    G = Generator(init_size=16).to(device)  # 创建生成器
    G.l1.in_features = 100
    G.l1.out_features = 8192

    # 加载生成器的权重
    state_dict = torch.load("../generator.pth", map_location=torch.device('cpu'))
    load_my_state_dict(G, state_dict)

    D = Discriminator().to(device)  # 创建鉴别器
    D.adv_layer[0] = nn.Linear(512, 1)

    # 加载鉴别器的权重
    state_dict = torch.load("../discriminator.pth", map_location=torch.device('cpu'))
    load_my_state_dict(D, state_dict)

    GAN_model = MyGANModel().to(device)  # 创建GAN模型
    GAN_model.G = G
    GAN_model.D = D

    audio_file = file
    sr = 48000
    denoise_with_GAN(GAN_model, audio_file, sr, output_dir)  # 用GAN去噪并进行标签恢复


def cw_main(audio_file, output_dir="../Adversarial_example_recognition"):
    audio_file = audio_file  # 从audio_classification_GAN.py获取的结果文件
    CarliniWagner(audio_file, output_dir)
