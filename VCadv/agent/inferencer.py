import os
import torch
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

from preprocessor.base import preprocess_one
from .base import BaseAgent
from util.dsp import Dsp

logger = logging.getLogger(__name__)

# 定义生成音频文件列表的函数
def gen_wav_list(path):
    # 如果路径是一个目录，获取该目录下所有的.wav文件
    if os.path.isdir(path):
        wav_list = glob(os.path.join(path, '*.wav'))
    # 如果路径是一个文件，将该文件加入音频文件列表
    elif os.path.isfile(path):
        wav_list = [path]
    # 如果路径既不是文件也不是目录，抛出异常
    else:
        raise NotImplementedError(f'{path} is invalid for generating wave file list.')
    return wav_list

# 定义音频数据类
class WaveData():
    def __init__(self, path):
        # 初始化音频文件路径
        self.path = path
        # 初始化处理标记为False
        self.processed = False
        # 初始化数据字典
        self.data = {}

    # 设置处理标记为True
    def set_processed(self):
        self.processed = True

    # 判断是否已经处理过
    def is_processed(self):
        return self.processed

    # 获取数据字典中的某个值
    def __getitem__(self, key):
        # 如果键是字符串类型，返回对应的值
        if type(key) is str:
            return self.data[key]
        # 否则抛出异常
        else:
            raise NotImplementedError

    # 设置数据字典中的某个值
    def __setitem__(self, key, value):
        # 如果键是字符串类型，设置对应的值
        if type(key) is str:
            self.data[key] = value
        # 否则抛出异常
        else:
            raise NotImplementedError

# 定义推理器类，继承自基础代理类
class Inferencer(BaseAgent):
    def __init__(self, config, args, trust_repo=False):
        # 调用父类的初始化方法
        super().__init__(config, args, trust_repo=trust_repo)
        # 初始化索引路径
        self.indexes_path = config.dataset.indexes_path
        # 初始化DSP模块字典
        self.dsp_modules = {}
        # 遍历配置中的特征列表
        for feat in config.dataset.feat:
            # 如果特征已经在DSP模块字典中，获取对应的模块
            if feat in self.dsp_modules.keys():
                module = self.dsp_modules[feat]
            # 否则，创建新的DSP模块，并添加到字典中
            else:
                module = Dsp(args.dsp_config.feat[feat])
                self.dsp_modules[feat] = module
        # 构建模型
        self.model_state, self.step_fn = self.build_model(config.build)
        # 加载模型
        self.model_state = self.load_model(self.model_state, args.load)

# class Inferencer(BaseAgent):
#     def __init__(self, config, args, trust_repo=False):
#         super().__init__(config, args, trust_repo=trust_repo)
#         self.indexes_path = config.dataset.indexes_path
#         self.dsp_modules = {}
#         for feat in config.dataset.feat:
#             if feat in self.dsp_modules.keys():
#                 module = self.dsp_modules[feat]
#             else:
#                 module = Dsp(args.dsp_config.feat[feat])
#                 self.dsp_modules[feat] = module
#         self.model_state, self.step_fn = self.build_model(config.build)
#         self.model_state = self.load_model(self.model_state, args.load)

    def build_model(self, build_config):
        return super().build_model(build_config, mode='inference', device=self.device)

    def load_wav_data(self, source_path, target_path, out_path):
        # 加载音频文件
        # 生成源音频文件列表
        sources = gen_wav_list(source_path)
        # 检查源音频文件列表是否为空，如果为空则抛出异常
        assert(len(sources) > 0), f'Source path "{source_path}" should be a wavefile or a directory which contains wavefiles.'
        # 生成目标音频文件列表
        targets = gen_wav_list(target_path)
        # 检查目标音频文件列表是否为空，如果为空则抛出异常
        assert(len(targets) > 0), f'Target path "{target_path}" should be a wavefile or a directory which contains wavefiles.'
        # 检查输出路径是否存在
        if os.path.exists(out_path):
            # 如果存在，检查是否是目录，如果不是则抛出异常
            assert(os.path.isdir(out_path)), f'Output path "{out_path}" should be a directory.'
        else:
            # 如果不存在，创建输出目录
            os.makedirs(out_path)
            logger.info(f'Output directory "{out_path}" is created.')
        # 创建输出目录和子目录
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path, 'wav'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'plt'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'mel'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'npy'), exist_ok=True)

        # 将源音频文件列表和目标音频文件列表中的每个文件封装为WaveData对象
        for i, source in enumerate(sources):
            sources[i] = WaveData(source)
        for i, target in enumerate(targets):
            targets[i] = WaveData(target)

        return sources, targets, out_path

    def process_wave_data(self, wav_data, seglen=None):
        # 如果音频数据已经处理过，直接返回
        if wav_data.is_processed():
            return
        else:
            # 否则，对音频数据进行预处理，并保存结果
            wav_path = wav_data.path
            basename = os.path.basename(wav_path)
            for feat, module in self.dsp_modules.items():
                wav_data[feat] = preprocess_one((wav_path, basename), module)
                if seglen is not None:
                    wav_data[feat] = wav_data[feat][:,:seglen]
                wav_data.set_processed()
            return

    # ====================================================
    #  inference
    # ====================================================
    def inference(self, source_path, target_path, out_path, seglen):
        # 加载源和目标音频文件的路径，并获取输出路径
        sources, targets, out_path = self.load_wav_data(source_path, target_path, out_path)

        # 使用torch.no_grad()上下文管理器，关闭梯度计算以减少内存使用并加速计算
        with torch.no_grad():
            # 遍历源音频文件
            for i, source in enumerate(sources):
                logger.info(f'Source: {source.path}')
                # 遍历目标音频文件
                for j, target in enumerate(targets):
                    logger.info(f'Target: {target.path}')
                    # 从源和目标音频文件的路径中获取文件名（无扩展名）
                    source_basename = os.path.basename(source.path).split('.wav')[0]
                    target_basename = os.path.basename(target.path).split('.wav')[0]
                    # 创建输出文件名，格式为"source_to_target"
                    output_basename = f'{source_basename}_to_{target_basename}'
                    # 创建输出音频文件和谱图文件的路径
                    output_wav = os.path.join(out_path, 'wav', output_basename + '.wav')
                    output_plt = os.path.join(out_path, 'plt', output_basename + '.png')
                    # 对源和目标音频数据进行处理
                    self.process_wave_data(source, seglen=seglen)
                    self.process_wave_data(target, seglen=seglen)
                    # 创建包含源和目标音频数据的字典
                    data = {
                        'source': source,
                        'target': target,
                    }
                    # 执行模型推断步骤，获取元数据
                    meta = self.step_fn(self.model_state, data)
                    # 从元数据中获取解码后的音频数据
                    dec = meta['dec']
                    # 将解码后的音频数据转换为音频文件，并保存
                    self.mel2wav(dec, output_wav)
                    # 绘制谱图，并保存
                    Dsp.plot_spectrogram(dec.squeeze().cpu().numpy(), output_plt)

                    # 创建源音频数据的谱图文件路径
                    source_plt = os.path.join(out_path, 'plt', f'{source_basename}.png')
                    # 绘制源音频数据的谱图，并保存
                    Dsp.plot_spectrogram(source['mel'], source_plt)
                    # 保存源音频数据的Mel谱图为.npy文件
                    np.save(os.path.join(out_path, 'mel', f'{source_basename}.npy'), source['mel'])
        # 打印输出路径，表示已保存生成的文件
        logger.info(f'The generated files are saved to {out_path}.')