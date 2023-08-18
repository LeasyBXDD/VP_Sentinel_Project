import pymysql
import numpy as np

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from db_func import funcs  # 导入自定义的数据库操作函数


# 定义用于预测两个音频文件是否来自同一个说话人的函数
def predict_speaker_similarity(filename_1, filename_2, base_model, tdnn_model):
    # 将音频文件转换为MFCC特征
    mfcc_1 = sample_from_mfcc(read_mfcc(filename_1, SAMPLE_RATE), NUM_FRAMES)
    mfcc_2 = sample_from_mfcc(read_mfcc(filename_2, SAMPLE_RATE), NUM_FRAMES)

    # 获取每个文件的嵌入向量（形状为(1, 512)）
    embeddings_1 = base_model.m.predict(np.expand_dims(mfcc_1, axis=0))
    embeddings_2 = base_model.m.predict(np.expand_dims(mfcc_2, axis=0))

    # 在嵌入向量上应用TDNN模型
    tdnn_predict_1 = tdnn_model.predict(embeddings_1)
    tdnn_predict_2 = tdnn_model.predict(embeddings_2)

    # 计算余弦相似度并返回结果
    return batch_cosine_similarity(tdnn_predict_1, tdnn_predict_2)


# 定义基础模型并加载预训练的权重
base_model = DeepSpeakerModel()
base_model.m.load_weights("../ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

# 定义TDNN模型
tdnn_input_shape = (512, 1)  # TDNN模型的输入形状应基于基础模型的输出
tdnn_output_dim = 128  # 您可以根据需要调整此值
tdnn_model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=tdnn_input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(units=tdnn_output_dim)
])
tdnn_model.compile(loss='mse', optimizer='adam')


def rescnntdnn():
    # 从数据库中获取音频文件路径
    audio_files = funcs.get_audio_files_from_database()

    # 为所有音频文件进行比较
    compare_all_audio_files(audio_files)

    # 对每一对音频文件进行预测，并将结果保存到数据库中
    for i in range(len(audio_files)):
        for j in range(i + 1, len(audio_files)):
            filename_1 = audio_files[i]
            filename_2 = audio_files[j]
            same_speaker_similarity = predict_speaker_similarity(filename_1, filename_2, base_model, tdnn_model)
            funcs.save_result_to_database(same_speaker_similarity, filename_1, filename_2)
