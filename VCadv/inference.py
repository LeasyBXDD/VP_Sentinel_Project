from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from agent.inferencer import Inferencer


def get_args():
    # 创建一个命令行参数解析器
    parser = get_parser(description='推断')

    # 必须的参数
    parser.add_argument('--load', '-l', type=str, help='加载一个检查点。', required=True)  # 加载预训练模型
    parser.add_argument('--source', '-s', help='源路径。一个.wav文件或包含.wav文件的目录。',
                        required=True)  # 输入源文件或目录
    parser.add_argument('--target', '-t', help='目标路径。一个.wav文件或包含.wav文件的目录。',
                        required=True)  # 输入目标文件或目录
    parser.add_argument('--output', '-o', help='输出目录。', required=True)  # 输出文件目录

    # 配置
    parser.add_argument('--config', '-c', help='关于恢复模型的训练配置。',
                        default='./config/train_again-c4s.yaml')  # 训练配置
    parser.add_argument('--dsp-config', '-d', help='关于训练数据的 dsp 配置。',
                        default='./config/preprocess.yaml')  # 预处理配置

    # dryrun
    parser.add_argument('--dry', action='store_true', help='是否进行干运行')  # 干运行参数，如果设置为True，程序将不会执行实际操作

    # 调试模式
    parser.add_argument('--debug', action='store_true', help='调试模式')  # 调试模式参数

    # 随机种子
    parser.add_argument('--seed', type=int, help='随机种子', default=961998)  # 设置随机种子

    # [--log-steps <LOG_STEPS>]
    parser.add_argument('--njobs', '-p', type=int, help='并发处理的任务数量', default=4)  # 并发任务数量
    parser.add_argument('--seglen', help='段落长度。', type=int, default=None)  # 设置处理音频的段落长度

    return parser.parse_args()


if __name__ == '__main__':
    # 配置
    args = get_args()  # 获取命令行参数
    config = Config(args.config)  # 加载训练配置
    same_seeds(args.seed)  # 设置随机种子
    args.dsp_config = Config(args.dsp_config)  # 加载预处理配置

    # 构建推断器
    inferencer = Inferencer(config=config, args=args)  # 创建Inferencer实例

    # 推断
    inferencer.inference(source_path=args.source, target_path=args.target, out_path=args.output,
                         seglen=args.seglen)  # 进行推断