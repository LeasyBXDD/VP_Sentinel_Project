## 基于GAN的对抗样本生成模块

### 1. 项目演示

[demo](./demo/vc/index.html)

### 2. 运行代码文件

> 根据原始 `inference.ipynb` 文件

```python
from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from agent.inferencer import Inferencer


def get_args():
    parser = get_parser(description='Inference')

    # required
    parser.add_argument('--load', '-l', type=str, help='Load a checkpoint.', required=True)
    parser.add_argument('--source', '-s', help='Source path. A .wav file or a directory containing .wav files.',
                        required=True)
    parser.add_argument('--target', '-t', help='Target path. A .wav file or a directory containing .wav files.',
                        required=True)
    parser.add_argument('--output', '-o', help='Output directory.', required=True)

    # config
    parser.add_argument('--config', '-c', help='The train config with respect to the model resumed.',
                        default='./config/train_again-c4s.yaml')
    parser.add_argument('--dsp-config', '-d', help='The dsp config with respect to the training data.',
                        default='./config/preprocess.yaml')

    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')

    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    # [--log-steps <LOG_STEPS>]
    parser.add_argument('--njobs', '-p', type=int, help='', default=4)
    parser.add_argument('--seglen', help='Segment length.', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)
    same_seeds(args.seed)
    args.dsp_config = Config(args.dsp_config)

    # build inferencer
    inferencer = Inferencer(config=config, args=args)

    # inference
    inferencer.inference(source_path=args.source, target_path=args.target, out_path=args.output, seglen=args.seglen)
```

### 3. 运行

pip install librosa==0.8.1
pip install --upgrade numpy

```
python ./VCadv/inference.py --load "./checkpoints/again/c4s/again-c4s_100000.pth" --source "../lib/wav48/p225/p225_001.wav" --target "../lib/wav48/p226/p226_001.wav" --output "./output"
```

### 4. 材料

[论文](https://arxiv.org/abs/2011.00316)

### 5. 后续

自动运行程序，使用[`LibriSpeech`](http://www.openslr.org/12/)，生成对抗样本，用于后续判别器训练
