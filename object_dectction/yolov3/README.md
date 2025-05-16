# YOLOv3 实战
***
# 该项目源自https://github.com/ultralytics/yolov3
***
# 1. 环境配置
* Python 3.7 或 3.8(为了防止出错，大家下载python版本可以和我一致Python=3.13.2)
* PyTorch ≥ 1.6.0 才能用官方 torch.cuda.amp 混合精度训练。
*  torch torchvision torchaudio 这三个包手动安装cuda版本的(requirements.txt里面是cpu版本的需要注释掉torchvision和torchaudio)
*  更多环境配置信息，请查看requirements.txt文件
*  使用gpu训练
***
# 2. 📁文件结构

```

yolov3/
├── data/                  # 数据相关资源目录
│   ├── hyps/               # 超参数配置文件（不同训练策略）
│   │   ├── hyp.Objects365.yaml
│   │   ├── hyp.VOC.yaml
│   │   ├── hyp.no-augmentation.yaml
│   │   ├── hyp.scratch-high.yaml
│   │   ├── hyp.scratch-low.yaml
│   │   └── hyp.scratch-med.yaml
│   ├── images/             # 示例图像（用于测试或演示）
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   ├── scripts/            # 数据集下载脚本
│   │   ├── download_weights.sh
│   │   ├── get_coco.sh
│   │   ├── get_coco128.sh
│   │   └── get_imagenet.sh
│   ├── videos/             # 测试视频
│   │   └── barvideoTest.mp4
│   ├── coco.yaml           # COCO 数据集定义文件
│   ├── coco128.yaml        # COCO128 小样本数据集定义文件
│   └── my_yolo_data.yaml   # 自定义数据集配置文件
│
├── models/                # 模型结构定义目录
│   ├── yolo.yaml           # YOLOv3 模型结构配置文件（如 backbone、head 等）
│   ├── common.py           # 常用模型组件（如 Conv、Bottleneck 等）
│   ├── experimental.py     # 实验性模块（可能包含新结构或改进）
│   └── ...                 # 其他模型相关文件（如损失函数、模型构建逻辑等）
│
├── utils/                 # 工具函数目录
│   ├── datasets.py         # 数据集处理工具（如加载、增强、预处理等）
│   ├── loss.py             # 损失函数实现
│   ├── metrics.py          # 评估指标（如 mAP、Precision、Recall）
│   ├── general.py          # 通用工具函数（如文件操作、日志、绘图等）
│   └── ...                 # 其他辅助功能
│
├── weights/               # 模型权重目录
│   ├── yolov3.pt           # 预训练权重文件（可用于迁移学习）
│   └── ...                 # 其他模型权重文件
│
├── train.py               # 训练主程序入口
├── val.py                 # 验证脚本
├── test.py                # 测试脚本
├── detect.py              # 推理脚本（用于图像、视频的目标检测）
├── export.py              # 模型导出脚本（ONNX、TorchScript 等格式）
└── README.md              # 项目说明文档
```

注释说明
| 文件/目录 | 作用 |
|----------|------|
| `data/` | 存放与数据相关的资源，包括数据集定义、超参数、示例图像和下载脚本。 |
| `models/` | 定义 YOLOv3 的网络结构、组件以及模型构建逻辑。 |
| `utils/` | 提供各种工具函数，涵盖数据处理、损失计算、评估指标等功能。 |
| `weights/` | 存放预训练模型权重文件，用于初始化训练或推理。 |
| `train.py` | 主训练脚本，控制训练流程、参数设置、优化器选择等。 |
| `val.py` | 验证脚本，用于在验证集上评估模型性能。 |
| `test.py` | 测试脚本，通常用于对测试集进行推理并输出结果。 |
| `detect.py` | 推理脚本，支持图像、视频输入的目标检测任务。 |
| `export.py` | 模型导出脚本，支持将模型转换为 ONNX、TorchScript 等部署格式。 |
| `README.md` | 项目说明文档，介绍使用方法、依赖、训练技巧等信息。 |
***
#  3. 标注数据集
* 标注数据格式为yolo格式，标注工具：https://github.com/tzutalin/labelImg
* 标注好的数据集请按照以下目录结构:

```
.
└── my_yolo_data/
    ├── images/
    │   ├── train
    │   └── val
    └── labels/
        ├── train
        └── val
```
# 4. 使用pascal voc格式的公共数据集(自己标注数据集的请跳过此步骤)
* 下载公共数据集后在数据集下面创建classes.txt文件，存放类别名称，以pascal voc2012为例：
```
.
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations
        ├── JPEGImages
        └── classes.txt
```
* 修改voc2yolo.py文件
修改文件里面对应的路径信息
```
voc_images_dir = '../datasets/VOCdevkit/VOC2012/JPEGImages'
voc_annotations_dir = '../datasets/VOCdevkit/VOC2012/Annotations'
classes_path = '../datasets/VOCdevkit/VOC2012/classes.txt'
output_dir = '../datasets/my_yolo_data'  # 最终输出的根目录
```
修改后然后运行脚本，会生成my_yolo_data目录结构，里面有images和labels两个目录，分别存放图片和标签文件

* my_yolo_data.yaml文件
 请自行修改my_yolo_data.yaml文件，将数据集路径修改为自己的my_yolo_data目录路径，并把里面classes信息和classes.txt文件的class名称顺序一致并保存。 
```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/my_yolo_data # dataset root dir
train: images/train # train images (relative to 'path') 128 images
val: images/val # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes
names:
  0: aeroplane
  1: bicycle
  2: bird
  3: boat
  4: bottle
  5: bus
  6: car
  7: cat
  8: chair
  9: cow
  10: diningtable
  11: dog
  12: horse
  13: motorbike
  14: person
  15: pottedplant
  16: sheep
  17: sofa
  18: train
  19: tvmonitor
```

# 5 预训练权重下载地址：
* 本套代码自带下载从github上下载的yolov3.pt权重(运行时会自动下载)
* 也可以关注公众号：智算学术 和我取得联系，获取权重下载地址

# 6 数据集，本教程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# 7 训练
* 运行train.py：python train.py --data my_yolo_data.yaml --weights yolov3.pt
* 运行predict.py：python predict.py --weights yolov3.pt --source data/images

# 8 其他
* yolov3技术已经过时了，但是作为学习的资料还是不错的，在yolo5和yolo8等系列中会更新一些训练和调参的技巧
* yolov3 学习资源
```
* yolov1 论文原理讲解：https://www.bilibili.com/video/BV1PKLdzDECC
* yolov1 论文原理万字解读文字版：https://mp.weixin.qq.com/s/kPuk1ZNSCIMDc47F9TkdGw
* yolov2 论文原理讲解：https://www.bilibili.com/video/BV1cPG9zeEXN
* yolov2 论文原理万字解读文字版：https://mp.weixin.qq.com/s/6AX1Elcz7s-tyzfgneAocA
* yolov3 论文原理讲解：https://www.bilibili.com/video/BV18iVRzWEEa
* yolov3 论文原理万字解读文字版：https://mp.weixin.qq.com/s/_KbT-184mZL24rxrlmzvcg
```
