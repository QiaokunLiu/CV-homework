# ShanghaiTech人群密度估计

这个项目使用MobileNetV3实现了对ShanghaiTech数据集的人群密度估计。

## 项目结构
```
.
├── data/                   # 数据集目录
├── models/                 # 模型定义
├── utils/                  # 工具函数
├── train.py               # 训练脚本
├── test.py                # 测试脚本
└── requirements.txt       # 项目依赖
```

## 环境配置
```bash
pip install -r requirements.txt
```

## 使用方法
1. 下载ShanghaiTech数据集并解压到data目录
2. 运行训练脚本：
```bash
python train.py
```
3. 运行测试脚本：
```bash
python test.py
``` 