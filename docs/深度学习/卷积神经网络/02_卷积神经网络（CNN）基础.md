---
sidebar_position: 2
---

# 卷积神经网络（CNN）基础

## 1.1 CNN 简介

卷积神经网络（Convolutional Neural Network, CNN）是一种专门用于处理网格结构数据（如图像）的深度学习模型。CNN 主要通过局部连接和权重共享来减少计算复杂度，同时利用卷积操作提取数据的空间特征。

## 1.2 CNN 的核心原理

### 1.2.1 卷积层（Convolutional Layer）

卷积层是 CNN 的核心组件，利用卷积操作提取局部特征。对于输入图像$I $和卷积核$K$，二者的卷积计算如下：

$$
O(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) K(m, n)
$$

其中，$O(i,j)$是输出特征图，$M$和$N $是卷积核的尺寸。

### 1.2.2 激活函数（Activation Function）

常用的激活函数是 ReLU（修正线性单元）：

$$
ReLU(x) = \max(0, x)
$$

ReLU 可以增加模型的非线性能力，并避免梯度消失问题。

### 1.2.3 池化层（Pooling Layer）

池化层用于减少特征图的尺寸，提高计算效率，并增强模型的平移不变性。常见的池化操作有最大池化（Max Pooling）：

$$
O(i, j) = \max_{m=0}^{M-1} \max_{n=0}^{N-1} I(i+m, j+n)
$$

### 1.2.4 全连接层（Fully Connected Layer）

全连接层用于将提取的特征映射到最终的分类或回归输出。

## 1.3 CNN 的应用场景

CNN 主要应用于以下领域：

- **计算机视觉**：图像分类、目标检测、语义分割
- **医学影像分析**：X 光片、CT 扫描分析
- **自动驾驶**：目标识别、车道检测
- **自然语言处理（NLP）**：文本分类、情感分析

## 1.4 CNN 的优缺点

### 1.4.1 优点

- 能够自动学习特征，无需人工设计
- 共享权重减少参数，提高计算效率
- 具有平移、缩放不变性，适用于图像任务

### 1.4.2 缺点

- 计算量较大，对硬件要求高
- 训练时间较长，容易过拟合
- 对旋转不变性处理较差，需要数据增强

## 1.5 PyTorch 代码示例

```Python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义 CNN 网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 训练 CNN
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    for epoch in range(5):  # 训练 5 轮
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

train_model()
```

## 1.6 结论

CNN 是深度学习中最重要的架构之一，广泛应用于计算机视觉等领域。尽管 CNN 具有强大的特征提取能力，但也有计算量大、易过拟合的问题。PyTorch 提供了灵活的框架，可以高效地搭建和训练 CNN 模型。