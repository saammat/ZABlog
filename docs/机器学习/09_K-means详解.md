---
sidebar_position: 9
---

# K-means

**K-means**是一种经典的无监督学习算法，主要用于数据的聚类分析。它的目标是将数据分成KKK 个簇（cluster），使得每个簇内部的样本相似度较高，而不同簇之间的样本差异较大。K-means 聚类广泛应用于图像处理、客户细分、数据降维等场景。

## 1.1 原理

K-means 聚类的基本思想是通过迭代的方式找到$K$个簇的中心，使得簇内数据点的相似度最大，簇间数据点的相似度最小。

### 1.1.1 步骤

- **初始化**：选择$K$个初始聚类中心（质心）。可以随机选择，或者通过其他方法（如 K-means++）来初始化质心，以提高聚类效果。
- **分配簇**：对于数据集中的每个样本，计算它到所有聚类中心的距离（通常使用欧氏距离），并将样本分配给最近的聚类中心。
- **更新质心**：对于每个簇，计算簇内所有样本的均值，并将质心更新为该均值。
- **重复步骤 2 和 3**，直到聚类结果收敛，即簇的划分不再发生变化或达到指定的迭代次数。

### 1.1.2 数学公式

- **初始化聚类中心**：随机选择$K$个样本作为初始聚类中心$\mu_1,\mu_2,...,\mu_K$。
- **分配簇**：对于每个样本$x_i$，计算它与每个聚类中心$\mu_k$的欧氏距离：

$$
d\left(x_{i}, \mu_{k}\right)=\sqrt{\sum_{j=1}^{n}\left(x_{i j}-\mu_{k j}\right)^{2}}
$$

    其中，$x_{ij}$和$\mu_{kj}$分别是样本$x_i$和聚类中心$\mu_k$在第$j$个特征上的值。

    将每个样本分配给距离最近的聚类中心，即：

$$
C_{k}=\left\{x_{i} \mid d\left(x_{i}, \mu_{k}\right) \leq d\left(x_{i}, \mu_{j}\right), \forall j \neq k\right\}
$$
- **更新质心**：对于每个簇$C_k$，计算其所有成员点的均值，更新质心：

$$
\mu_{k}=\frac{1}{\left|C_{k}\right|} \sum_{x_{i} \in C_{k}} x_{i}
$$
- **停止条件**：当聚类中心的变化小于预定的阈值，或者达到最大迭代次数时，算法终止。



## 1.2 应用

K-means 是一种非常常见且高效的聚类算法，适用于各种领域，尤其在以下场景中表现突出：

- **客户细分**：将用户群体按照相似的行为或特征进行划分，从而实施个性化营销。
- **图像压缩**：通过聚类图像中的颜色像素，将颜色数量减少，达到图像压缩的目的。
- **文档聚类**：根据文档的特征（如词频等）对大量文档进行聚类，便于信息检索和组织。
- **异常检测**：通过聚类，找出不符合大多数簇的异常数据点。
- **市场分析**：根据消费者的购买行为、兴趣等数据对客户进行分类，进行个性化推荐。

## 1.3 优缺点

### 1.3.1 优点

- **简单易实现**：K-means 算法结构简单，容易理解和实现，计算效率高。
- **收敛速度快**：对于大多数数据集，K-means 可以在相对较少的迭代次数内收敛。
- **适用于大规模数据集**：K-means 的时间复杂度较低，通常为O(n⋅K⋅I)O(n \cdot K \cdot I)O(n⋅K⋅I)，其中nnn是样本数，KKK是簇数，III是迭代次数，适用于大规模数据。
- **便于可视化**：K-means 聚类结果容易通过图形化方式进行展示，便于分析。
- 聚类效果较好，特别是对于分布均匀、密度相似的数据效果较好。

### 1.3.2 缺点

- **K 的选择问题**：K-means 需要事先指定簇的数量$K$，而实际问题中，$K$通常是未知的，且不同的$K$会导致不同的聚类结果。
- **对初始质心敏感**：K-means 对初始质心选择较为敏感，不同的初始质心可能导致不同的聚类结果，可能陷入局部最优解。
- **对噪声和离群点敏感**：K-means 对噪声和离群点较为敏感，离群点可能导致聚类结果发生偏差。
- **假设簇为圆形或球形**：K-means 假设每个簇是一个圆形或球形区域，因此不适用于形状不规则的簇。
- **无法处理高维数据**：在高维空间中，K-means 的效果可能不佳，因为在高维空间中，数据点之间的距离差异会变得不明显，称为“维度灾难”。

### 1.3.3 优化与改进

- **K-means++**：为了避免 K-means 算法对初始质心选择的敏感性，K-means++ 提供了一种改进的初始化方法，能够更好地选择初始质心，通常可以得到更好的聚类结果。

    K-means++ 的初始质心选择方法为：

    1. 随机选择一个样本作为第一个质心。
    2. 对于剩下的每个样本，计算其与已有质心的最小距离，将该距离作为该样本的概率分布。
    3. 选择下一个质心，使得该样本被选中的概率与其距离的平方成正比。
- **Elbow 方法**：在选择$K$时，可以使用 Elbow 方法，通过绘制聚类误差平方和（SSE）随$K$值变化的图形，寻找 "肘部"（即误差变化的拐点），该点对应的$K$值通常是最佳的聚类数量。

## 1.4 Scikit-Learn实现

**K-means 聚类**（默认K-means++）代码示例：

```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成一个简单的二维数据集
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 使用 K-means++ 初始化，K-means++是默认值
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)

# 训练模型
kmeans.fit(X)

# 获取聚类结果
y_kmeans = kmeans.predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 绘制质心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-means++ Clustering")
plt.show()


```

- `make_blobs`：生成一个简单的二维数据集，这里有 4 个聚类中心。
- `KMeans`：使用`KMeans`类来创建 K-means 聚类模型，`n_clusters=4`表示将数据分成 4 个簇。
- `fit()`：训练 K-means 模型，找到最佳的聚类中心。
- `predict()`：对数据进行预测，获取每个样本所属的簇的标签。
- `cluster_centers_`：访问聚类后的质心。

**选择最佳 K 值（Elbow 方法）**

```Python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 使用肘部法则选择最佳 K 值
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # inertia_ 是每个簇的聚类误差平方和

# 绘制肘部法则图
plt.plot(range(1, 11), sse, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.show()

```

- `inertia_`：返回聚类误差平方和（SSE），即聚类中心到样本点的总距离。
- 通过绘制 SSE 随$K$变化的曲线，可以直观地找到最佳的$K$值。

## 1.5 小结

K-means 是一种高效、简单的聚类算法，适用于处理大规模数据集。尽管它有一些局限性（如对初始质心的敏感性、需要指定$K$等），但在许多应用场景中，它仍然是非常有效的选择。通过结合一些优化方法（如 K-means++ 和 Elbow 方法），可以提高 K-means 聚类的效果和准确性。