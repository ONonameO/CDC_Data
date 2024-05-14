from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
  
# 创建一个聚类数据集，共有500个样本点，每个样本有2个特征，共有3类
bolobs_X, blobs_y = make_blobs(n_samples=500, centers=5, n_features=2, random_state=20)
# 创建一个形状为两个半月形的数据集，共有500个样本点
moons_X, moons_y = make_moons(n_samples=500, noise=0.1, random_state=30)
  
# 可视化数据集  
plt.scatter(bolobs_X[:, 0], bolobs_X[:, 1], c=blobs_y, cmap='Set1', marker='o')
plt.show()

plt.scatter(moons_X[:, 0], moons_X[:, 1], c=moons_y, cmap='Set1', marker='o')
plt.show()


# 将 特征X 和 标签y 转换为 DataFrame  
blobs_df_X = pd.DataFrame(bolobs_X, columns=[f'feature_{i}' for i in range(bolobs_X.shape[1])])  
blobs_df_y = pd.DataFrame(blobs_y+1, columns=['label'])

moons_df_X = pd.DataFrame(moons_X, columns=[f'feature_{i}' for i in range(moons_X.shape[1])])  
moons_df_y = pd.DataFrame(moons_y+1, columns=['label']) 


# 如果你想将X和y保存在同一个CSV文件中，可以先合并它们  
blobs_df = pd.concat([blobs_df_X, blobs_df_y], axis=1)
moons_df = pd.concat([moons_df_X, moons_df_y], axis=1)
  
# 保存DataFrame到CSV文件  
blobs_df.to_csv('data/blobs_data.txt', sep='\t',index=False,header=False)
moons_df.to_csv('data/moons_data.txt', sep='\t',index=False,header=False)

# 统计分类分布
blobs_counts = Counter(blobs_y)
moons_counts = Counter(moons_y)
print(blobs_counts)
print(moons_counts)