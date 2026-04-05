import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# Завантаження вхідних даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Кількість кластерів
num_clusters = 5

# Візуалізація вхідних даних
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Вхідні дані')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# Створення моделі KMeans
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=0)

# Навчання моделі
kmeans.fit(X)

# Побудова сітки для візуалізації областей кластерів
step_size = 0.01
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size))

# Прогноз для кожної точки сітки
mesh_output = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])

# Перетворення у форму сітки
mesh_output = mesh_output.reshape(x_values.shape)

# Відображення результатів кластеризації
plt.figure()
plt.clf()
plt.imshow(mesh_output, interpolation='nearest',
           extent=(x_values.min(), x_values.max(),
                   y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')

plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)

# Центроїди
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='*', s=300, linewidths=2,
            color='black', zorder=10)

plt.title('Результати кластеризації методом k-середніх')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# Оцінка якості кластеризації
score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
print("Silhouette Score:", round(score, 3))
