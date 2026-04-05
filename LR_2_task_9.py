import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Завантаження вхідних даних з файлу
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна (bandwidth) для набору даних X
# Параметр quantile впливає на розмір вікна:
# чим більше значення, тим ширше вікно і тим менше кластерів
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))

# Кластеризація даних методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_model.fit(X)

# Витягування міток кластерів для кожної точки
labels = meanshift_model.labels_

# Витягування центрів кластерів
cluster_centers = meanshift_model.cluster_centers_

# Оцінка кількості кластерів
num_clusters = len(np.unique(labels))

# Виведення результатів у термінал
print("Оцінена ширина вікна (bandwidth):", round(bandwidth, 3))
print("Кількість кластерів:", num_clusters)
print("Координати центрів кластерів:")
for i, center in enumerate(cluster_centers):
    print(f"Кластер {i + 1}: {center}")

# Відображення на графіку точок та центрів кластерів
plt.figure()

# Кольори для різних кластерів
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

for i, color in zip(range(num_clusters), colors):
    # Вибір точок поточного кластера
    cluster_points = X[labels == i]

    # Відображення точок поточного кластера
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=color, s=50, label=f'Кластер {i + 1}')

    # Відображення центру поточного кластера
    plt.scatter(cluster_centers[i][0], cluster_centers[i][1],
                color='black', marker='x', s=200, linewidths=3)

plt.title('Кластеризація методом зсуву середнього')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.legend()
plt.grid(True)
plt.show()
