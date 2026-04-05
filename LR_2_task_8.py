import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score

# Завантажуємо набір даних Iris
iris = load_iris()

# Ознаки набору даних:
# довжина чашолистка, ширина чашолистка, довжина пелюстки, ширина пелюстки
X = iris.data

# Справжні мітки класів
y = iris.target

# Створюємо модель KMeans для 3 кластерів,
# оскільки в наборі Iris є 3 класи квітів
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)

# Навчаємо модель на даних
kmeans.fit(X)

# Отримуємо мітки кластерів для кожного об’єкта
y_kmeans = kmeans.predict(X)

# Відображаємо результат кластеризації
# Для наочності беремо перші дві ознаки:
# довжина чашолистка та ширина чашолистка
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Отримуємо координати центрів кластерів
centers = kmeans.cluster_centers_

# Відображаємо центри кластерів на графіку
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7)

plt.title('Кластеризація K-середніх для набору даних Iris')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.show()

# Обчислюємо коефіцієнт силуету для оцінки якості кластеризації
score = silhouette_score(X, y_kmeans)
print("Silhouette Score:", round(score, 3))


# Власна реалізація пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізація генератора випадкових чисел
    rng = np.random.RandomState(rseed)

    # Випадковий вибір початкових центрів із наявних точок
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Визначаємо найближчий центр для кожної точки
        labels = pairwise_distances_argmin(X, centers)

        # Обчислюємо нові центри як середнє значення точок кожного кластера
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Якщо центри більше не змінюються, цикл завершується
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels


# Перевірка власної реалізації для 3 кластерів
centers_custom, labels_custom = find_clusters(X, 3)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_custom, s=50, cmap='viridis')
plt.scatter(centers_custom[:, 0], centers_custom[:, 1], c='black', s=200, alpha=0.7)
plt.title('Власна реалізація K-середніх для Iris')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.show()


# Ще один запуск власної реалізації з іншим початковим значенням
centers_custom2, labels_custom2 = find_clusters(X, 3, rseed=0)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_custom2, s=50, cmap='viridis')
plt.scatter(centers_custom2[:, 0], centers_custom2[:, 1], c='black', s=200, alpha=0.7)
plt.title('Власна реалізація K-середніх для Iris (rseed=0)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.show()


# Кластеризація через fit_predict
labels_fit_predict = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_fit_predict, s=50, cmap='viridis')
plt.title('KMeans через fit_predict для Iris')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.show()
