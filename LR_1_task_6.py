import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження даних з файлу
data = np.loadtxt("data_multivar_nb.txt", delimiter=",")
X, y = data[:, :-1], data[:, -1].astype(int)

# Створення наївного байєсовського класифікатора
nb_classifier = GaussianNB()

# Тренування класифікатора
nb_classifier.fit(X, y)

# Прогнозування значень для тренувальних даних
y_pred_nb = nb_classifier.predict(X)

# Обчислення показників якості класифікації
accuracy_nb = accuracy_score(y, y_pred_nb)
precision_nb = precision_score(y, y_pred_nb, average="weighted", zero_division=0)
recall_nb = recall_score(y, y_pred_nb, average="weighted", zero_division=0)
f1_nb = f1_score(y, y_pred_nb, average="weighted", zero_division=0)

print("Naive Bayes classifier metrics:")
print("Accuracy :", round(accuracy_nb, 4))
print("Precision:", round(precision_nb, 4))
print("Recall   :", round(recall_nb, 4))
print("F1-score :", round(f1_nb, 4))

# Створення SVM-класифікатора з нормалізацією ознак
svm_classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", gamma="scale"))
])

# Тренування SVM-класифікатора
svm_classifier.fit(X, y)

# Прогнозування значень для тренувальних даних
y_pred_svm = svm_classifier.predict(X)

# Обчислення показників якості класифікації
accuracy_svm = accuracy_score(y, y_pred_svm)
precision_svm = precision_score(y, y_pred_svm, average="weighted", zero_division=0)
recall_svm = recall_score(y, y_pred_svm, average="weighted", zero_division=0)
f1_svm = f1_score(y, y_pred_svm, average="weighted", zero_division=0)

print("\nSVM classifier metrics:")
print("Accuracy :", round(accuracy_svm, 4))
print("Precision:", round(precision_svm, 4))
print("Recall   :", round(recall_svm, 4))
print("F1-score :", round(f1_svm, 4))

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, stratify=y
)

# Навчання наївного байєсовського класифікатора на навчальних даних
nb_classifier_test = GaussianNB()
nb_classifier_test.fit(X_train, y_train)

# Прогнозування для тестових даних
y_test_pred_nb = nb_classifier_test.predict(X_test)

# Обчислення якості класифікатора
print("\nNaive Bayes (test data):")
print("Accuracy :", round(accuracy_score(y_test, y_test_pred_nb), 4))
print("Precision:", round(precision_score(y_test, y_test_pred_nb, average="weighted", zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_test_pred_nb, average="weighted", zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_test_pred_nb, average="weighted", zero_division=0), 4))

# Навчання SVM-класифікатора на навчальних даних
svm_classifier_test = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", gamma="scale"))
])
svm_classifier_test.fit(X_train, y_train)

# Прогнозування для тестових даних
y_test_pred_svm = svm_classifier_test.predict(X_test)

# Обчислення якості класифікатора
print("\nSVM (test data):")
print("Accuracy :", round(accuracy_score(y_test, y_test_pred_svm), 4))
print("Precision:", round(precision_score(y_test, y_test_pred_svm, average="weighted", zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_test_pred_svm, average="weighted", zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_test_pred_svm, average="weighted", zero_division=0), 4))
