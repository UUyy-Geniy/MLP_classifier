import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функция для обучения моделей и расчета метрик
def evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    # Инициализация моделей с настройкой гиперпараметров для MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='tanh', solver='adam',
                        alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    svm = SVC(random_state=42)

    # Обучение моделей
    logging.info(f"Начало обучения MLP Classifier для {dataset_name}...")
    mlp.fit(X_train, y_train)
    logging.info("Обучение MLP Classifier завершено.")

    logging.info(f"Начало обучения Random Forest для {dataset_name}...")
    random_forest.fit(X_train, y_train)
    logging.info("Обучение Random Forest завершено.")

    logging.info(f"Начало обучения SVM для {dataset_name}...")
    svm.fit(X_train, y_train)
    logging.info("Обучение SVM завершено.")

    # Предсказания для моделей классификации
    y_pred_mlp = mlp.predict(X_test)
    y_pred_rf = random_forest.predict(X_test)
    y_pred_svm = svm.predict(X_test)

    # Оценка метрик для каждой модели
    metrics = {}

    # Accuracy
    metrics['MLP Accuracy'] = accuracy_score(y_test, y_pred_mlp)
    metrics['RF Accuracy'] = accuracy_score(y_test, y_pred_rf)
    metrics['SVM Accuracy'] = accuracy_score(y_test, y_pred_svm)

    # Precision
    metrics['MLP Precision'] = precision_score(y_test, y_pred_mlp, average='macro')
    metrics['RF Precision'] = precision_score(y_test, y_pred_rf, average='macro')
    metrics['SVM Precision'] = precision_score(y_test, y_pred_svm, average='macro')

    # Recall
    metrics['MLP Recall'] = recall_score(y_test, y_pred_mlp, average='macro')
    metrics['RF Recall'] = recall_score(y_test, y_pred_rf, average='macro')
    metrics['SVM Recall'] = recall_score(y_test, y_pred_svm, average='macro')

    # F1 Score
    metrics['MLP F1 Score'] = f1_score(y_test, y_pred_mlp, average='macro')
    metrics['RF F1 Score'] = f1_score(y_test, y_pred_rf, average='macro')
    metrics['SVM F1 Score'] = f1_score(y_test, y_pred_svm, average='macro')

    # Добавляем название датасета
    metrics['Dataset'] = dataset_name

    # Возвращаем метрики как словарь
    return metrics

# Список для хранения результатов для обоих датасетов
results = []

# ======= Анализ CIFAR-10 Dataset =======
logging.info("Начинается анализ CIFAR-10 Dataset...")
cifar10 = fetch_openml('CIFAR_10', version=1)

X_cifar = cifar10.data
y_cifar = cifar10.target.astype(int)

# Разделяем данные на тренировочную и тестовую выборки
X_train_cifar, X_test_cifar, y_train_cifar, y_test_cifar = train_test_split(X_cifar, y_cifar, test_size=0.3, random_state=42)

# Нормализуем данные для CIFAR-10
scaler_cifar = StandardScaler()
X_train_cifar = scaler_cifar.fit_transform(X_train_cifar.astype(float))
X_test_cifar = scaler_cifar.transform(X_test_cifar.astype(float))

# Получаем метрики для CIFAR-10
cifar_metrics = evaluate_models(X_train_cifar, X_test_cifar, y_train_cifar, y_test_cifar, "CIFAR-10")
results.append(cifar_metrics)

# ======= Анализ Wine Quality Dataset =======
logging.info("Начинается анализ Wine Quality Dataset...")
wine_quality = fetch_openml(name='wine-quality-white', version=1)
X_wine = wine_quality.data
y_wine = wine_quality.target.astype(int)

# Разделяем данные на тренировочную и тестовую выборки
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# Нормализуем данные для Wine
scaler_wine = StandardScaler()
X_train_wine = scaler_wine.fit_transform(X_train_wine.astype(float))
X_test_wine = scaler_wine.transform(X_test_wine.astype(float))

# Получаем метрики для Wine Quality
wine_metrics = evaluate_models(X_train_wine, X_test_wine, y_train_wine, y_test_wine, "Wine Quality")
results.append(wine_metrics)

# Преобразуем результаты в DataFrame для удобства анализа
df_results = pd.DataFrame(results)

# Упорядочиваем столбцы для лучшего отображения
columns_order = ['Dataset',
                 'MLP Accuracy', 'RF Accuracy', 'SVM Accuracy',
                 'MLP Precision', 'RF Precision', 'SVM Precision',
                 'MLP Recall', 'RF Recall', 'SVM Recall',
                 'MLP F1 Score', 'RF F1 Score', 'SVM F1 Score']

# Применяем порядок столбцов
df_results = df_results[columns_order]

# Транспонирование DataFrame
df_results_t = df_results.T

# Выводим результаты
print("Транспонированный DataFrame:")
print(df_results_t)
