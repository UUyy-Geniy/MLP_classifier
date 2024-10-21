import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
import joblib
import os
from sklearn.datasets import fetch_openml

alpha = 0.0001
hidden_layer_size = (64, 32)
max_iter = 200


def iris_sample(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    """
    Параметры
    MLPClassifier:

    1. ** hidden_layer_sizes ** (default=(100,)): - Размер скрытых слоев нейронной
    сети. Можно указать одно число(один слой) или кортеж чисел(несколько слоев).
    - Пример: (64, 16, 4) — три слоя с 64, 16 и 4 нейронами соответственно.
    2. ** activation ** (default='relu'): - Функция активации для скрытых слоев.
        - Возможные значения:
            - 'identity' — линейная функция
                f(x) = x.
            - 'logistic' — логистическая функция сигмоиды.
            - 'tanh' — гиперболический тангенс.
            - 'relu' — ректифицированная линейная
            функция(Rectified Linear Unit).

    3. ** solver ** (default='adam'):
    - Алгоритм оптимизации весов.
    - Возможные значения:
        - 'adam' — стохастический градиентный метод(рекомендуется для большинства задач).
        - 'lbfgs' — оптимизация с использованием метода Левенберга - Марквардта(используется для малых
        данных).
        - 'sgd' — стохастический градиентный спуск.

    4. ** alpha ** (default=0.0001):
    - Параметр регуляризации L2, который предотвращает
    переобучение, добавляя штраф за большие веса
    модели.

    5. ** batch_size ** (default='auto'):
    - Размер мини - пакета данных для обучения. Если
    указано 'auto', размер пакета равен 200.

    6. ** learning_rate ** (default='constant'):
    - Стратегия изменения скорости обучения.
    - Возможные значения:
        - 'constant' — постоянная скорость обучения.
        - 'invscaling' — скорость уменьшается с шагом обучения.
        - 'adaptive' — скорость обучения уменьшается, если модель не улучшает точность.
    
    7. ** max_iter ** (default=200):
        - Максимальное количество итераций(эпох) для оптимизации.
    
    8. ** random_state ** (default=None):
        - Начальное состояние случайного генератора для воспроизводимости результатов.
    
    9. ** tol ** (default=1e-4):
        - Допуск для остановки обучения, если улучшение потерь меньше этого значения.
    
    10. ** early_stopping ** (default=False):
        - Остановка обучения при отсутствии улучшений на валидационной выборке. 
        - Если True, то модель будет делить данные на обучение и валидацию.
    
    11. ** learning_rate_init ** (default=0.001): 
    - Начальная скорость обучения, используемая при обучении с методом 'sgd' или 'adam'.
    
    12. ** momentum ** (default=0.9):  
    - Используется для ускорения сходимости при использовании метода 'sgd'.
    
    13. ** n_iter_no_change ** (default=10): 
        - Количество итераций без улучшений, после которых обучение будет остановлено
        (если используется 'early_stopping').
    
    14. ** validation_fraction ** (default=0.1):
        - Доля данных, используемая для валидации при 'early_stopping'. 
    
    15. ** shuffle ** (default=True):
    - Перемешивать ли данные перед каждой итерацией.
    
    16. ** verbose ** (default=False):
    - Выводить ли на экран промежуточные результаты работы модели.
    
    17. ** warm_start ** (default=False):
        - Если True, модель будет продолжать обучение с момента, 
        на котором остановилась в предыдущем вызове метода `fit`.
    """

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size,
                        max_iter=max_iter, activation='relu',
                        alpha=alpha, solver='adam',
                        random_state=42)

    mlp.fit(X_train, y_train)

    # Предсказание на тестовых данных
    y_pred = mlp.predict(X_test)
    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy * 100:.2f}%")


def iris_sample_with_grid_search(X, y, verbose=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    mlp = MLPClassifier(random_state=42, tol=1e-3, solver="adam")

    # Задание гиперпараметров для GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [(32,), (64, 32), (128, 64), (64, 16, 4)],  # Различные архитектуры сети
        'alpha': [0.0001, 0.01],  # Коэффициенты регуляризации
        'learning_rate': ['constant', 'adaptive'],  # Стратегии изменения скорости обучения
        'max_iter': [200, 400, 800],  # Количество итераций
    }

    # Настройка GridSearchCV
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', verbose=verbose, n_jobs=1)

    # Обучение модели на тренировочных данных
    grid_search.fit(X_train, y_train)

    # Получение лучших параметров
    print("Лучшие параметры: ", grid_search.best_params_)

    # Предсказание на тестовых данных
    y_pred = grid_search.predict(X_test)

    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели с подобранными параметрами: {accuracy * 100:.2f}%")
    return grid_search.best_estimator_


def save_model(model_filename, estimator):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)
    # Проверка существования файла перед сохранением
    if not os.path.exists(model_path):
        joblib.dump(estimator, model_path)
        print(f"Модель сохранена как '{model_filename}'")
    else:
        print(f"Модель уже существует: '{model_filename}'")


def load_and_use_model(X, y, model_filename='best_mlp_model_iris.pkl'):
    # Загрузка сохраненной модели
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)

    # Проверка существования файла модели перед загрузкой
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Модель загружена: '{model_filename}'")

        # Применение модели к новым данным
        predictions = model.predict(X)
        results_df = pd.DataFrame({'Предсказания': predictions, 'Истинные значения': y})
        print(results_df)
        return predictions
    else:
        print(f"Модель не найдена по пути: '{model_filename}'")
        return None


def iris_task():
    filename = 'best_mlp_model_iris.pkl'
    iris = load_iris()
    X_data, y_data = iris.data, iris.target
    iris_sample(X_data, y_data)
    estimator = iris_sample_with_grid_search(X_data, y_data)
    save_model(model_filename=filename, estimator=estimator)
    # Что не так?!...
    load_and_use_model(X_data[:5], y_data[:5])


def wine_task():
    filename = 'best_mlp_model_wine.pkl'
    filename_scaled = 'best_mlp_model_wine_sc.pkl'
    wine_quality = fetch_openml(name='wine-quality-white', version=1)
    X_data = wine_quality.data
    y_data = wine_quality.target.astype(int)
    scaler_wine = StandardScaler()

    X_data_sc = scaler_wine.fit_transform(X_data.astype(float))

    estimator = iris_sample_with_grid_search(X_data, y_data, verbose=2)
    save_model(model_filename=filename, estimator=estimator)

    estimator = iris_sample_with_grid_search(X_data_sc, y_data, verbose=2)
    save_model(model_filename=filename_scaled, estimator=estimator)

    # Для модели без скейла
    load_and_use_model(X_data[:5], y_data[:5], model_filename=filename)
    load_and_use_model(X_data_sc[:5], y_data[:5], model_filename=filename_scaled)


wine_task()
