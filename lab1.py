import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_classification

# Просто рандомное число для генерации одних и тех же данных
rs = 10

# Создаем данные с определенными параметрами
# n_samples - количество объектов
# n_features - количество признаков
# n_redundant - количество ненужных признаков
# n_informative - количество информативных признаков, которые учитываются (начиная с первого признака)
# random_state - рандомное число для генерации одних и тех же данных
# n_clusters_per_class - количество кластеров на класс
# X - матрица признаков (объекты - строки), y - целевая переменная для предсказывания
X, y = make_classification(n_samples=500,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           random_state=rs,
                           n_clusters_per_class=1)

# Стандартизируем данные
X = StandardScaler().fit_transform(X)

# Разделяем наши данные на тестовые и тренировочные
# test_size - % тренировочных
# random_state - рандомное число для того, чтобы брать всегда определенные объекты
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Создаем рандомный генератор
rng = np.random.RandomState(2)

# Добавляем "рандом" в данные
X += 2 * rng.uniform(size=X.shape)
linearly_dataset = (X, y)

# Создаем группу графиков 4 на 3
figure = plt.figure(1, figsize=(16, 9))

# Создаем n графиков
axis = figure.subplots(4)

# Лист цветов
cm = ListedColormap(['#5b3655', "#18d1e4"])

# Переменная для ошибок регрессий
errors = []

# Функция для выполнения всех регрессий (Линейной, полиномиальной, гребневой полиномиальной)
def make_regression(model):
    # Тренируем
    model.fit(X_train, y_train)
    # Проверяем
    model = model.predict(X_test)
    # Вычисляем ошибку
    errors.append(mean_squared_error(y_test, model))
    return model


# Добавляет данные на графики
def add_scatter(label, data, i):
    # График данных по каждой модели
    axis[i].scatter(X_test[:, 0], X_test[:, 1], c=data, cmap=cm)
    axis[i].set_title(label)
    axis[i].set_xlabel('X')
    axis[i].set_ylabel('Y')


# Получаем данные и добавляем для каждого график
results = {add_scatter('Начальные', y_test, 0),
           add_scatter('Линейная регрессия', make_regression(LinearRegression()), 1),
           add_scatter('Полиномиальная регрессия',
                       make_regression(make_pipeline(PolynomialFeatures(degree=3), LinearRegression())), 2),
           add_scatter('Гребневая полиномиальная регрессия',
                       make_regression(make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))), 3)
           }

# Добавляем расстояние между графиками
figure.subplots_adjust(hspace=0.5)
plt.show()

# Сравнение качества регрессий
print('Линейная - средняя ошибка', errors[0] * 100, ' %')
print('Полиномиальная (степень=3) - средняя ошибка', errors[1] * 100, ' %')
print('Гребневая (степень=3, alpha=1.0) - средняя ошибка', errors[2] * 100, ' %')
