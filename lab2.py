from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_features = 10
n = 800
# Создаем группу графиков 4 на 3
figure = plt.figure(1, figsize=(16, 9))

# Создаем 4 графика
axis = figure.subplots(1, 4)

# Генерируем исходные данные: 750 строк-наблюдений и 10 столбцов-признаков
np.random.seed(0)
size = n
X = np.random.uniform(0, 1, (size, n_features))

# Задаем функцию-выход: регрессионную проблему Фридмана
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] ** 5 + np.random.normal(0, 1))

# Добавляем зависимость признаков
X[:, n_features-4:] = X[:, :4] + np.random.normal(0, .025, (size, 4))


# Функция для преобразования вывода оценок к словарю
def rank_to_dict(r, tags):
    r = np.abs(r)
    minmax = MinMaxScaler()
    r = minmax.fit_transform(np.array(r).reshape(n_features, 1)).ravel()
    r = map(lambda x: round(x, 2), r)
    return dict(zip(tags, r))


# Добавляет данные на графики
def add_scatter(k, v, i):
    # График данных по каждой модели
    pred = lambda x: "x" + str(x + 1)
    axis[i].bar(list(pred(i) for i in range(n_features)), list(v.values()), label=k)
    axis[i].set_title(k)


# Гребневая модель
ridge = Ridge()
ridge.fit(X, Y)

# Случайное Лассо
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)

# Рекурсивное сокращение признаков
# Создаю классификатор для оценки важности признаков
rfe = RFE(estimator=LinearRegression(),
          n_features_to_select=4)  # сюда еще можно засунуть n_features_to_select=n - сохраняем
# определенное количество признаков
rfe.fit(X, Y)

# Создаем вывод
names = ["x%s" % i for i in range(1, n_features + 1)]

rfe_res = rfe.ranking_
# Приводим значения RFE модели к диапазону (0, 1)
for i in range(rfe_res.size):
    rfe_res[i] = n_features - rfe_res[i]

ranks = {"Ridge": rank_to_dict(ridge.coef_, names), "Lasso": rank_to_dict(lasso.coef_, names),
         "RFE": rank_to_dict(rfe.ranking_, names)}

# Создаем пустой список для данных
mean = {}

# «Бежим» по списку ranks
for key, value in ranks.items():
    # «Пробегаемся» по списку значений ranks, которые являются парой имя:оценка
    for item in value.items():
        # имя будет ключом для нашего mean
        # если элемента с текущим ключем в mean нет - добавляем
        if item[0] not in mean:
            mean[item[0]] = 0
            # суммируем значения по каждому ключу-имени признака
            mean[item[0]] += item[1]

# Находим среднее по каждому признаку
for key, value in mean.items():
    res = value / len(ranks)
    mean[key] = round(res, 2)

# Сортируем и распечатываем список
mean = dict(sorted(mean.items(), key=lambda y: y[1], reverse=True))

ranks["Mean"] = mean

for key, value in ranks.items():
    ranks[key] = dict(sorted(value.items(), key=lambda y: y[1], reverse=True))

# Создаем DataFrame из результатов ранжирования
df = pd.DataFrame(ranks)

# Выводим результаты на экран
print("Ранжирование признаков:")
print(df)

# Визуализируем результаты
i = 0
for key, value in ranks.items():
    add_scatter(key, value, i)
    i += 1

plt.show()
