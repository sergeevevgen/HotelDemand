# Лабораторная работа 6. Нейронная сеть
## Задание
Использовать нейронную сеть по варианту (24 % 2 == 0) для данных из таблицы, самостоятельно  сформулировав  задачу. 
Оценить, насколько хорошо она подходит для решения сформулированной мной задачи.

Ссылка на мой датасет: https://www.kaggle.com/datasets/mlardi/hotel-booking-demand-with-economic-indicators
## Задача
Прогнозирование среднего дохода отеля (adr) на основе различных 
экономических и операционных показателей, используя многослойный персептрон (MLPRegressor)
### Запуск программы
Файл lab6.py содержит и запускает программу.

### Описание программы
Программа состоит из двух частей:
1. Она считывает файл с данными о двух отелях: City Hotel и Resort Hotel. Содержит множество различных метрик
2. Далее определяет необходимые признаки для характеристики дохода
3. Обучает нейронную сеть и выводит степень ошибок по различным метрикам
### Результаты тестирования
По результатам тестирования, можно сказать следующее:

Вывод:
* Среднеквадратичная ошибка (MSE): 29.84% - показывает самое большое отклонение, но является неплохим результатом, 
хоть и может быть оптимизировано лучше
* Среднеабсолютное отклонение (MAE): 21.11% - показывает средний уровень отклонения, но является неплохим результатом
* Коэффициент детерминации (R^2): 59.650000000000006% - показывает неплохой уровень изменчивости целевой переменной, 
которое может быть объяснено моделью, но может быть улучшено

Результаты показывают, что модель может быть еще больше улучшена, 
так как имеется неплохой уровень ошибки, но всё же эта модель показывает лучший результат, чем модель регрессии


